from FisheyeClasses import *
import json
import numpy as np
import tkinter as tk
import tkinter.filedialog as tkFileDialog
from tkinter.messagebox import *
import ctypes
import sys
import traceback
from copy import deepcopy
from math import pi
from PIL import Image, ImageTk
from scipy.optimize import minimize
from threading import Thread
from mathFunction import *


# Tkinter GUI window for loading a fisheye image.
class FisheyeAlignmentGUI:
    def __init__(self, parent, src_file, lens):
        # Set flag once all window objects created.
        self.init_done = False
        # Final result is the lens object.
        self.lens = lens
        # Load the input file.
        self.img = Image.open(src_file)
        # Create frame for this window with two vertical panels...
        parent.wm_title('Fisheye Alignment')
        self.frame = tk.Frame(parent)
        self.controls = tk.Frame(self.frame)
        # Make sliders for adjusting the lens parameters quaternion.
        self.x = self._make_slider(self.controls, 0, 'Center-X (px)',
                                   lens.get_x(), self.img.size[0])
        self.y = self._make_slider(self.controls, 1, 'Center-Y (px)',
                                   lens.get_y(), self.img.size[1])
        self.r = self._make_slider(self.controls, 2, 'Radius (px)',
                                   lens.radius_px, self.img.size[0])
        self.f = self._make_slider(self.controls, 3, 'Field of view (deg)',
                                   lens.fov_deg, 240, res=0.1)
        # Create a frame for the preview image, which resizes based on the
        # outer frame but does not respond to the contained preview size.
        self.preview_frm = tk.Frame(self.frame)
        self.preview_frm.bind('<Configure>', self._update_callback)  # Update on resize
        # Create the canvas object for the preview image.
        self.preview = tk.Canvas(self.preview_frm)
        # Finish frame creation.
        self.controls.pack(side=tk.LEFT)
        self.preview.pack(fill=tk.BOTH, expand=1)
        self.preview_frm.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
        self.frame.pack(fill=tk.BOTH, expand=1)
        # Render the image once at default size
        self.init_done = True
        self.update_preview((800 ,800))
        # Disable further size propagation.
        self.preview_frm.update()
        self.preview_frm.pack_propagate(0)

    # Redraw the preview image using latest GUI parameters.
    def update_preview(self, psize):
        # Safety check: Ignore calls during construction/destruction.
        if not self.init_done: return
        # Copy latest user settings to the lens object.
        self.lens.fov_deg = self.f.get()
        self.lens.radius_px = self.r.get()
        self.lens.center_px[0] = self.x.get()
        self.lens.center_px[1] = self.y.get()
        # Re-scale the image to match the canvas size.
        # Note: Make a copy first, because thumbnail() operates in-place.
        self.img_sc = self.img.copy()
        self.img_sc.thumbnail(psize, Image.NEAREST)
        self.img_tk = ImageTk.PhotoImage(self.img_sc)
        # Re-scale the x/y/r parameters to match the preview scale.
        pre_scale = float(psize[0]) / float(self.img.size[0])
        x = self.x.get() * pre_scale
        y = self.y.get() * pre_scale
        r = self.r.get() * pre_scale
        # Clear and redraw the canvas.
        self.preview.delete('all')
        self.preview.create_image(0, 0, anchor=tk.NW, image=self.img_tk)
        self.preview.create_oval( x -r, y- r, x + r, y + r,
                                 outline='#C00000', width=3)

    # Make a combined label/textbox/slider for a given variable:
    def _make_slider(self, parent, rowidx, label, inival, maxval, res=0.5):
        # Create shared variable and set initial value.
        tkvar = tk.DoubleVar()
        tkvar.set(inival)
        # Set a callback for whenever tkvar is changed.
        # (The 'command' callback on the SpinBox only applies to the buttons.)
        tkvar.trace('w', self._update_callback)  # 变量被ui界面写入时调用回调函数
        # Create the Label, SpinBox, and Scale objects.
        label = tk.Label(parent, text=label)
        spbox = tk.Spinbox(parent,
                           textvariable=tkvar,
                           from_=0, to=maxval, increment=res)
        slide = tk.Scale(parent,
                         orient=tk.HORIZONTAL,
                         showvalue=0,
                         variable=tkvar,
                         from_=0, to=maxval, resolution=res)
        label.grid(row=rowidx, column=0)
        spbox.grid(row=rowidx, column=1)
        slide.grid(row=rowidx, column=2)
        return tkvar

    # Find the largest output size that fits within the given bounds and
    # matches the aspect ratio of the original source image.
    def _get_aspect_size(self, max_size):
        img_ratio = float(self.img.size[1]) / float(self.img.size[0])
        return (min(max_size[0], max_size[1] / img_ratio),
                min(max_size[1], max_size[0] * img_ratio))

    # Thin wrapper for update_preview(), used to strip Tkinter arguments.
    def _update_callback(self, *args):
        # Sanity check that initialization is completed:
        if not self.init_done: return
        # Determine the render size.  (Always 2:1 aspect ratio.)
        psize = self._get_aspect_size((self.preview_frm.winfo_width(),
                                       self.preview_frm.winfo_height()))
        # Render the preview at the given size.
        if psize[0] >= 10 and psize[1] >= 10:
            self.update_preview(psize)


# Tkinter GUI window for calibrating fisheye alignment.
class PanoramaAlignmentGUI:
    def __init__(self, parent, panorama, psize=512):
        self.init_done = False
        # Store source and preview size
        self.panorama = panorama
        # Create frame for this window with two vertical panels...
        parent.wm_title('Panorama Alignment')
        self.frame = tk.Frame(parent)
        self.controls = tk.Frame(self.frame)
        # Make a drop-menu to select the rendering mode.
        tk.Label(self.controls, text='Preview mode').grid(row=0, column=0, sticky=tk.W)
        self.mode = tk.StringVar()
        self.mode.set('align')
        self.mode.trace('w', self._update_callback)
        mode_list = self.panorama.get_render_modes()
        mode_drop = tk.OptionMenu(self.controls, self.mode, *mode_list)
        mode_drop.grid(row=0, column=1, columnspan=2, sticky='NESW')
        # Determine which axis marks the main 180 degree rotation.
        front_qq = panorama.sources[0].lens.center_qq
        back_qq = panorama.sources[1].lens.center_qq
        diff_qq = mul_qq(front_qq, back_qq)
        # Create the axis selection toggle. (Flip on Y or Z)
        self.flip_axis = tk.BooleanVar()
        self.flip_axis.trace('w', self._update_callback)
        if abs(diff_qq[2]) > abs(diff_qq[3]):
            self.flip_axis.set(False)
            flip_qq = [0, 0, 1, 0]
        else:
            self.flip_axis.set(True)
            flip_qq = [0, 0, 0, 1]
        tk.Label(self.controls, text='Flip axis').grid(row=1, column=0, sticky=tk.W)
        axis_chk = tk.Checkbutton(self.controls, variable=self.flip_axis)
        axis_chk.grid(row=1, column=1, columnspan=2, sticky='NESW')
        # Extract the (hopefully small) alignment offset.
        flip_conj = conj_qq(mul_qq(flip_qq, front_qq))
        align_qq = mul_qq(back_qq, flip_conj)
        # Make three sliders for adjusting the relative alignment.
        self.slide_rx = self._make_slider(self.controls, 2, 'Rotate X', front_qq[1])
        self.slide_ry = self._make_slider(self.controls, 3, 'Rotate Y', front_qq[2])
        self.slide_rz = self._make_slider(self.controls, 4, 'Rotate Z', front_qq[3])
        self.slide_ax = self._make_slider(self.controls, 5, 'Align X', align_qq[1])
        self.slide_ay = self._make_slider(self.controls, 6, 'Align Y', align_qq[2])
        self.slide_az = self._make_slider(self.controls, 7, 'Align Z', align_qq[3])
        # Finish control-frame creation.
        self.controls.pack(side=tk.LEFT)
        # Create a frame for the preview image, which resizes based on the
        # outer frame but does not respond to the contained preview size.
        self.preview_frm = tk.Frame(self.frame)
        self.preview_frm.bind('<Configure>', self._update_callback)  # Update on resize
        # Add the preview.
        self.preview_lbl = tk.Label(self.preview_frm)  # Label displays image
        self.preview_lbl.pack()
        self.preview_frm.pack(fill=tk.BOTH, expand=1)
        # Finish frame creation.
        self.frame.pack(fill=tk.BOTH, expand=1)
        # Render the image once at default size
        self.init_done = True
        self.update_preview(psize)
        # Disable further size propagation.
        self.preview_frm.update()
        self.preview_frm.pack_propagate(0)

    # Update the GUI preview using latest alignment parameters.
    def update_preview(self, psize):
        # Sanity check that initialization is completed:
        if not self.init_done: return
        # Determine the primary axis of rotation.
        if self.flip_axis.get():
            flip_qq = [0, 0, 0, 1]
        else:
            flip_qq = [0, 0, 1, 0]
        # Calculate the orientation of both lenses.
        front_qq = norm_qq(self.slide_rx.get(),
                           self.slide_ry.get(),
                           self.slide_rz.get())
        align_qq = norm_qq(self.slide_ax.get(),
                           self.slide_ay.get(),
                           self.slide_az.get())
        back_qq = mul_qq(align_qq, mul_qq(flip_qq, front_qq))
        self.panorama.sources[0].lens.center_qq = front_qq
        self.panorama.sources[1].lens.center_qq = back_qq
        # Render the preview.
        # Note: The Tk-Label doesn't maintain a reference to the image object.
        #       To avoid garbage-collection, keep one in this class.
        self.preview_img = ImageTk.PhotoImage(
            self.panorama.render_equirectangular(psize, self.mode.get()))
        # Assign the new icon.
        self.preview_lbl.configure(image=self.preview_img)

    # Find the largest output size that fits within the given bounds and
    # matches the 2:1 aspect ratio of the equirectangular preview.
    def _get_aspect_size(self, max_size):
        return (min(max_size[0], max_size[1] / 2),
                min(max_size[1], max_size[0] * 2))

    # Make a combined label/textbox/slider for a given variable:
    def _make_slider(self, parent, rowidx, label, inival):
        # Set limits and resolution.
        lim = 1.0
        res = 0.001
        # Create shared variable.
        tkvar = tk.DoubleVar()
        tkvar.set(inival)
        # Set a callback for whenever tkvar is changed.
        # (The 'command' callback on the SpinBox only applies to the buttons.)
        tkvar.trace('w', self._update_callback)
        # Create the Label, SpinBox, and Scale objects.
        label = tk.Label(parent, text=label)
        spbox = tk.Spinbox(parent,
                           textvariable=tkvar,
                           from_=-lim, to=lim, increment=res)
        slide = tk.Scale(parent,
                         orient=tk.HORIZONTAL,
                         showvalue=0,
                         variable=tkvar,
                         from_=-lim, to=lim, resolution=res)
        label.grid(row=rowidx, column=0, sticky='W')
        spbox.grid(row=rowidx, column=1)
        slide.grid(row=rowidx, column=2)
        return tkvar

    # Thin wrapper for update_preview(), used to strip Tkinter arguments.
    def _update_callback(self, *args):
        # Sanity check that initialization is completed:
        if not self.init_done: return
        # Determine the render size.  (Always 2:1 aspect ratio.)
        psize = min(self.preview_frm.winfo_width() / 2,
                    self.preview_frm.winfo_height())
        # Render the preview at the given size.
        # TODO: Fudge factor of -2 avoids infinite resize loop.
        #       Is there a better way?
        if psize >= 10:
            self.update_preview(psize - 2)


# Tkinter GUI window for end-to-end alignment and rendering.
class PanoramaGUI:
    def __init__(self, parent):
        # Store reference object for creating child dialogs.
        self.parent = parent
        self.win_lens1 = None
        self.win_lens2 = None
        self.win_align = None
        self.work_done = False
        self.work_error = None
        self.work_status = None
        # Create dummy lens configuration.
        self.lens1 = FisheyeLens()
        self.lens2 = FisheyeLens()
        self.lens2.center_qq = [0, 0, 1, 0]  # Default flip along Y axis.
        # Create frame for this GUI.
        parent.wm_title('Panorama Creation Tool')
        frame = tk.Frame(parent)
        # Make file-selection inputs for the two images.
        img_frame = tk.LabelFrame(frame, text='Input Images')
        self.img1 = self._make_file_select(img_frame, 0, 'Image #1')
        self.img2 = self._make_file_select(img_frame, 1, 'Image #2')
        img_frame.pack()
        # Make buttons to load, save, and adjust the lens configuration.
        lens_frame = tk.LabelFrame(frame, text='Lens Configuration and Alignment')
        btn_lens1 = tk.Button(lens_frame, text='Lens 1', command=self._adjust_lens1)
        btn_lens2 = tk.Button(lens_frame, text='Lens 2', command=self._adjust_lens2)
        btn_align = tk.Button(lens_frame, text='Align', command=self._adjust_align)
        btn_auto = tk.Button(lens_frame, text='Auto', command=self._auto_align_start)
        btn_load = tk.Button(lens_frame, text='Load', command=self.load_config)
        btn_save = tk.Button(lens_frame, text='Save', command=self.save_config)
        btn_lens1.grid(row=0, column=0, sticky='NESW')
        btn_lens2.grid(row=0, column=1, sticky='NESW')
        btn_align.grid(row=0, column=2, sticky='NESW')
        btn_auto.grid(row=0, column=3, sticky='NESW')
        btn_load.grid(row=1, column=0, columnspan=2, sticky='NESW')
        btn_save.grid(row=1, column=2, columnspan=2, sticky='NESW')
        lens_frame.pack(fill=tk.BOTH)
        # Buttons to render the final output in different modes.
        out_frame = tk.LabelFrame(frame, text='Final output rendering')
        btn_rect = tk.Button(out_frame, text='Equirectangular',
                             command=self._render_rect)
        btn_cube = tk.Button(out_frame, text='Cubemap',
                             command=self._render_cube)
        btn_rect.pack(fill=tk.BOTH)
        btn_cube.pack(fill=tk.BOTH)
        out_frame.pack(fill=tk.BOTH)
        # Status indicator box.
        self.status = tk.Label(frame, relief=tk.SUNKEN,
                               text='Select input images to begin.')
        self.status.pack(fill=tk.BOTH)
        # Finish frame creation.
        frame.pack()

    # Helper function to destroy an object.
    def _destroy(self, obj):
        if obj is not None:
            obj.destroy()

    # Popup dialogs for each alignment step.
    def _adjust_lens1(self):
        self._destroy(self.win_lens1)
        try:
            self.win_lens1 = tk.Toplevel(self.parent)
            FisheyeAlignmentGUI(self.win_lens1, self.img1.get(), self.lens1)
        except IOError:
            self._destroy(self.win_lens1)
            ctypes.windll.user32.MessageBoxA(0, 'Unable to read image file #1'.encode('gbk'), 'wips', 1)
            # tkMessageBox.showerror('Error', 'Unable to read image file #1.')
        except:
            self._destroy(self.win_lens1)
            tkMessageBox.showerror('Dialog creation error', traceback.format_exc())

    def _adjust_lens2(self):
        self._destroy(self.win_lens2)
        try:
            self.win_lens2 = tk.Toplevel(self.parent)
            FisheyeAlignmentGUI(self.win_lens2, self.img2.get(), self.lens2)
        except IOError:
            self._destroy(self.win_lens2)
            tkMessageBox.showerror('Error', 'Unable to read image file #2.')
        except:
            self._destroy(self.win_lens2)
            tkMessageBox.showerror('Dialog creation error', traceback.format_exc())

    def _adjust_align(self):
        self._destroy(self.win_align)
        try:
            pan = self._create_panorama()
            self.win_align = tk.Toplevel(self.parent)
            PanoramaAlignmentGUI(self.win_align, pan)
        except:
            self._destroy(self.win_align)
            print(traceback.format_exc())
            #tkMessageBox.showerror('Dialog creation error', traceback.format_exc())

    # Automatic alignment.
    # Use worker thread, because this may take a while.
    def _auto_align_start(self):
        try:
            # Create panorama object from within GUI thread, since it depends
            # on Tk variables which are NOT thread-safe.
            pan = self._create_panorama()
            # Display status message and display hourglass...
            self._set_status('Starting auto-alignment...', 'wait')
            # Create a new worker thread.
            work = Thread(target=self._auto_align_work, args=[pan])
            work.start()
            # Set a timer to periodically check for completion.
            self.parent.after(200, self._auto_align_timer)
        except:
            tkMessageBox.showerror('Auto-alignment error', traceback.format_exc())

    def _auto_align_work(self, pan):
        try:
            # Repeat alignment at progressively higher resolution.
            self._auto_align_step(pan, 16, 128, 'Stage 1/4')
            self._auto_align_step(pan, 8, 128, 'Stage 2/4')
            self._auto_align_step(pan, 4, 192, 'Stage 3/4')
            self._auto_align_step(pan, 2, 256, 'Stage 4/4')
            # Signal success!
            self.work_status = 'Auto-alignment completed.'
            self.work_error = None
            self.work_done = True
        except:
            # Signal error.
            self.work_status = 'Auto-alignment failed.'
            self.work_error = traceback.format_exc()
            self.work_done = True

    def _auto_align_step(self, pan, scale, psize, label):
        # Update status message.
        self.work_status = 'Auto-alignment: ' + str(label)
        # Create a panorama object at 1/scale times original resolution.
        pan_sc = deepcopy(pan)
        pan_sc.downsample(scale)
        # Run optimization, rendering each hypothesis at the given resolution.
        pan_sc.optimize(psize)
        # Update local lens parameters.
        # Note: These are not Tk variables, so are safe to change.
        self.lens1 = pan_sc.scale_lens(0, scale)
        self.lens2 = pan_sc.scale_lens(1, scale)

    # Timer callback object checks outputs from worker thread.
    # (Tkinter objects are NOT thread safe.)
    def _auto_align_timer(self, *args):
        # Check thread status.
        if self.work_done:
            # Update status message, with popup on error.
            if self.work_status is not None:
                self._set_status(self.work_status)
            if self.work_error is not None:
                self._set_status('Auto-alignment failed.')
                tkMessageBox.showerror('Auto-alignment error', self.work_error)
            # Clear the 'done' flag for future runs.
            self.work_done = False
        else:
            # Update status message and keep hourglass.
            if self.work_status is not None:
                self._set_status(self.work_status, 'wait')
            # Reset timer to be called again.
            self.parent.after(200, self._auto_align_timer)

    # Create panorama object using current settings.
    def _create_panorama(self):
        img1 = FisheyeImage(self.img1.get(), self.lens1)
        img2 = FisheyeImage(self.img2.get(), self.lens2)
        return PanoramaImage((img1, img2))

    # Load or save lens configuration and alignment.
    def load_config(self, filename=None):
        if filename is None:
            file_obj = tkFileDialog.askopenfile()
            if file_obj is None: return
        else:
            file_obj = open(filename, 'r')
        try:
            load_config(file_obj, self.lens1, self.lens2)
        except:
            showinfo(title='Config load error', message=traceback.format_exc())
            #tkMessageBox.showerror('Config load error', traceback.format_exc())

    def save_config(self, filename=None):
        if filename is None:
            file_obj = tkFileDialog.asksaveasfile()
            if file_obj is None: return
        else:
            file_obj = open(filename, 'w')
        try:
            save_config(file_obj, self.lens1, self.lens2)
        except:
            tkMessageBox.showerror('Config save error', traceback.format_exc())

    # Render and save output in various modes.
    def _render_generic(self, render_type, render_size=1024):
        # Popup asks user for output file.
        file_obj = tkFileDialog.asksaveasfile(mode='wb')
        # Abort if user clicks 'cancel'.
        if file_obj is None: return
        # Proceed with rendering...
        self._set_status('Rendering image: ' + file_obj.name, 'wait')
        try:
            panorama = self._create_panorama()
            render_func = getattr(panorama, render_type)  # 获取对象的属性值，这个属性值可以存在变量里
            render_func(render_size).save(file_obj)  # 使用Image对象的save功能
            self._set_status('Done!')
        except:
            ctypes.windll.user32.MessageBoxA(0, 'Render error'.encode('gbk'), 'wips', 1)
            # tkMessageBox.showerror('Render error', traceback.format_exc())
            self._set_status('Render failed.')

    def _render_rect(self):
        self._render_generic('render_equirectangular')

    def _render_cube(self):
        self._render_generic('render_cubemap')

    # Callback to create a file-selection popup.
    def _file_select(self, tkstr):
        result = tkFileDialog.askopenfile()
        if result is not None:
            tkstr.set(result.name)
            result.close()

    # Make a combined label/textbox/slider for a given variable:
    def _make_file_select(self, parent, rowidx, label):
        # Create string variable.
        tkstr = tk.StringVar()
        # Create callback event handler.
        cmd = lambda: self._file_select(tkstr)
        # Create the Label, Entry, and Button objects.
        label = tk.Label(parent, text=label)
        entry = tk.Entry(parent, textvariable=tkstr)
        button = tk.Button(parent, text='...', command=cmd)
        label.grid(row=rowidx, column=0, sticky='W')
        entry.grid(row=rowidx, column=1)
        button.grid(row=rowidx, column=2)
        return tkstr

    # Set status text, and optionally update cursor.
    def _set_status(self, status, cursor='arrow'):
        self.parent.config(cursor=cursor)
        self.status.configure(text=status)