# -*- coding: utf-8 -*-
# Dual-fisheye to 360-photo conversion tool
# Supports equirectangular and cubemap output formats
#
# Usage instructions:
#   python fisheye.py'
#     Start interactive alignment GUI.
#   python fisheye.py -help
#     Print this help message.
#   python fisheye.py lens.cfg in1.jpg in2.jpg gui
#     Launch interactive GUI with specified default options
#   python fisheye.py lens.cfg in1.jpg in2.jpg rect=out.png
#     Render and save equirectangular panorama using specified
#     lens configuration and source images.'
#   python fisheye.py lens.cfg in1.jpg in2.jpg cube=out.png
#     Render and save cubemap panorama using specified
#     lens configuration and source images.
#
# Copyright (c) 2016 Alexander C. Utter
# 2020 Du Tongxin

import json
import numpy as np
import tkinter as tk
import tkinter.filedialog as tkFileDialog
import ctypes
import sys
import traceback
from copy import deepcopy
from math import pi
from PIL import Image, ImageTk
from scipy.optimize import minimize
from threading import Thread
from FisheyeClasses import *
from GUIClasses import *

def launch_tk_gui(flens='', fimg1='', fimg2=''):
    # Create TK root object and GUI window.
    root = tk.Tk()
    gui = PanoramaGUI(root)
    # Load parameters if specified.
    if flens is not None and len(flens) > 0:
        gui.load_config(flens)
    if fimg1 is not None and len(fimg1) > 0:
        gui.img1.set(fimg1)
    if fimg2 is not None and len(fimg2) > 0:
        gui.img2.set(fimg2)
    # Start main loop.
    root.mainloop()

if __name__ == "__main__":
    # If we have exactly four arguments, run command-line version.
    if len(sys.argv) == 5 and sys.argv[4].startswith('gui'):
        # Special case for interactive mode.
        launch_tk_gui(sys.argv[1], sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 5:
        # First argument is the lens alignment file.
        lens1 = FisheyeLens()
        lens2 = FisheyeLens()
        cfg = open(sys.argv[1], 'r')
        load_config(cfg, lens1, lens2)
        # Second and third arguments are the source files.
        img1 = FisheyeImage(sys.argv[2], lens1)
        img2 = FisheyeImage(sys.argv[3], lens2)
        # Fourth argument is the mode and output filename.
        if sys.argv[4].startswith('cube='):
            out = sys.argv[5:]
            pan = PanoramaImage((img1, img2))
            pan.render_cubemap(1024).save(out)
        elif sys.argv[4].startswith('rect='):
            out = sys.argv[5:]
            pan = PanoramaImage((img1, img2))
            pan.render_equirectangular(1024).save(out)
        else:
            print('Unrecognized render mode (cube=, rect=, gui)')
    elif len(sys.argv) > 1:
        # If requested, print command-line usage information.
        print('Usage instructions:')
        print('  python fisheye.py')
        print('    Start interactive alignment GUI.')
        print('  python fisheye.py -help')
        print('    Print this help message.')
        print('  python fisheye.py lens.cfg in1.jpg in2.jpg gui')
        print('    Launch interactive GUI with specified default options')
        print('  python fisheye.py lens.cfg in1.jpg in2.jpg rect=out.png')
        print('    Render and save equirectangular panorama using specified')
        print('    lens configuration and source images.')
        print('  python fisheye.py lens.cfg in1.jpg in2.jpg cube=out.png')
        print('    Render and save cubemap panorama using specified')
        print('    lens configuration and source images.')
    else:
        # Otherwise, start the interactive GUI with all fields blank.
        launch_tk_gui()
