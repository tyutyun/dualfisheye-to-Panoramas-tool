
# Parameters for a fisheye lens, including its orientation.
import json
from copy import deepcopy
from math import pi
from PIL import Image, ImageTk
from scipy.optimize import minimize


from mathFunction import *

class FisheyeLens:
    def __init__(self, rows=1024, cols=1024):
        # Fisheye lens parameters.
        self.fov_deg = 180
        self.radius_px = min(rows,cols) / 2
        # Pixel coordinates of the optical axis (X,Y).
        self.center_px = np.matrix([[cols/2], [rows/2]])
        # Quaternion mapping intended to actual optical axis.
        self.center_qq = [1, 0, 0, 0]

    def downsample(self, dsamp):
        self.radius_px /= dsamp
        self.center_px /= dsamp

    def get_x(self):
        return np.asscalar(self.center_px[0])

    def get_y(self):
        return np.asscalar(self.center_px[1])

    def to_dict(self):
        return {'cx':self.get_x(),
                'cy':self.get_y(),
                'cr':self.radius_px,
                'cf':self.fov_deg,
                'qw':self.center_qq[0],
                'qx':self.center_qq[1],
                'qy':self.center_qq[2],
                'qz':self.center_qq[3]}

    def from_dict(self, data):
        self.center_px[0] = data['cx']
        self.center_px[1] = data['cy']
        self.radius_px    = data['cr']
        self.fov_deg      = data['cf']
        self.center_qq[0] = data['qw']
        self.center_qq[1] = data['qx']
        self.center_qq[2] = data['qy']
        self.center_qq[3] = data['qz']

# Load or save lens configuration and alignment.
def load_config(file_obj, lens1, lens2):
    [data1, data2] = json.load(file_obj)
    lens1.from_dict(data1)
    lens2.from_dict(data2)

def save_config(file_obj, lens1, lens2):
    data = [lens1.to_dict(), lens2.to_dict()]
    json.dump(data, file_obj, indent=2, sort_keys=True)


# Fisheye source image, with lens and rotation parameters.
# Contains functions for extracting pixel data given direction vectors.
class FisheyeImage:
    # Load image file and set default parameters
    def __init__(self, src_file, lens=None):
        # Load the image file, and convert to a numpy matrix.
        self._update_img(Image.open(src_file))
        # Set lens parameters.
        if lens is None:
            self.lens = FisheyeLens(self.rows, self.cols)
        else:
            self.lens = lens

    # Update image matrix and corresponding size variables.
    def _update_img(self, img):
        self.img = np.array(img)
        self.rows = self.img.shape[0]
        self.cols = self.img.shape[1]
        self.clrs = self.img.shape[2]

    # Shrink source image and adjust lens accordingly.
    def downsample(self, dsamp):
        # Adjust lens parameters.
        self.lens.downsample(dsamp)
        # Determine the new image dimensions.
        # Note: PIL uses cols, rows whereas numpy uses rows, cols
        shape = (self.img.shape[1] / dsamp,     # Cols
                 self.img.shape[0] / dsamp)     # Rows
        # Convert matrix back to PIL Image and resample.
        img = Image.fromarray(self.img)
        img.thumbnail(shape, Image.BICUBIC)
        # Convert back and update size.
        self._update_img(img)

    # Given an 3xN array of "XYZ" vectors in panorama space (+X = Front),
    # convert each ray to 2xN coordinates in "UV" fisheye image space.
    def get_uv(self, xyz_vec):
        # Extract lens parameters of interest.
        fov_rad = self.lens.fov_deg * pi / 180
        fov_scale = np.float32(2 * self.lens.radius_px / fov_rad)
        # Normalize the input vector and rotate to match lens reference axes.
        xyz_rot = get_rotation_matrix(self.lens.center_qq) * matrix_norm(xyz_vec)
        # Convert to polar coordinates relative to lens boresight.
        # (In lens coordinates, unit vector's X axis gives boresight angle;
        #  normalize Y/Z to get a planar unit vector for the bearing.)
        # Note: Image +Y maps to 3D +Y, and image +X maps to 3D +Z.
        theta_rad = np.arccos(xyz_rot[0,:])
        proj_vec = matrix_norm(np.concatenate((xyz_rot[2,:], xyz_rot[1,:])))
        # Fisheye lens maps 3D angle to focal-plane radius.
        # TODO: Do we need a better model for lens distortion?
        rad_px = theta_rad * fov_scale
        # Convert back to focal-plane rectangular coordinates.
        uv = np.multiply(rad_px, proj_vec) + self.lens.center_px
        return np.asarray(uv + 0.5, dtype=int)

    # Given an 2xN array of UV pixel coordinates, check if each pixel is
    # within the fisheye field of view. Returns N-element boolean mask.
    def get_mask(self, uv_px):
        # Check whether each coordinate is within outer image bounds,
        # and within the illuminated area under the fisheye lens.
        x_mask = np.logical_and(0 <= uv_px[0], uv_px[0] < self.cols)
        y_mask = np.logical_and(0 <= uv_px[1], uv_px[1] < self.rows)
        # Check whether each coordinate is within the illuminated area.
        r_mask = matrix_len(uv_px - self.lens.center_px) < self.lens.radius_px
        # All three checks must pass to be considered visible.
        all_mask = np.logical_and(r_mask, np.logical_and(x_mask, y_mask))
        return np.squeeze(np.asarray(all_mask))

    # Given an 2xN array of UV pixel coordinates, return a weight score
    # that is proportional to the distance from the edge.
    def get_weight(self, uv_px):
        mm = self.get_mask(uv_px)
        rr = self.lens.radius_px - matrix_len(uv_px - self.lens.center_px)
        rr[~mm] = 0
        return rr

    # Given a 2xN array of UV pixel coordinates, return the value of each
    # corresponding pixel. Output format is Nx1 (grayscale) or Nx3 (color).
    # Pixels outside the fisheye's field of view are pure black (0) or (0,0,0).
    def get_pixels(self, uv_px):
        # Create output array with default pixel values.
        pcount = uv_px.shape[1]
        result = np.zeros((pcount, self.clrs), dtype=self.img.dtype)
        # Overwrite in-bounds pixels as specified above.
        self.add_pixels(uv_px, result)
        return result

    # Given a 2xN array of UV pixel coordinates, write the value of each
    # corresponding pixel to the linearized input/output image (Nx3).
    # Several weighting modes are available.
    def add_pixels(self, uv_px, img1d, weight=None):
        # Lookup row & column for each in-bounds coordinate.
        mask = self.get_mask(uv_px)
        xx = uv_px[0,mask]
        yy = uv_px[1,mask]
        # Update matrix according to assigned weight.
        if weight is None:
            img1d[mask] = self.img[yy,xx]
        elif np.isscalar(weight):
            img1d[mask] += self.img[yy,xx] * weight
        else:
            w1 = np.asmatrix(weight, dtype='float32')
            w3 = w1.transpose() * np.ones((1,3))
            img1d[mask] += np.multiply(self.img[yy,xx], w3[mask])


# A panorama image made from several FisheyeImage sources.
# TODO: Add support for supersampled anti-aliasing filters.
class PanoramaImage:
    def __init__(self, src_list):
        self.debug = True
        self.sources = src_list
        self.dtype = self.sources[0].img.dtype
        self.clrs = self.sources[0].clrs

    # Downsample each source image.
    def downsample(self, dsamp):
        for src in self.sources:
            src.downsample(dsamp)

    # Return a list of 'mode' strings suitable for render_xx() methods.
    def get_render_modes(self):
        return ['overwrite', 'align', 'blend']

    # Retrieve a scaled copy of lens parameters for the Nth source.
    def scale_lens(self, idx, scale=None):
        temp = deepcopy(self.sources[idx].lens)
        temp.downsample(1.0 / scale)
        return temp

    # Using current settings as an initial guess, use an iterative optimizer
    # to better align the source images.  Adjusts FOV of each lens, as well
    # as the rotation quaternions for all lenses except the first.
    # TODO: Implement a higher-order loop that iterates this step with
    #       progressively higher resolution.  (See also: create_panorama)
    # TODO: Find a better scoring heuristic.  Present solution always
    #       converges on either FOV=0 or FOV=9999, depending on wt_pixel.
    def optimize(self, psize=256, wt_pixel=1000, wt_blank=1000):
        # Precalculate raster-order XYZ coordinates at given resolution.
        [xyz, rows, cols] = self._get_equirectangular_raster(psize)
        # Scoring function gives bonus points per overlapping pixel.
        score = lambda svec: self._score(svec, xyz, wt_pixel, wt_blank)
        # Multivariable optimization using gradient-descent or similar.
        # https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html
        svec0 = self._get_state_vector()
        final = minimize(score, svec0, method='Nelder-Mead',
                         options={'xtol':1e-4, 'disp':True})
        # Store final lens parameters.
        self._set_state_vector(final.x)

    # Render combined panorama in equirectangular projection mode.
    # See also: https://en.wikipedia.org/wiki/Equirectangular_projection
    def render_equirectangular(self, out_size, mode='blend'):
        # Render the entire output in a single pass.
        [xyz, rows, cols] = self._get_equirectangular_raster(out_size)
        return Image.fromarray(self._render(xyz, rows, cols, mode))

    # Render combined panorama in cubemap projection mode.
    # See also: https://en.wikipedia.org/wiki/Cube_mapping
    def render_cubemap(self, out_size, mode='blend'):
        # Create coordinate arrays.
        cvec = np.arange(out_size, dtype='float32') - out_size/2        # Coordinate range [-S/2, S/2)
        vec0 = np.ones(out_size*out_size, dtype='float32') * out_size/2 # Constant vector +S/2
        #将数组展开，每一个元素重复out_size次
        vec1 = np.repeat(cvec, out_size)                                # Increment every N steps
        vec2 = np.tile(cvec, out_size)                                  # Sweep N times
        # Create XYZ coordinate vectors and render each cubemap face.
        render = lambda xyz: self._render(xyz, out_size, out_size, mode)
        xm = render(np.matrix([-vec0, vec1, vec2]))     # -X face
        xp = render(np.matrix([vec0, vec1, -vec2]))     # +X face
        ym = render(np.matrix([-vec1, -vec0, vec2]))    # -Y face
        yp = render(np.matrix([vec1, vec0, vec2]))      # +Y face
        zm = render(np.matrix([-vec2, vec1, -vec0]))    # -Z face
        zp = render(np.matrix([vec2, vec1, vec0]))      # +Z face
        # Concatenate the individual faces in canonical order:
        # https://en.wikipedia.org/wiki/Cube_mapping#Memory_Addressing
        img_mat = np.concatenate([zp, zm, ym, yp, xm, xp], axis=0)
        return Image.fromarray(img_mat)

    # Get XYZ vectors for an equirectangular render, in raster order.
    # (Each row left to right, with rows concatenates from top to bottom.)
    def _get_equirectangular_raster(self, out_size):
        # Set image size (2x1 aspect ratio)
        rows = int(out_size)
        cols = int(2*out_size)
        # Calculate longitude of each column.
        theta_x = np.linspace(-pi, pi, cols, endpoint=False, dtype='float32')
        cos_x = np.cos(theta_x).reshape(1,cols)
        sin_x = np.sin(theta_x).reshape(1,cols)
        # Calculate lattitude of each row.
        ystep = pi / rows
        theta_y = np.linspace(-pi/2 + ystep/2, pi/2 - ystep/2, rows, dtype='float32')
        cos_y = np.cos(theta_y).reshape(rows,1)
        sin_y = np.sin(theta_y).reshape(rows,1)
        # Calculate X, Y, and Z coordinates for each output pixel.
        x = cos_y * cos_x
        y = sin_y * np.ones((1,cols), dtype='float32')
        z = cos_y * sin_x
        # Vectorize the coordinates in raster order.
        xyz = np.matrix([x.ravel(), y.ravel(), z.ravel()])
        return [xyz, rows, cols]

    # Convert all lens parameters to a state vector. See also: optimize()
    def _get_state_vector(self):
        nsrc = len(self.sources)
        assert nsrc > 0
        #第一个镜头只有一个视场角参数，后面镜头都是四个参数，包括视场角与三个旋转角
        svec = np.zeros(4*nsrc - 3)
        # First lens: Only the FOV is stored.
        svec[0] = self.sources[0].lens.fov_deg - 180
        # All other lenses: Store FOV and quaternion parameters.
        for n in range(1, nsrc):
            svec[4*n-3] = self.sources[n].lens.fov_deg - 180
            svec[4*n-2] = self.sources[n].lens.center_qq[1]
            svec[4*n-1] = self.sources[n].lens.center_qq[2]
            svec[4*n-0] = self.sources[n].lens.center_qq[3]
        return svec

    # Update lens parameters based on state vector.  See also: optimize()
    def _set_state_vector(self, svec):
        # Sanity check on input vector.
        nsrc = len(self.sources)
        assert len(svec) == (4*nsrc - 3)
        # First lens: Only the FOV is changed.
        self.sources[0].lens.fov_deg = svec[0] + 180
        # All other lenses: Update FOV and quaternion parameters.
        for n in range(1, nsrc):
            self.sources[n].lens.fov_deg = svec[4*n-3] + 180
            self.sources[n].lens.center_qq[1] = svec[4*n-2]
            self.sources[n].lens.center_qq[2] = svec[4*n-1]
            self.sources[n].lens.center_qq[3] = svec[4*n-0]

    # Add pixels from every source to form a complete output image.
    # Several blending modes are available. See also: get_render_modes()
    def _render(self, xyz, rows, cols, mode):
        # Allocate Nx3 or Nx1 "1D" pixel-list (raster-order).
        img1d = np.zeros((rows*cols, self.clrs), dtype='float32')
        # Determine rendering mode:
        if mode == 'overwrite':
            # Simplest mode: Draw first, then blindly overwrite second.
            for src in self.sources:
                uv = src.get_uv(xyz)
                src.add_pixels(uv, img1d)
        elif mode == 'align':
            # Alignment mode: Draw each one at 50% intensity.
            for src in self.sources:
                uv = src.get_uv(xyz)
                src.add_pixels(uv, img1d, 0.5)
        elif mode == 'blend':
            # Linear nearest-source blending.
            uv_list = []
            wt_list = []
            wt_total = np.zeros(rows*cols, dtype='float32')
            # Calculate per-image and total weight matrices.
            for src in self.sources:
                uv = src.get_uv(xyz)
                wt = src.get_weight(uv)
                uv_list.append(uv)
                wt_list.append(wt)
                wt_total += wt
            # Render overall image using calculated weights.
            for n in range(len(self.sources)):
                wt_norm = wt_list[n] / wt_total
                self.sources[n].add_pixels(uv_list[n], img1d, wt_norm)
        else:
            raise ValueError('Invalid render mode.')
        # Convert to fixed-point image matrix and return.
        img2d = np.reshape(img1d, (rows, cols, self.clrs))
        return np.asarray(img2d, dtype=self.dtype)

    # Compute a normalized alignment score, based on size of overlap and
    # the pixel-differences in that region.  Note: Lower = Better.
    def _score(self, svec, xyz, wt_pixel, wt_blank):
        # Update lens parameters from state vector.
        self._set_state_vector(svec)
        # Determine masks for each input image.
        uv0 = self.sources[0].get_uv(xyz)
        uv1 = self.sources[1].get_uv(xyz)
        wt0 = self.sources[0].get_weight(uv0) > 0
        wt1 = self.sources[1].get_weight(uv1) > 0
        # Count overlapping pixels.
        ovr_mask = np.logical_and(wt0, wt1)             # Overlapping pixel
        pix_count = np.sum(wt0) + np.sum(wt1)           # Total drawn pixels
        blk_count = np.sum(np.logical_and(~wt0, ~wt1))  # Number of blank pixels
        # Allocate Nx3 or Nx1 "1D" pixel-list (raster-order).
        pcount = max(xyz.shape)
        img1d = np.zeros((pcount, self.clrs), dtype='float32')
        # Render the difference image, overlapping region only.
        self.sources[0].add_pixels(uv0, img1d, 1.0*ovr_mask)
        self.sources[1].add_pixels(uv1, img1d, -1.0*ovr_mask)
        # Sum-of-differences.
        sum_sqd = np.sum(np.sum(np.sum(np.square(img1d))))
        # Compute overall score.  (Note: Higher = Better)
        score = sum_sqd + wt_blank * blk_count - wt_pixel * pix_count
        # (Debug) Print status information.
        if (self.debug):
            print(str(svec) + ' --> ' + str(score))
        return score
