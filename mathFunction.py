
import numpy as np

# Create rotation matrix from an arbitrary quaternion.  See also:
# https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Conversion_to_and_from_the_matrix_representation
def get_rotation_matrix(qq):
    # Normalize matrix and extract individual items.
    qq_norm = np.sqrt(np.sum(np.square(qq)))
    w = qq[0] / qq_norm
    x = qq[1] / qq_norm
    y = qq[2] / qq_norm
    z = qq[3] / qq_norm
    # Convert to rotation matrix.
    return np.matrix([[w*w+x*x-y*y-z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
                      [2*x*y+2*w*z, w*w-x*x+y*y-z*z, 2*y*z-2*w*x],
                      [2*x*z-2*w*y, 2*y*z+2*w*x, w*w-x*x-y*y+z*z]], dtype='float32')

# Conjugate a quaternion to apply the opposite rotation.
def conj_qq(qq):
    return np.array([qq[0], -qq[1], -qq[2], -qq[3]])

# Multiply two quaternions:ab = (a0b0 - av dot bv; a0*bv + b0av + av cross bv)
# https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Conversion_to_and_from_the_matrix_representation
def mul_qq(qa, qb):
    return np.array([qa[0]*qb[0] - qa[1]*qb[1] - qa[2]*qb[2] - qa[3]*qb[3],
                     qa[0]*qb[1] + qa[1]*qb[0] + qa[2]*qb[3] - qa[3]*qb[2],
                     qa[0]*qb[2] + qa[2]*qb[0] + qa[3]*qb[1] - qa[1]*qb[3],
                     qa[0]*qb[3] + qa[3]*qb[0] + qa[1]*qb[2] - qa[2]*qb[1]])

# Generate a normalized quaternion [W,X,Y,Z] from [X,Y,Z]
def norm_qq(x, y, z):
    rsq = x**2 + y**2 + z**2
    if rsq < 1:
        w = np.sqrt(1-rsq)
        return [w, x, y, z]
    else:
        r = np.sqrt(rsq)
        return [0, x/r, y/r, z/r]


# Return length of every column in an MxN matrix.
def matrix_len(x):
    #return np.sqrt(np.sum(np.square(x), axis=0))
    return np.linalg.norm(x, axis=0)

# Normalize an MxN matrix such that all N columns have unit length.
def matrix_norm(x):
    return x / (matrix_len(x) + 1e-9)