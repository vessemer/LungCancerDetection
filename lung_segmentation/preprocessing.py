from numpy import *
from numpy import round
<<<<<<< HEAD
from scipy.misc import imresize
=======
>>>>>>> 1362926d9120cec3f7e15c0c2bfb790e6ac8f408
import scipy


def resample(image, scan, new_spacing=[0.7, 0.6, 0.6]):
    # Determine current pixel spacing
    try:
        spacing = array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=float32)
    except AttributeError:
<<<<<<< HEAD
        spacing = scan.GetSpacing()
        spacing = array([spacing[-1], spacing[0], spacing[1]])
=======
        try:
            spacing = scan.GetSpacing()
            spacing = array([spacing[-1], spacing[0], spacing[1]])
        except:
            spacing = scan
>>>>>>> 1362926d9120cec3f7e15c0c2bfb790e6ac8f408

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    return image, new_spacing

