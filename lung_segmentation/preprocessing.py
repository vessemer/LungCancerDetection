from numpy import *
from numpy import round
import scipy


def resample(image, scan, new_spacing=[2.5, 0.8, 0.8]):
    # Determine current pixel spacing
    try:
        spacing = array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=float32)
    except AttributeError:
        spacing = scan.GetSpacing()
        spacing = array([spacing[-1], spacing[0], spacing[1]])

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    return image, new_spacing

