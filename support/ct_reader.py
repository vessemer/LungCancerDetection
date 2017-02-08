import dicom
from numpy import *
import os


def read_ct_scan(folder_name):
    # type: (object) -> object
    # Read the slices from the dicom file
    slices = [(dicom.read_file(os.path.join(folder_name, filename)), filename)
              for filename in os.listdir(folder_name)]

    # Sort the dicom slices in their respective order
    slices.sort(key=lambda x: int(x[0].InstanceNumber))
    return list(zip(*slices))


def extract_array(ct_scan):
        heights = asarray([int(ct_slice.SliceLocation)for ct_slice in ct_scan])
        ct_scan = stack([ct_slice.pixel_array for ct_slice in ct_scan])
        ct_scan[ct_scan == -2000] = 0
        return ct_scan, heights

