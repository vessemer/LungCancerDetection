import dicom
from numpy import *
import SimpleITK as sitk
import os


def read_ct_scan(path, verbose=False):
    # type: (object) -> object
    # Read the slices from the dicom file
    slices = []
    if os.path.isfile(path):
        try:
            return sitk.ReadImage(path)
        except:
            if verbose:
                print('Neither a DICOM nor a MHD file: %s' % os.path.basename(path))

    if os.path.isdir(path):
        files = os.listdir(path)
        for filename in files:
            try:
                slices.append(dicom.read_file(os.path.join(path, filename)))
            except dicom.filereader.InvalidDicomError:
                if verbose:
                    print('Neither a DICOM nor a MHD file: %s' % filename)

        slices.sort(key=lambda x: int(x.InstanceNumber))

        try:
            slice_thickness = abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
        except AttributeError:
            slice_thickness = abs(slices[0].SliceLocation - slices[1].SliceLocation)

        for s in slices:
            s.SliceThickness = slice_thickness

        return slices


def extract_array(ct_scan):
        heights = asarray([int(ct_slice.SliceLocation)for ct_slice in ct_scan])
        ct_scan = stack([ct_slice.pixel_array for ct_slice in ct_scan])
        ct_scan[ct_scan == -2000] = 0
        return ct_scan, heights


def get_pixels_hu(slices):
    try:
        image = stack([s.pixel_array for s in slices])
    except AttributeError:
        return sitk.GetArrayFromImage(slices)
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):

        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(float64)
            image[slice_number] = image[slice_number].astype(int16)

        image[slice_number] += int16(intercept)

    return array(image, dtype=int16)
