from segmentation import *
from support.ct_reader import *
from preprocessing import *
from paths import *
from glob import glob
from os.path import join
from os import listdir
from tqdm import tqdm


def is_visited(path):
    for ct_file in listdir(path):
        if '.npy' in ct_file.lower():
            return True
    return False

patients = glob(join(PATH['DATA'], '*'))

for patient in tqdm(patients):

    if is_visited(patient):
        continue

    ct_scan = read_ct_scan(patient)

    ct_scan_px = get_pixels_hu(ct_scan)
    ct_scan_px, spacing = resample(ct_scan_px, ct_scan, [1, 1, 1])

    # print(patient)
    # print("Shape after resampling\t", ct_scan_px.shape)

    ct_mask_F = segment_lung_mask(ct_scan_px, False)
    ct_mask_T = segment_lung_mask(ct_scan_px, True)
    ct_mask_diff = ct_mask_T - ct_mask_F
    # ct_mask = segment_lung_from_ct_scan(ct_scan)

    save(join(patient, 'lungs'), ct_scan_px * ct_mask_T)
    save(join(patient, 'mask'), ct_mask_T)
    save(join(patient, 'diff'), ct_mask_diff)

