import sys, os
sys.path.append('../')
from segmentation import *
from support.ct_reader import *
from preprocessing import *
from paths import *
from glob import glob
from os.path import join, basename
from os import listdir
from tqdm import tqdm
from multiprocessing import Pool
import time



CPU = 48

"""
This values was obtain as mean between 10th percentiles
from both datasets LUNA & DSB2017
"""
SPACING = [0.7, 0.6, 0.6]


def is_visited(path):
    processed = listdir(join(PATH['LUNA_OUT'], 'SOBEL_IMG'))
    name = basename(path).split('.mhd')[0]
    for patient in processed:
        if name.lower() in patient.lower():
            return True

    try:
        for ct_file in listdir(path):
            if '.npy' in ct_file.lower():
                return True
    except:
        return False
    return False


def evaluate_masks(patients):
    for patient in tqdm(patients):


        if is_visited(patient):
            continue
        ct_scan = read_ct_scan(patient)

        ct_scan_px = get_pixels_hu(ct_scan)
        ct_scan_px, spacing = resample(ct_scan_px, ct_scan, SPACING)

        ct_mask_F = segment_lung_mask(ct_scan_px, False)
        ct_mask_T = segment_lung_mask(ct_scan_px, True)
        ct_mask_diff = ct_mask_T - ct_mask_F

        segmenteds = []
        lungfilters = []
        sobel_gradients = []

        # start = time.time()
        #
        # with Pool(CPU) as pool:
        #     ct_excluded = pool.map(exclude_lungs, [ct_slice for ct_slice in ct_scan_px])
        #
        # end = time.time()
        # print(end - start)

        for ct_slice in ct_scan_px:
            segmented, lungfilter, sobel_gradient = exclude_lungs(ct_slice)
            segmenteds.append(segmented)
            lungfilters.append(lungfilter)
            sobel_gradients.append(sobel_gradient)


        # ct_mask = segment_lung_from_ct_scan(ct_scan)
        patient = basename(patient).split('.mhd')[0]
        save(join(PATH['LUNA_OUT'], 'LUNGS_IMG', patient + 'lungs'), asarray(segmenteds))
        save(join(PATH['LUNA_OUT'], 'MASKS', patient + 'mask'), ct_mask_T)
        save(join(PATH['LUNA_OUT'], 'MASKS', patient + 'diff'), ct_mask_diff)
        save(join(PATH['LUNA_OUT'], 'MASKS', patient + 'filter'), asarray(lungfilters))
        save(join(PATH['LUNA_OUT'], 'SOBEL_IMG', patient + 'sobel'), asarray(sobel_gradients))


# patients = glob(join(PATH['DATA'], '*'))
patients = glob(join(PATH['LUNA_DATA'], '*.mhd'))


with Pool(CPU) as p:
    res = p.map(evaluate_masks, [[patient] for patient in sort(patients)[:300]])



