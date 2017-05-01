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
import pandas as pd
from lung_segmentation.lung_separation import *
from skimage.draw import ellipsoid

CPU = 48

"""
This values was obtain as mean between 10th percentiles
from both datasets LUNA & DSB2017
"""
SPACING = array([.7, .6, .6])


def is_visited(path):
    if os.path.isfile(path):
        processed = listdir(join(PATH['LUNA_OUT'], 'SOBEL_IMG'))
        name = basename(path).split('.mhd')[0]
        for patient in processed:
            if name.lower() in patient.lower():
                return True
    else:
        processed = listdir(join(PATH['DATA_OUT'], 'SOBEL_IMG'))
        name = basename(path)
        for patient in processed:
            if name.lower() in patient.lower():
                return True
    return False


def overlap(itk_image, anotations, patient, diff):
    nodules = anotations[anotations.seriesuid == patient][['coordX', 'coordY', 'coordZ', 'diameter_mm']]
    origin = itk_image.GetOrigin()

    nodules.coordX = nodules.coordX.apply(lambda x: ((x - origin[0]) / SPACING[1]).astype(int16))
    nodules.coordY = nodules.coordY.apply(lambda y: ((y - origin[1]) / SPACING[2]).astype(int16))
    nodules.coordZ = nodules.coordZ.apply(lambda z: ((z - origin[2]) / SPACING[0]).astype(int16))
    nodules.diameter_mm = nodules.diameter_mm.apply(lambda d: (d / (2 * SPACING) + 4).astype(int8))

    if not nodules.shape[0]:
        return diff, []


    vale = list()
    for i, row in nodules.iterrows():
        el = ellipsoid(row.diameter_mm[2],
                       row.diameter_mm[1],
                       row.diameter_mm[0])

        relative = list()
        dim = asarray(el.shape) // 2
        relative.append(array([clip(row.coordZ - dim[0], 0, diff.shape[1]),
                               clip(row.coordZ + dim[0], 0, diff.shape[1])]))
        relative.append(array([clip(row.coordY - dim[1], 0, diff.shape[1]),
                               clip(row.coordY + dim[1], 0, diff.shape[1])]))
        relative.append(array([clip(row.coordX - dim[2], 0, diff.shape[2]),
                               clip(row.coordX + dim[2], 0, diff.shape[2])]))

        vale.append((diff[relative[0][0]:relative[0][1],
                        relative[1][0]:relative[1][1],
                        relative[2][0]:relative[2][1]] *
                        el[:relative[0][1] - row.coordZ + row.coordZ - relative[0][0],
                           :relative[1][1] - row.coordY + row.coordY - relative[1][0],
                           :relative[2][1] - row.coordX + row.coordX - relative[2][0]]).sum() / el.shape[0])

        diff[relative[0][0]:relative[0][1],
             relative[1][0]:relative[1][1],
             relative[2][0]:relative[2][1]] += \
            (array([2]) * el[:relative[0][1] - row.coordZ + row.coordZ - relative[0][0],
                             :relative[1][1] - row.coordY + row.coordY - relative[1][0],
                             :relative[2][1] - row.coordX + row.coordX - relative[2][0]])

    return diff, vale


def crop_and_rotate(segmenteds, left, right):
    segmenteds_left = left * segmenteds
    segmenteds_right = right * segmenteds

    x, y, z = where(left)
    if len(x):
        segmenteds_left = segmenteds_left[x.min():x.max(), 
                                          y.min():y.max(), 
                                          z.min():z.max()]
    else:
        segmenteds_left = array([])
    x, y, z = where(right)
    if len(x):
        segmenteds_right = segmenteds_right[x.min():x.max(), 
                                            y.min():y.max(), 
                                            z.min():z.max()]
    else:
        segmenteds_left = array([])
    return segmenteds_left, segmenteds_right


def evaluate_masks(patients, anotations=None):

    for i, patient in enumerate(patients):

        if is_visited(patient):
            continue

        vale = list()

        ct_scan = read_ct_scan(patient)

        ct_scan_px = get_pixels_hu(ct_scan)
        ct_scan_px, spacing = resample(ct_scan_px, ct_scan, SPACING)
        ct_mask_F = segment_lung_mask(ct_scan_px, False)
        ct_mask_T = segment_lung_mask(ct_scan_px, True)
        ct_mask_diff = ct_mask_T - ct_mask_F

        if anotations is not None:
            ct_mask_diff, vale = overlap(ct_scan,
                                         anotations,
                                         basename(patient).split('.mhd')[0],
                                         ct_mask_diff)

        # start = time.time()

        segmented = list()
        lung_filter = list()
        sobel = list()

        with Pool(CPU) as pool:
            ct_excluded = pool.map(exclude_lungs, [ct_slice for ct_slice in ct_scan_px])

        # end = time.time()
        # print(end - start)

        for ct_slice in ct_excluded:
            segmented.append(ct_slice[0])
            lung_filter.append(ct_slice[1].astype(bool_))
            sobel.append(ct_slice[2] * ct_slice[1])

        lung_filter = asarray(lung_filter)
        left, right = separate_lungs3d(lung_filter)
        segmented_left, segmented_right = crop_and_rotate(segmented, left, right)
        sobel_left, sobel_right = crop_and_rotate(sobel, left, right)
        diff_left, diff_right = crop_and_rotate(ct_mask_diff, left, right)

        # ct_mask = segment_lung_from_ct_scan(ct_scan)
        patient = basename(patient).split('.mhd')[0]
        save(join(PATH['DATA_OUT'], 'LUNGS_IMG', patient + 'lungs_left'), segmented_left.astype(int16))
        save(join(PATH['DATA_OUT'], 'LUNGS_IMG', patient + 'lungs_right'), segmented_right.astype(int16))

        save(join(PATH['DATA_OUT'], 'MASKS', patient + 'diff_left'), diff_left.astype(int8))
        save(join(PATH['DATA_OUT'], 'MASKS', patient + 'diff_right'), diff_right.astype(int8))

        save(join(PATH['DATA_OUT'], 'SOBEL_IMG', patient + 'sobel_left'), asarray(sobel_left).astype(float16))
        save(join(PATH['DATA_OUT'], 'SOBEL_IMG', patient + 'sobel_right'), asarray(sobel_right).astype(float16))

        if len(vale) and anotations is not None:
            vale = hstack([load(join(PATH['LUNA_OUT'], 'vals.npy')), asarray(vale)])
            save(join(PATH['LUNA_OUT'], 'vals'), vale)


patients = glob(join(PATH['DATA'], '*'))
patients.sort()
# patients = glob(join(PATH['LUNA_DATA'], '*.mhd'))
# annotations = pd.read_csv(join(PATH['LUNA_CSV'], 'annotations.csv'))
evaluate_masks(patients)
