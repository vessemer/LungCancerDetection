import sys

sys.path.append('../')
from skimage import measure
from paths import *
from os.path import basename, join
import pandas as pd
from numpy import *
from glob import glob
from tqdm import tqdm


BORDER = .8
REAL_NODULE = True
BORDER_VESSEL = 25
NODULE_LABEL = 2
VESSEL_LABEL = 1

LOWER = 1000
UPPER = 2000


def extract_patches_nodules(lung, diff, real_nodule=True):
    if real_nodule:
        candidate = diff == NODULE_LABEL
    else:
        candidate = diff == VESSEL_LABEL

    if not candidate.sum():
        return [], []

    patches = list()
    masks = list()

    labels, num = measure.label(candidate, background=0, return_num=True)
    for label in range(1, num + 1):
        coords = where(labels == label)
        min_max = list()
        deltas = list()

        for i, coord in enumerate(coords):
            min_max.append((coord.min(), coord.max()))
            deltas.append(int(BORDER * (min_max[-1][1] - min_max[-1][0])))
            deltas[-1] = (clip(min_max[-1][0] - deltas[-1], 0, lung.shape[i]),
                          clip(min_max[-1][1] + deltas[-1], 0, lung.shape[i]))

        patches.append(lung[deltas[0][0]:deltas[0][1], deltas[1][0]:deltas[1][1], deltas[2][0]:deltas[2][1]])
        masks.append(diff[deltas[0][0]:deltas[0][1], deltas[1][0]:deltas[1][1], deltas[2][0]:deltas[2][1]])

    return patches, masks


def extract_patches_vessels(lung, diff, amount=2):
    candidate = diff == VESSEL_LABEL
    if not candidate.sum():
        return [], []

    flag = 0
    start = 1

    labels, num = measure.label(candidate, background=0, return_num=True)

    marks = arange(start, num + 1)
    random.shuffle(marks)

    patches = list()
    masks = list()

    for i, label in enumerate(marks):
        if flag >= amount:
            return patches, masks

        overlaped = labels == label
        area = overlaped.sum()

        if area < LOWER or area > UPPER:
            continue
        coords = where(labels == label)

        medians = list()
        deltas = list()
        for j, coord in enumerate(coords):
            medians.append(median(coord))
            deltas.append((clip(int(medians[-1] - BORDER_VESSEL), 0, lung.shape[j]),
                           clip(int(medians[-1] + BORDER_VESSEL), 0, lung.shape[j])))

        patches.append(lung[deltas[0][0]:deltas[0][1], deltas[1][0]:deltas[1][1], deltas[2][0]:deltas[2][1]])
        masks.append(diff[deltas[0][0]:deltas[0][1], deltas[1][0]:deltas[1][1], deltas[2][0]:deltas[2][1]])
        flag += 1

    return patches, masks


annotations = pd.read_csv(join(PATH['LUNA_CSV'], 'annotations.csv'))
preprocessed_files = asarray([basename(preprocessed_file).split('lungs')[0]
                              for preprocessed_file in glob(join(PATH['LUNA_OUT'],
                                                                 'LUNGS_IMG',
                                                                 '*lungs_*.npy'))])
files_with_nodules = unique(annotations.seriesuid.values)
preprocessed_files_with_nodules = intersect1d(files_with_nodules, preprocessed_files)
preprocessed_files_pure = setdiff1d(preprocessed_files, preprocessed_files_with_nodules)

# patches_with_nodules = list()
# masks_with_nodules = list()
# names = list()
#
# for name in tqdm(preprocessed_files_with_nodules):
#     for side in ['left', 'right']:
#         lung = load(join(PATH['LUNA_OUT'], 'LUNGS_IMG', name + 'lungs_' + side + '.npy'))
#         diff = load(join(PATH['LUNA_OUT'], 'MASKS', name + 'diff_' + side + '.npy'))
#         patch, mask = extract_patches_nodules(lung, diff)
#         patches_with_nodules += patch
#         masks_with_nodules += mask
#         names += [name + side + str(i) for i in range(len(mask))]
#
# for patch, mask, name in tqdm(zip(patches_with_nodules,
#                                   masks_with_nodules,
#                                   names)):
#     save(join(PATH['LUNA_OUT'], 'PATCHES', 'NODULES', name + '_patch'), patch)
#     save(join(PATH['LUNA_OUT'], 'PATCHES', 'NODULES', name + '_mask'), mask)
#
# del patches_with_nodules
# del masks_with_nodules

patches_with_vessels = list()
masks_with_vessels = list()
names = list()

for name in tqdm(preprocessed_files_pure):
    for side in ['left', 'right']:
        lung = load(join(PATH['LUNA_OUT'], 'LUNGS_IMG', name + 'lungs_' + side + '.npy'))
        diff = load(join(PATH['LUNA_OUT'], 'MASKS', name + 'diff_' + side + '.npy'))
        patch, mask = extract_patches_vessels(lung, diff)
        patches_with_vessels += patch
        masks_with_vessels += mask
        names += [name + side + str(i) for i in range(len(mask))]

for patch, mask, name in tqdm(zip(patches_with_vessels,
                                  masks_with_vessels,
                                  names)):
    save(join(PATH['LUNA_OUT'], 'PATCHES', 'VESSELS', name + '_patch'), patch)
    save(join(PATH['LUNA_OUT'], 'PATCHES', 'VESSELS', name + '_mask'), mask)

