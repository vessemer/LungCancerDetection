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

LOWER = 400
UPPER = 1600


def extract_patches_nodules(lung, sobel, diff):
    candidate = diff == NODULE_LABEL

    if not candidate.sum():
        return [], [], []

    patches = list()
    sobels = list()
    masks = list()

    labels, num = measure.label(candidate, background=0, return_num=True)
    for label in range(1, num + 1):
        coords = where(labels == label)
        min_max = list()
        deltas = list()

        for j, coord in enumerate(coords):
            min_max.append((coord.min(), coord.max()))
            deltas.append(int(BORDER * (min_max[-1][1] - min_max[-1][0])))
            deltas[-1] = (clip(min_max[-1][0] - deltas[-1], 0, lung.shape[j]),
                          clip(min_max[-1][1] + deltas[-1], 0, lung.shape[j]))

        patches.append(lung[deltas[0][0]:deltas[0][1], deltas[1][0]:deltas[1][1], deltas[2][0]:deltas[2][1]])
        sobels.append(sobel[deltas[0][0]:deltas[0][1], deltas[1][0]:deltas[1][1], deltas[2][0]:deltas[2][1]])
        masks.append(diff[deltas[0][0]:deltas[0][1], deltas[1][0]:deltas[1][1], deltas[2][0]:deltas[2][1]])

    return patches, sobels, masks


def extract_patches_vessels(lung, sobel, diff, amount=2):
    candidate = diff == VESSEL_LABEL
    if not candidate.sum():
        return [], [], []

    flag = 0
    start = 1

    labels, num = measure.label(candidate, background=0, return_num=True)

    marks = arange(start, num + 1)
    random.shuffle(marks)

    patches = list()
    sobels = list()
    masks = list()

    for k, label in enumerate(marks):
        if flag >= amount:
            return patches, sobels, masks

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
        sobels.append(sobel[deltas[0][0]:deltas[0][1], deltas[1][0]:deltas[1][1], deltas[2][0]:deltas[2][1]])
        masks.append(diff[deltas[0][0]:deltas[0][1], deltas[1][0]:deltas[1][1], deltas[2][0]:deltas[2][1]])
        flag += 1
    return patches, sobels, masks


annotations = pd.read_csv(join(PATH['LUNA_CSV'], 'annotations.csv'))
preprocessed_files = asarray([basename(preprocessed_file).split('sobel')[0]
                              for preprocessed_file in glob(join(PATH['LUNA_OUT'],
                                                                 'SOBEL_IMG',
                                                                 '*sobel_*.npy'))])

preprocessed_nodules = asarray([basename(preprocessed_file).split('_sobel')[0][:-1]
                                for preprocessed_file in glob(join(PATH['LUNA_OUT'],
                                                                   'PATCHES',
                                                                   'NODULES',
                                                                   '*_sobel.npy'))])

preprocessed_nodules = [path.split('left')[0] if 'left' in path else path.split('right')[0]
                        for path in preprocessed_nodules]

files_with_nodules = unique(annotations.seriesuid.values)
preprocessed_files_with_nodules = intersect1d(files_with_nodules, preprocessed_files)
preprocessed_files_with_nodules = setdiff1d(preprocessed_files_with_nodules, preprocessed_nodules)
preprocessed_files_pure = setdiff1d(preprocessed_files, preprocessed_files_with_nodules)


for name in preprocessed_files_with_nodules:
    for side in ['left', 'right']:
        lung = load(join(PATH['LUNA_OUT'], 'LUNGS_IMG', name + 'lungs_' + side + '.npy'))
        sobel = load(join(PATH['LUNA_OUT'], 'SOBEL_IMG', name + 'sobel_' + side + '.npy'))
        diff = load(join(PATH['LUNA_OUT'], 'MASKS', name + 'diff_' + side + '.npy'))

        patch, sobel, mask = extract_patches_nodules(lung, sobel, diff)
        for i in range(len(mask)):
            save(join(PATH['LUNA_OUT'], 'PATCHES', 'NODULES', name + side + str(i) + '_patch'), patch[i])
            save(join(PATH['LUNA_OUT'], 'PATCHES', 'NODULES', name + side + str(i) + '_sobel'), sobel[i])
            save(join(PATH['LUNA_OUT'], 'PATCHES', 'NODULES', name + side + str(i) + '_mask'), mask[i])


for name in preprocessed_files_pure:
    for side in ['left', 'right']:
        lung = load(join(PATH['LUNA_OUT'], 'LUNGS_IMG', name + 'lungs_' + side + '.npy'))
        sobel = load(join(PATH['LUNA_OUT'], 'SOBEL_IMG', name + 'sobel_' + side + '.npy'))
        diff = load(join(PATH['LUNA_OUT'], 'MASKS', name + 'diff_' + side + '.npy'))
        patch, sobel, mask = extract_patches_vessels(lung, sobel, diff)
        add = random.randint(0, 9)
        for i in range(len(mask)):
            save(join(PATH['LUNA_OUT'], 'PATCHES', 'VESSELS', name + side + str(i + add) + '_patch'), patch[i])
            save(join(PATH['LUNA_OUT'], 'PATCHES', 'VESSELS', name + side + str(i + add) + '_sobel'), sobel[i])
            save(join(PATH['LUNA_OUT'], 'PATCHES', 'VESSELS', name + side + str(i + add) + '_mask'), mask[i])

