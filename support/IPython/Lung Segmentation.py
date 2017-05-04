import sys

sys.path.append('../')
sys.path.append('../support/')
sys.path.append('../lung_segmentation/')
from glob import glob
from skimage import morphology
from scipy.ndimage import morphology as scpm
import timeit
from skimage import measure
from scipy import stats
from sklearn.cluster import KMeans
from skimage.transform import resize
from sklearn.cluster import KMeans
from os.path import join, basename
from scipy import ndimage as ndi
from multiprocessing import Pool
from scipy.ndimage.interpolation import rotate
from tqdm import tqdm
from functools import partial
import pandas as pd
from paths import *
from ct_reader import *
from lung_separation import *
import csv
import numpy as np


def read_ct(path):
    patient = read_ct_scan(path)
    patient = get_pixels_hu(patient)
    return patient


def ethalon_ovarlapping(center, eth, markers=None):
    max_count = 0
    lungs = [1, 1]
    flag = False
    if markers is None:
        flag = True
        center, markers = label(center, return_num=True)
        markers = list(range(1, markers + 1))

    if len(markers) == 0:
        return None

    for side in [0, 1]:
        for labe in markers:
            count = count_nonzero((center == labe) * eth[side])
            if max_count < count:
                max_count = count
                lungs[side] = labe
    if flag:
        return (center == lungs[0]), (center == lungs[1])

    return lungs

def get_thresh_img(patient):
    border = (array(patient.shape[1:]) * .2).astype(int)
    middle = patient[:,
             patient.shape[1] // 2 - border[0]: patient.shape[1] // 2 + border[0],
             patient.shape[2] // 2 - border[1]: patient.shape[2] // 2 + border[1]]
    new_mean = mean(middle)
    new_max = patient.max()
    new_min = patient.min()
    # move the underflow bins
    patient[patient == new_max] = new_mean
    patient[patient == new_min] = new_mean
    kmeans = KMeans(n_clusters=2).fit(middle.reshape(-1, 1))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = mean(centers)
    thresh_img = where(patient < threshold, 1.0, 0.0).astype(int)
    thresh_img = scpm.binary_opening(thresh_img)
    thresh_img = clear_border_maped(thresh_img)
    thresh_img, num = label(thresh_img, return_num=True)

    markers = vstack([bincount(thresh_img.flatten())[1:], arange(1, num + 1)])
    markers = asarray(sorted(markers.T, key=lambda x: x[0]))[-7:, 1]

    origin = thresh_img.shape[0] // 2
    border = int(origin * .1)
    center = thresh_img[origin - border: origin + border]
    new_eth = (stack([ethalon == 1] * border * 2),
               stack([ethalon == 2] * border * 2))

    lungs = ethalon_ovarlapping(center, new_eth, markers)
    thresh_img = (thresh_img == lungs[0]) + (thresh_img == lungs[1])

    # new_eth = (ethalon == 1,
    #            ethalon == 2)
    # new_eth = ethalon_ovarlapping(thresh_img.shape[0] // 2, new_eth)

    # for i in range(thresh_img.shape[0] // 2, 0, -1):
    #     new_eth = ethalon_ovarlapping(thresh_img, new_eth)
    #     if lungs is None:
    #         return thresh_img
    #
    #     thresh_img[i] = (new_eth[0] + new_eth[1]) > 0

    return thresh_img


def clear_border_maped(input_x):
    for i in range(input_x.shape[0]):
        clear_border(input_x[i], in_place=True)

    return input_x


def clear_border3D(input_x, labels=None,
                   is_labeled=True, background_start=0):
    if not is_labeled:
        input_x = label(input_x)

    if labels is None:
        labels = unique(input_x)
        labels = labels[labels > background_start]

    for i, el in enumerate(labels):
        coords = where(input_x == el)
        coords_min = array([coords[1].min(),
                            coords[2].min()])

        coords_max = array([coords[1].max(),
                            coords[2].max()])

        out_max = any(coords_max == (array(input_x.shape[1:]) - 1))
        out_min = any(coords_min == 0)
        if out_max or out_min:
            labels[i] = background_start

    return labels[labels != background_start]


from numpy import *
from skimage.measure import label
from scipy.ndimage.morphology import binary_erosion
from skimage.segmentation import clear_border


def label_size(labeled_matrix, label): return len(labeled_matrix[labeled_matrix == label])


def remove_trash(labeled_matrix, label_num):
    # Здесь был clear_border, я его убрал,
    # т.к. тебе передаются снимки ужеобработанные clear_border3d
    max_label_size = 0
    new_label_num = 0
    for i in range(1, label_num + 1):
        if len(labeled_matrix[labeled_matrix == i]) > max_label_size:
            max_label_size = len(labeled_matrix[labeled_matrix == i])

    for i in range(1, label_num + 1):
        if label_size(labeled_matrix, i) < trash_threshold * max_label_size:
            labeled_matrix[labeled_matrix == i] = 0
        else:
            new_label_num += 1
            labeled_matrix[labeled_matrix == i] = new_label_num
    return new_label_num


def smart_labelling(matrix):
    labeled, label_num = label(matrix, connectivity=1, return_num=True)
    max_size, max2_size, max_label, max2_label, if_tricky = 0, 0, 0, 0, False

    if label_num != 2:
        if_tricky = True

    for i in range(1, label_num + 1):
        label_size = len((where(labeled == i)[1]))
        if label_size > max_size:
            max2_size, max2_label = max_size, max_label
            max_size, max_label = label_size, i
        elif label_size > max2_size:
            max2_size, max2_label = label_size, i

    for i in range(1, label_num + 1):
        if i != max_label and i != max2_label:
            labeled[labeled == i] = 0
    min1 = min(where(labeled == max_label)[1])
    min2 = min(where(labeled == max2_label)[1])
    if min1 < min2:
        labeled[labeled == max_label] = -1
        labeled[labeled == max2_label] = -2
    else:
        labeled[labeled == max_label] = -2
        labeled[labeled == max2_label] = -1
    return -labeled, if_tricky


def if_separate(mask):
    mask, count = label(mask, return_num=True)
    count = remove_trash(mask, count)
    return count != 1


def separate_lungs(label_matrix, layer_num):
    before_morph_open = label_matrix
    while not if_separate(label_matrix):
        label_matrix = binary_erosion(label_matrix, structure=ones((7, 1)))
    label_matrix = label(label_matrix, connectivity=1)
    inverse_erosion(label_matrix, before_morph_open, layer_num)
    return label_matrix


def inverse_erosion(label_matrix, mask, slice_num):
    xs, ys = where(label_matrix < mask)
    border_coords = list(zip(xs, ys))
    while len(border_coords):
        to1 = []
        to2 = []
        new_border_coords = []
        for x, y in border_coords:
            chunk = label_matrix[x - 1:x + 2, y - 1:y + 2]
            near1 = len(where(chunk == 1)[0])
            near2 = len(where(chunk == 2)[0])
            if near1 and near2:
                if slice_num % 2:
                    to1.append((x, y))
                else:
                    to2.append((x, y))
            elif near1:
                to1.append((x, y))
            elif len(where(chunk == 2)[0]):
                to2.append((x, y))
            else:
                new_border_coords.append((x, y))
        if (len(to1) == 0 and len(to2) == 0):
            for x, y in new_border_coords:
                label_matrix[x, y] = 0
            return
        for x, y in to1:
            label_matrix[x, y] = 1
        for x, y in to2:
            label_matrix[x, y] = 2
        border_coords = new_border_coords


def separate_new_slice(new_slice, prev_slice, slice_num):
    intersect = new_slice * prev_slice
    inverse_erosion(intersect, new_slice, slice_num)
    return intersect


def separate_lungs3d(file):
    #     if (start_slice_ind == -1):
    #         start_slice = separate_lungs(file[file.shape[0] // 2], file.shape[0] // 2)
    #         start_slice_ind = file.shape[0] // 2
    #     else:
    tricky_slice = asarray([])
    cur_slice, if_tricky = smart_labelling(file[file.shape[0] // 2])
    if if_tricky:
        tricky_slice = cur_slice.copy()
    ret = file.astype(int)
    ret[file.shape[0] // 2] = cur_slice
    for i in (range(file.shape[0] // 2 + 1, file.shape[0])):
        ret[i] = separate_new_slice(file[i].astype(int), cur_slice, i)
        cur_slice = ret[i]

    cur_slice = ret[file.shape[0] // 2]
    for i in (range(file.shape[0] // 2 - 1, -1, -1)):
        ret[i] = separate_new_slice(file[i].astype(int), cur_slice, i)
        cur_slice = ret[i]

    return ret, tricky_slice


def extract_lungs(separated):
    left_lung = separated.copy()
    left_lung[left_lung != 1] = 0
    right_lung = separated.copy()
    right_lung[right_lung != 2] = 0
    right_lung[right_lung == 2] = 1
    return left_lung, right_lung


def operate(patient):
    start = timeit.default_timer()
    t_img = get_thresh_img(patient)
    # print('Thresholding: done in %f seconds'
    #       % (timeit.default_timer() - start))

    start = timeit.default_timer()
    lungs = separate_lungs3d(t_img)
    # print('Separation: done in %f seconds'
    #       % (timeit.default_timer() - start))
    return lungs


def is_correct_name(path, acceptable_names):
    for name in acceptable_names:
        if name in path:
            return True
    return False


batch_size = 888
CPU = 1
from multiprocessing import Pool


ethalon = load(join(PATH['WEIGHTS'], 'ethalon.npy'))

file_list = list()
for fold in glob(join(PATH['LUNA_DATA'], '*')):
    file_list += glob(join(fold, '*.mhd'))

base_names = set([basename(path) for path in file_list])
base_names = list(base_names.difference(set([basename(path).split('.npy')[0]
                                             for path in glob(join(PATH['LUNA_OUT'], '*.npy'))])))

file_list = [path for path in file_list if is_correct_name(path, base_names)]

for i in range(len(file_list) // batch_size + 1):
    batch_files = file_list[i * batch_size: (i + 1) * batch_size]
    patients = [read_ct(path)
                for path in file_list]

    mask_t_slice = list()
    # with Pool(CPU) as pool:
    #     mask_t_slice = pool.map(operate,
    #                             patients)  # в t_slice или слайс, в котором не две компоненты, либо пустой лист. в mask - маска 0,1,2
    for patient, path in zip(patients, batch_files):
        mask_t_slice = operate(patient)
        save(join(PATH['LUNA_OUT'], basename(path).split('.mhd')[0]), mask_t_slice[0].astype(uint8))
        save(join(PATH['LUNA_TRICKY'], basename(path).split('.mhd')[0]), mask_t_slice[1].astype(uint8))

    # for mts, path in zip(mask_t_slice, batch_files):
    #     save(join(PATH['LUNA_OUT'], basename(path).split('.mhd')[0]), mts[0].astype(uint8))
    #     save(join(PATH['LUNA_TRICKY'], basename(path).split('.mhd')[0]), mts[1].astype(uint8))
    # print('here')