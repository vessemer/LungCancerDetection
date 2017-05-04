from numpy import *
from skimage.measure import label
from scipy.ndimage.morphology import binary_erosion
from skimage.segmentation import clear_border


def label_size(labeled_matrix, label): return len(labeled_matrix[labeled_matrix == label])


trash_threshold = .5


def remove_trash(labeled_matrix, label_num):
    clear_border(labeled_matrix)
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


def if_separate(mask):
    mask, count = label(mask, connectivity=1, return_num=True)
    count = remove_trash(mask, count)
    return count != 1

def define_lungs(labeled):
    labeld = labeled+500
    labels = bincount(labeld.flatten()).argsort()[-3:-1]
    label0_rightest = max(where(labeld==labels[0])[1])
    label1_rightest = max(where(labeld==labels[1])[1])
    label0_rightest,label1_rightest
    if label0_rightest<label1_rightest:
        labeld[labeld == labels[0]] = 1
        labeld[labeld == labels[1]] = 2
    else:
        labeld[labeld == labels[0]] = 2
        labeld[labeld == labels[1]] = 1
    labeld[labeld>2] = 0
    return labeld

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
    interval = int(file.shape[0] * 0.1)
    start_slice_ind = -1
    start_slice = []
    for i in range(file.shape[0] // 2 - interval, file.shape[0] // 2 + interval):
        if (if_separate(file[i])):
            start_slice_ind = i
            break
    if (start_slice_ind == -1):
        start_slice = separate_lungs(file[file.shape[0] // 2], file.shape[0] // 2)
        start_slice_ind = file.shape[0] // 2
    else:
        start_slice = define_lungs(label(file[start_slice_ind], connectivity=1))
    cur_slice = start_slice
    ret = file.astype(int)
    ret[start_slice_ind] = start_slice
    for i in range(start_slice_ind + 1, file.shape[0]):
        new_slice = separate_new_slice(file[i].astype(int), cur_slice, i)
        ret[i] = new_slice
        cur_slice = new_slice
    cur_slice = start_slice
    for i in range(start_slice_ind - 1, -1, -1):
        new_slice = separate_new_slice(file[i].astype(int), cur_slice, i)
        ret[i] = new_slice
        cur_slice = new_slice
    return extract_lungs(ret)


def extract_lungs(separated):
    left_lung = separated.copy()
    left_lung[left_lung != 1] = 0
    right_lung = separated.copy()
    right_lung[right_lung != 2] = 0
    right_lung[right_lung == 2] = 1
    return left_lung, right_lung
