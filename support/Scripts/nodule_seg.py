
# coding: utf-8

# In[1]:

import sys
sys.path.append('../')
sys.path.append('../support/')
sys.path.append('../lung_segmentation/')

from preprocessing import *
from ct_reader import *
import pandas as pd
from os.path import join, basename, isfile
from scipy.ndimage.interpolation import zoom
from glob import glob
from multiprocessing import Pool
from scipy.ndimage import morphology
from scipy.ndimage import label
from skimage import measure
from tqdm import tqdm
from multiprocessing import Pool
import pickle
from paths import * 
from skimage.transform import resize
from numpy import *
import warnings
# warnings.filterwarnings('ignore')


# In[107]:

BORDER = 32
BATCH_SIZE = 32
OUT_LUNGS = -9000
SPACING = array([.9, .7, .7])
UPSIDES = pickle.load(open(join(PATH['WEIGHTS'], 
                                'upsides'), 'rb'))
CPU = 24


# In[3]:

def read_ct(path, ret_xy_spacing=False, ret_original_format=True):
    patient = read_ct_scan(path)
    image = get_pixels_hu(patient)
#     image[image == image[0,0,0]] = 0
    
    if ret_original_format:
        return image, patient
    
    if ret_xy_spacing:
        return image, patient.GetSpacing()[0]
    
    return image


# In[4]:

def segment_nodules(patch, mask, is_nodule=True, magic_const=50):
    prepared = (patch - patch.min()) / (patch.max() - patch.min())

    kmeans = KMeans(n_clusters=2)
    data = prepared[coords]
    
    if data.shape[0] <= 2:
        return mask
    
    data = kmeans.fit_predict(expand_dims(data, 1))

    kmean = zeros(mask.shape)
    kmean[coords] = data + magic_const
    labels, num = label(kmean, return_num=True, background=0)

    nodule_a = argmax([sum(labels == i) for i in range(1, num + 1)]) + 1
    init = kmeans.predict(expand_dims(prepared[labels == nodule_a], 1)).min()
    nodule_b = list()
    for i in range(1, num + 1):
        if i != nodule_a:
            if kmeans.predict(expand_dims(prepared[where(labels == i)], 1)).min() != init:
                nodule_b.append((sum(labels == i), i))

    nodule_b = max(nodule_b)[1]

    A = prepared[labels == nodule_a]
    B = prepared[labels == nodule_b]

    if mean(A.reshape(-1)) > mean(B.reshape(-1)):
        labels = labels == nodule_a
    else:
        labels = labels == nodule_b

    return labels


# In[192]:

def overlap(lung, mask):
#     iso = binary_dilation(imresize(isolated[163], (512, 512)))
    labeled, num = label(mask)
    coords = list()
    for colour in range(1, labeled.max() + 1):
        coords.append(where(labeled == colour))
        
    coords = array([[int(coord[0].mean())
                     for coord in coords], 
                    [int(coord[1].mean())
                     for coord in coords],
                    [int(coord[2].mean())
                     for coord in coords]])
    
    pads = ((BORDER, BORDER), 
            (BORDER, BORDER), 
            (BORDER, BORDER))
    
    lung = pad(lung, pads, 
               mode='edge')
    res = zeros(lung.shape)
    labeled = pad(labeled, pads, 
                  mode='edge')
    patches = list()
    masks = list()
    for coord in coords.T:
        res[coord[0]: coord[0] + 2 * BORDER,
            coord[1]: coord[1] + 2 * BORDER,
            coord[2]: coord[2] + 2 * BORDER] += \
            segment_nodule(lung[coord[0]: coord[0] + 2 * BORDER,
                                coord[1]: coord[1] + 2 * BORDER,
                                coord[2]: coord[2] + 2 * BORDER],
                           labeled[coord[0]: coord[0] + 2 * BORDER,
                                   coord[1]: coord[1] + 2 * BORDER,
                                   coord[2]: coord[2] + 2 * BORDER])

    return res[BORDER: -BORDER,
               BORDER: -BORDER,
               BORDER: -BORDER]


# In[178]:

def region_growing(img, seed, minthr, maxthr, structure=None):
    """code was taken from:
    https://github.com/loli/medpy/wiki/Basic-image-manipulation
    """
    img[seed] = minthr
    thrimg = (img <= maxthr) & (img >= minthr)
    lmap, _ = label(thrimg, structure=structure)
    lids = unique(lmap[seed])
    region = zeros(img.shape, bool_)
    for lid in lids:
        region |= lmap == lid
    return region


# In[201]:

def operate(path):
    lung, ct_lung = read_ct(path, ret_original_format=True)
    
    lung, spacing = resample(lung, ct_lung, (1, 1, 1))
    
    name = basename(path)
    mask = load(join(PATH['DATA_ENHANCED'], 
                         name + '.npy'))
    lung_mask = load(join(PATH['DATA_OUT'], 
                              name + '.npy'))
    lung += OUT_LUNGS * (lung_mask == 0)
#     mask, spacing = resample(mask, (1, 1, 1), SPACING)
    
    if name in UPSIDES:
        lung = flipud(lung)
        mask = flipud(mask)
    
    lung_mask = (lung_mask // 8) * 2 + overlap(lung, mask)
    save(join(PATH['DATA_OUT'], 
                              name + '.npy'), lung_mask)
    return name


# In[202]:

def segment_nodule(btch,mask):
    batch = btch.copy()
    interval = [64, 64]
    grown = [mask == find_closest_label(batch,mask)] * 2
    minval = median(batch[grown[-1]])
    maxval = median(batch[grown[-1]])
    sums = [grown[0].sum()] * 2
    while True: #unique((mask == mask[32][32]) == grown).shape[0] == 1:
        grown.append(region_growing(batch, 
                                    grown[-1], 
                                    minval, 
                                    maxval))
        minval -= interval[0]
        maxval += interval[1]
        grown.pop(0)
        sums.append(grown[-1].sum())
        sums.pop(0)
        if 2 * sums[0] < sums[1]:
            interval[0] = interval[0] // 2
            interval[1] = interval[1] // 2
            grown = [grown[0]] * 2
            sums = [grown[0].sum()] * 2
            if interval[0] == 1:
                break
        if grown[0].sum() > 27000:
            break
    return grown[0]


# In[203]:

def find_closest_label(lung, enh):
    center = zeros(lung.shape)
    center[lung.shape[0] // 2, lung.shape[1] // 2, lung.shape[2] // 2] = 1
    ans = (enh * center).max()
    while ans == 0:
        center = morphology.binary_dilation(center)
        ans = (enh * center).max()
    return ans


# In[204]:

def get_remind_files():
    file_list = set(glob(join(PATH['DATA'], '*')))
    err_file_names = load(join(PATH['WEIGHTS'],'erroneus'))
    file_list = file_list.difference(set([join(PATH['DATA'], path) 
                            for path in err_file_names]))
#     file_list = file_list.difference(set([join(PATH['DATA'], basename(path).split('.npy')[0][:32]) 
#                             for path in glob(join(PATH['ENHANCED_CROPPED'], '*.npy'))]))
    return sorted(list(file_list))


# In[195]:

file_list = get_remind_files()
with Pool(CPU) as pool:
    processed = pool.map(operate, file_list)
pickle.dump(processed, open(join(PATH['WEIGHTS'], 'processed_grown_nodules'), 'wb'))
# for file in file_list:
    
# #     batch_files = file_list[BATCH_SIZE * counter:
# #                             BATCH_SIZE * (counter + 1)]

# #     nodule_mask, lung, lung_mask = operate(batch_files[0])
    
# #     print('1 finished')
# #     nodule_mask, lung = operate(batch_files[0])
# #     nodule_masks.append(nodule_mask)
# #     lungs.append(lung)
#     break

