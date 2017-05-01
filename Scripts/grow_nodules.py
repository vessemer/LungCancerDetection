
# coding: utf-8

# In[30]:

from numpy import *
import pickle

# In[3]:

import sys
sys.path.append('../')
sys.path.append('../support/')
sys.path.append('../lung_segmentation/')
import os
from preprocessing import *
from ct_reader import *
from scipy.ndimage import morphology 
from tqdm import tqdm
import time
from os.path import join, basename, isfile
from multiprocessing import Pool
from scipy.ndimage import label
import scipy.ndimage.filters as filters
from glob import glob
from paths import *
import functools

from scipy.linalg import norm
from scipy.ndimage.filters import gaussian_filter, laplace
import pandas as pd


# In[4]:

def read_ct(path, ret_ct_scan=False, ret_xy_spacing=False):
    patient = read_ct_scan(path)
    if ret_ct_scan:
        return get_pixels_hu(patient), patient
    if ret_xy_spacing:
        return get_pixels_hu(patient), patient.GetSpacing()[0]
    
    return get_pixels_hu(patient)


# In[5]:

SPACING = array([1., 1., 1.])
ISOLATED_THRESHOLD = -600
DOT_ENHANCED_THRESHOLD = 6
OUT_LUNGS = -9000
BOTTOM = 4
BORDER = 32
TOP = 16
CPU = 6
FILTERS_AMOUNT = 6
ISOLATED_MIN_VOLUME = 9
ISOLATED_MAX_VOLUME = 500
JUXTAVASCULAR_MIN_VOLUME = 9
JUXTAPLEURAL_MIN_VALUME = 1


# In[6]:

def get_scales(bottom=BOTTOM, top=TOP, 
               amount=FILTERS_AMOUNT):
    radius = (top / bottom) ** (1. / (amount - 1))
    sigmas = [bottom / 4.]
    for i in range(amount - 1):
        sigmas.append(sigmas[0] * (radius ** i + 1))
    return sigmas


# In[7]:

def load_data(patient_path, mask_path):    
    ct_scan = read_ct_scan(patient_path)
    mask = load(mask_path)
    patient = get_pixels_hu(ct_scan)
    patient, spacing = resample(patient, ct_scan, SPACING)
    mask, spacing = resample(mask, ct_scan, SPACING)
    
    mask = morphology.binary_fill_holes(
        morphology.binary_dilation(
            morphology.binary_fill_holes(mask > 0), 
            iterations=4)
    )

    return patient, mask

# In[ ]:

paths = PATH['STAGE_ENHANCED']
data_path = PATH['STAGE_DATA']

patients = list(sorted(glob(join(PATH['STAGE_DATA'], '*')), reverse=True))
processed = [basename(path).split('amounts.npy')[0] 
             for path in glob(join(paths,'*amounts.npy'))]

patients = [patient 
            for patient in patients
           if basename(patient) not in processed]


erroneus = pickle.load(open(join(PATH['STAGE_MASKS'], 'erroneus.pkl'), 'rb'))

patients = [patient 
            for patient in patients
           if basename(patient) not in erroneus]

erroneus = set(erroneus)
erroneus = erroneus.difference(pickle.load(open(join(PATH['WEIGHTS'], 
                                                    'still_erroneus'), 'rb')))
patients = [data_path + fixed for fixed in list(erroneus)]



def region_growing(img, seed, minthr, maxthr, structure=None):
    """code was taken from:
    https://github.com/loli/medpy/wiki/Basic-image-manipulation
    """
    img[seed] = minthr
    thrimg = (img <= maxthr) & (img >= minthr)
    lmap, _ = label(thrimg, structure=structure)
    lids = unique(lmap[seed])
    region = zeros(img.shape, bool)
    for lid in lids:
        region |= lmap == lid
    return region


def find_closest_label(lung, enh):
    center = zeros(lung.shape)
    center[lung.shape[0] // 2, lung.shape[1] // 2, lung.shape[2] // 2] = 1
    ans = (enh * center).max()
    while ans == 0:
        center = morphology.binary_dilation(center)
        ans = (enh * center).max()
    return ans


def segment_nodule(btch, mask):
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


def overlap(lung, mask):
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
               mode='constant', constant_values=OUT_LUNGS)
    res = zeros(lung.shape)
    labeled = pad(labeled, pads, 
                  mode='constant', constant_values=OUT_LUNGS)
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

print(len(patients))


for i, patient_path in enumerate(patients):
    mask_path = join(PATH['STAGE_MASKS'], basename(patient_path) + '.npy')
    patient, mask = load_data(patient_path, mask_path)

    nodules = load(join(paths, basename(patient_path) + '.npy'))
    patient += OUT_LUNGS * (mask == 0)
    save(mask_path, mask + 16 * overlap(patient, nodules))
    