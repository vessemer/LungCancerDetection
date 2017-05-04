
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


# In[8]:


def hessian(field, coords):
    grad = gradient(field)
    axis = [[0, 1, 2], [1, 2], [2]]
    hess = [gradient(deriv, axis=j) 
            for i, deriv in enumerate(grad) 
            for j in axis[i]]

#   [(0, xx), (1, xy), (2, xz), (3, yy), (4, yz), (5, zz)]
#   x, y, z -> 3, 3, x, y, z -> 3, 3, N

    for j in range(len(hess)):
        hess[j] = hess[j][coords]

    return asarray([[hess[0], hess[1], hess[2]],
                    [hess[1], hess[3], hess[4]],
                    [hess[2], hess[4], hess[5]]])


def enhanced_filter(patient, coords, sigma):
    filtered = gaussian_filter(patient, sigma=sigma)
    hess = hessian(filtered, coords=coords)
    hess = [hess[:, :, i] for i in range(hess.shape[-1])]
    with Pool(CPU) as pool:
        eigs = pool.map(linalg.eigvalsh, 
                        hess)

    sigma_sqr = sigma ** 2
    z_dot = [sigma_sqr * (eig_val[2] ** 2) / abs(eig_val[0]) 
             if eig_val[0] < 0 
             and eig_val[1] < 0 
             and eig_val[2] < 0 
             else 0
             for eig_val in eigs]

    z_line = [sigma_sqr * abs(eig_val[1]) 
              * (abs(eig_val[1]) - abs(eig_val[2])) 
              / abs(eig_val[0]) 
              if eig_val[0] < 0 
              and eig_val[1] < 0 
              else 0
              for eig_val in eigs]
    return z_dot, z_line


def apply_enhs_filters(patient, mask, include_plane=False):
    sigmas = get_scales()
    enh_dot = zeros(mask.shape)
    enh_line = zeros(mask.shape)
    coords = where(mask)
    
    z_dot = list()
    z_line = list()
    for sigma in sigmas:
        dot, line = enhanced_filter(patient, coords, sigma)
        z_dot.append(dot)
        z_line.append(line)


    enh_dot[coords] = asarray(z_dot).max(axis=0)
    enh_line[coords] = asarray(z_line).max(axis=0)

    return enh_dot, enh_line

# In[9]:

def divergence(sigma, patient):
    grad = asarray(gradient(patient))
    grad /= norm(grad, axis=0) + 1e-3
    grad = [gaussian_filter(deriv, sigma=sigma) for deriv in grad]
    return sum([gradient(el, axis=i) 
                for i, el in enumerate(grad)], axis=0)


# In[10]:

def apply_divergence(masks_pats):
    divs_list = []
    for mask,pat,_ in tqdm(masks_pats):
        with Pool(CPU) as pool:
            _divs = pool.map(functools.partial(divergence, 
                                               patient=pat), sigmas)
        _divs = -1 * asarray(_divs) * mask 
        _divs = _divs.max(axis=0)
        divs_list.append(_divs.copy())
    return divs_list


# isolated nodules

# In[11]:

def is_in(colour, labe, dng_colours):
    if colour in dng_colours:
        return labe == colour


# In[37]:

def get_pure_isol(patient, mask, enh_dot):
    isolated = (patient > -600) * (mask > 0) * (enh_dot < 6) 
    labe, iso_nodules_num = label(isolated)
    volumes = bincount(labe.flatten())
    colours = where((volumes > ISOLATED_MIN_VOLUME) 
                & (volumes < ISOLATED_MAX_VOLUME))[0]
    
    isolated = zeros(isolated.shape).astype(bool)
    for colour in colours:
        isolated |= labe == colour
        
    return isolated, iso_nodules_num


# In[52]:

def get_pure_j_va(patient, mask, enh_line, iso):
    juxtavascular = (patient > -600) * (mask > 0) * (enh_line > 150)
    j_va_candidates = (1 - juxtavascular) * (1 - iso)
    labe, j_va_nodules_num = label(j_va_candidates)

    volumes = bincount(labe.flatten())
    colours = where((volumes > JUXTAVASCULAR_MIN_VOLUME) 
                    & (volumes < ISOLATED_MAX_VOLUME))[0]
    j_va = zeros(juxtavascular.shape).astype(bool)
    for colour in colours:
        j_va |= labe == colour
    
    return j_va, j_va_nodules_num

# In[61]:

def get_pure_j_pl(patient, mask, enh_dot):
    fixed_mask = morphology.binary_erosion(mask > 0,iterations=4)
    border_mask = fixed_mask * (1 - morphology.binary_erosion(fixed_mask > 0,iterations=4))
    juxtapleural = (patient > -400) * (border_mask > 0) * (enh_dot > 4)

    labe, j_pl_num = label(juxtapleural)
    volumes = bincount(labe.flatten())
    colours = where((volumes > JUXTAPLEURAL_MIN_VALUME) 
                    & (volumes < ISOLATED_MAX_VOLUME))[0]
    j_pl = zeros(juxtapleural.shape).astype(bool)
    for colour in colours:
        j_pl |= labe == colour
    return j_pl, j_pl_num

# In[64]:

def get_pure_nodules(patient, mask, enh):
    """
    Here: 
    1 is for isolated
    2 is for j_va
    4 is for j_pl
    """
    iso, iso_num = get_pure_isol(patient, mask, enh[0])
    j_va, j_va_num = get_pure_j_va(patient, mask, enh[1], iso)
    j_pl, j_pl_num = get_pure_j_pl(patient, mask, enh[0])
    return 2 * j_va + iso + 4 * j_pl, (iso_num, j_va_num, j_pl_num)


# In[65]:


def compute_candidates(masks_pats, enhs, 
                       divs_list=None, pure_only=True):
    nodules = list()
    amounts = list()
    for mp, enh, div in tqdm(zip(masks_pats, enhs, divs_list)):
        if pure_only:
            n1, n1_num = get_pure_nodules(mp[1], mp[0], enh, div)
        else:
            n1, n1_num = get_nodules(mp[1], mp[0], enh, div)
        nodules.append(n1)
        amounts.append(n1_num)
    return nodules, amounts


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


for i, patient_path in enumerate(patients):
    mask_path = join(PATH['STAGE_MASKS'], basename(patient_path) + '.npy')
    patient, mask = load_data(patient_path, mask_path)
    enhs = apply_enhs_filters(patient, mask,
                              include_plane=False)
    
    nodules, amounts = get_pure_nodules(patient, 
                                        mask, 
                                        enhs)
    
    save(join(paths, basename(patient_path)), nodules.astype(int8))
    save(join(paths, basename(patient_path) + 'amounts'), amounts)
    
