import sys
sys.path.append('../')
sys.path.append('../support/')
sys.path.append('../lung_segmentation/')
import os
import SimpleITK
from lung_separation import *
from ct_reader import *
from os.path import join, basename, isfile
from glob import glob
from skimage import filters
from skimage import exposure
from tqdm import tqdm
from skimage import morphology as skm
from skimage.morphology import watershed

from scipy.ndimage import generate_binary_structure
from skimage import measure
from multiprocessing import Pool
import pickle
from segmentation import *
from numpy import *

import SimpleITK as sitk
from paths import * 
from skimage.transform import resize
from scipy.ndimage import label
from scipy.ndimage import morphology

def read_ct(path, ret_xy_spacing=False):
    patient = read_ct_scan(path)
    image = get_pixels_hu(patient)
    image[image == image[0,0,0]] = 0
    
    if ret_xy_spacing:
        try:
            return image, patient[0].PixelSpacing[0]
        except AttributeError:
            return image, scan.GetSpacing()[0]
    
    return image

# In[3]:

NUM_SEED_SLICES = 25
BRONCHIAL_THRESHOLD = -900
INITIAL_THRESHOLD = -950
STEP = 64
CPU = 24


# properties:  
# 
#     average HU below −950,  
#     minimum size of 50 mm^2 ,   
#     maximum size of 1225 mm^2 ,   
#     mean x- and y-coordinates not further than 30%   
#     of the x- and y-dimensions of the image away from the center of the slice.  

# In[4]:

def region_growing(img, seed, minthr, maxthr, structure=None):
    """code was taken from:
    https://github.com/loli/medpy/wiki/Basic-image-manipulation
    """
    img[seed] = minthr
    thrimg = (img < maxthr) & (img >= minthr)
    lmap, _ = label(thrimg, structure=structure)
    lids = unique(lmap[seed])
    region = zeros(img.shape, bool_)
    for lid in lids:
        region |= lmap == lid
    return region


# In[5]:

def extract_bronchial(ct_slice, xy_spacing):
    labeled = measure.label(ct_slice < BRONCHIAL_THRESHOLD)
    areas = bincount(labeled.flatten())
    labels = [i 
              for i, area in enumerate(areas) 
              if (area * xy_spacing >= 50) and (area * xy_spacing <= 1225)]
    coords = [where(labeled == i) for i in labels]

    center = array(ct_slice.shape) // 2
    max_dist = array(ct_slice.shape) * .3
    labels = [(mean(coord, axis=1), labe)
              for labe, coord in zip(labels, coords) 
              if (abs(center - mean(coord, axis=1)) < max_dist).all()]
    
    if len(labels) != 0:
        return labeled == min(labels, key=lambda x: sum((x[0] - center) ** 2))[1]
    
    return None


# In[6]:

def select_bronchial(bronchials, ct_slices, levels):
    center = array(bronchials[0].shape) // 2
    coords = [(mean(where(bool_slice), axis=1), i) 
              for i, bool_slice in enumerate(bronchials)]
    el = min(coords, key=lambda x: sum((x[0] - center) ** 2))
    return bronchials[el[1]], ct_slices[el[1]], levels[el[1]]


def select_seeds(bronchial, ct_clice):
    return ct_clice * bronchial == ct_clice[bronchial].min()


# In[7]:

def extract_seeds(patient, xy_spacing):
    bronchials = list()
    bronch_cts = list()
    levels = list()
    for i in range(55):
        bronchial = extract_bronchial(patient[i], xy_spacing)
        if bronchial is not None:
            bronchials.append(bronchial)
            bronch_cts.append(patient[i])
            levels.append(i)

            
    for i in range(-55, 0, 1):
        bronchial = extract_bronchial(patient[i], xy_spacing)
        if bronchial is not None:
            bronchials.append(bronchial)
            bronch_cts.append(patient[i])
            levels.append(i)
            
    bronchial, ct_slice, level = select_bronchial(bronchials, 
                                                  bronch_cts, 
                                                  levels)
    
    seeds = zeros(patient.shape)
    seeds[level] = select_seeds(bronchial, ct_slice)

    return seeds


# In[8]:

def growing_bronchis(patient, seeds, 
                     threshold=INITIAL_THRESHOLD, 
                     step=STEP,
                     full_extraction=True):
    
    
    seeds = seeds.astype(bool_)
    seeds = region_growing(patient.copy(), seeds, -5010, threshold)
    volume = count_nonzero(seeds)
    
    lungs_thresh = filters.threshold_otsu(patient[patient.shape[0] // 2])
    
    ret = None
    while True:    
        labeled = region_growing(patient.copy(), seeds, -5010, threshold + step)
        new_volume = count_nonzero(labeled)
        if new_volume >= volume * 2:
            if step == 4:
                ret = seeds.copy()
                if not full_extraction:
                    return ret
                
            if step == 2:
                return ret, seeds
            step = ceil(step * 0.5)
            continue
        
        threshold += step
        volume = new_volume
        seeds = labeled
        
        if threshold >= lungs_thresh:
            if ret is None:
                ret = seeds.copy()
            
            if not full_extraction:
                return ret
            
            return ret, seeds


# In[9]:

def grow_lungs(patient, seeds):
    lungs_seeds = patient * seeds == patient[seeds].min()
    lungs_seeds = lungs_seeds.astype(bool_)
    threshold = filters.threshold_otsu(patient[patient.shape[0] // 2])

    lungs_seeds = region_growing(patient.copy(), lungs_seeds, -1010, threshold)
    return morphology.binary_opening(lungs_seeds - morphology.binary_opening(seeds))


def lung_separation(patient, lungs_seeds):
    labeled = label(lungs_seeds)[0]
    markers = bincount(labeled.flatten())
    markers = vstack([markers[1:], arange(1, markers.shape[0])])
    markers = asarray(sorted(markers.T, key=lambda x: x[0]))[-2:]
    if len(markers) < 2:
        left, right = separate_lungs3d(lungs_seeds)
        return left, right, True
    
    if markers[0, 0] / markers[1, 0] < 0.3:
        left, right = separate_lungs3d(lungs_seeds)
        return left, right, True
    
    centroids = (mean(where(labeled == markers[0, 1]), axis=1)[-1],
                 mean(where(labeled == markers[1, 1]), axis=1)[-1])
    
    if centroids[0] > centroids[1]:
        return labeled == markers[1, 1], labeled == markers[0, 1], False
    
    return labeled == markers[0, 1], labeled == markers[1, 1], False


def lungs_postprocessing(lungs_seeds):
    for i in range(lungs_seeds.shape[1]):
        lungs_seeds[:, i] = morphology.binary_fill_holes(lungs_seeds[:, i])
    return lungs_seeds


# In[10]:

def plot_lungs_structures(left, right, trachea, bronchi, patient=None, plot=True):
    """
    Structure:
        1.  left lung
        2.  right lung
        4.  bronchi
        8.  trachea
        
        3.  left    overlapped by right
        
        5.  bronchi overlapped by left
        6.  bronchi overlapped by right
        7.  bronchi overlapped by right, overlapped by left
        
        9.  trachea overlapped by left
        10. trachea overlapped by right
        11. trachea overlapped by right, overlapped by left
            
        12. bronchi overlapped by trachea
        13. bronchi overlapped by trachea, overlapped by left
        14. bronchi overlapped by trachea, overlapped by right
        15. bronchi overlapped by trachea, overlapped by right, overlapped by left
    """
    if plot:
        figure(figsize=(10,10))
        subplot(131)
        combined = left + trachea * 2 + bronchi * 4 + right * 8 
        imshow(combined[right.shape[0] // 2]);
        subplot(132)
        imshow(resize(combined[:, right.shape[1] // 2].astype(float64), 
                      [right.shape[1], right.shape[1]]));
        if patient is not None:
            subplot(133)
            imshow(patient[right.shape[0] // 2])
            show()
    return left + right * 2 + bronchi * 4 + trachea * 8


# In[11]:

def conventional_lung_segmentation(patient, xy_spacing, plot=True):
    seeds = extract_seeds(patient, xy_spacing)
    trachea, bronchi = growing_bronchis(patient, seeds)
    
#     print('-'*30)
#     print('Bronchisare grown')
    lungs_seeds = grow_lungs(patient, trachea)
    left, right, state = lung_separation(patient, lungs_seeds)
    selem = skm.ball(int(patient.shape[-1] * .01))
    
#     print('-'*30)
#     print('Lungs are grown & separated')
    left = skm.binary_closing(left, selem)
    right = skm.binary_closing(right, selem)

    right = morphology.binary_fill_holes(right)
    left = morphology.binary_fill_holes(left)

    left = lungs_postprocessing(left)
    right = lungs_postprocessing(right)
    
#     print('-'*30)
#     print('Smoothing has been applied')
    return plot_lungs_structures(left, 
                                 right, 
                                 trachea, 
                                 bronchi, 
                                 patient,
                                 plot).astype(int8), state


# In[12]:

def operate(path, out_dir='STAGE_MASKS'):
    patient, xy_spacing = read_ct(path, True)
    combined, state = conventional_lung_segmentation(patient, xy_spacing, False)
    if 'luna' in out_dir.lower():
        save(join(PATH[out_dir], basename(path).split('.mhd')[0]), combined)
    else:
        save(join(PATH[out_dir], basename(path)), combined)
    if isfile(join(PATH[out_dir], 'manual.npy')):
        manuals = load(join(PATH[out_dir], 'manual.npy')).tolist()
    else:
        manuals = list()
        
    if state:
        manuals.append(path)
    save(join(PATH[out_dir], 'manual'), asarray(manuals))
    if state:
        return path
    return None


# In[82]:

file_list = set(basename(path) for path in glob(join(PATH['STAGE_DATA'], '*')))
file_list = file_list.difference([basename(path).split('.npy')[0] 
                                  for path in glob(join(PATH['STAGE_MASKS'], '*.npy'))])
file_list = list(reversed([join(PATH['STAGE_DATA'], base_name) 
             for base_name in file_list]))

manuals = list()
with Pool(CPU) as pool:
    manuals = pool.map(operate,  file_list)

#manuals += load(join(PATH['LUNA_MASKS'], 'manual.npy')).tolist()
save(join(PATH['STAGE_MASKS'], 'manual_pool'), asarray(manuals))


