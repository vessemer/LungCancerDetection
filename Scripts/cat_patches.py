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
import pickle
import warnings
warnings.filterwarnings('ignore')


def read_ct(path, ret_xy_spacing=False, ret_original_format=True):
    patient = read_ct_scan(path)
    image = get_pixels_hu(patient)
#     image[image == image[0,0,0]] = 0
    
    if ret_original_format:
        return image, patient
    
    if ret_xy_spacing:
        return image, patient.GetSpacing()[0]
    
    return image


BORDER = 32
SPACING = array([.9, .7, .7])
BATCH_SIZE = 384
CPU = 24
UPSIDES = pickle.load(open(join(PATH['WEIGHTS'], 'upsides'), 'rb'))

def overlap(lung, mask):
#     iso = binary_dilation(imresize(isolated[163], (512, 512)))
    labeled, num = label(mask)
    coords = list()
    for colour in range(1, labeled.max() + 1):
        coords.append(where(labeled == colour))
    coords = array([[int(coord[0].mean() / SPACING[0])
                     for coord in coords], 
                    [int(coord[1].mean() / SPACING[1])
                     for coord in coords],
                    [int(coord[2].mean() / SPACING[2])
                     for coord in coords]])

    lung = pad(lung, 
               ((BORDER, BORDER), 
                (BORDER, BORDER), 
                (BORDER, BORDER)), 
               mode='edge')
    patches = list()
    for coord in coords.T:
        patch = lung[coord[0]: coord[0] + 2 * BORDER,
                     coord[1]: coord[1] + 2 * BORDER,
                     coord[2]: coord[2] + 2 * BORDER]
        patches.append(patch)

    return patches, coords


def operate(path, upsides=UPSIDES):
    lung, ct_lung = read_ct(path, ret_original_format=True)
    
    lung, spacing = resample(lung, ct_lung, SPACING)
    
    name = basename(path)
    mask = load(join(PATH['DATA_ENHANCED'], 
                     name + '.npy'))
    
    
    batch, coords = overlap(lung, mask)
    
        
    incorrects = list()
    if name in upsides:
        lung = flipud(lung)
        mask = flipud(mask)
        incorrects.append(-1)
        
    for patch, coord in zip(batch, coords.T):
        if patch.shape != (2 * BORDER, 
                           2 * BORDER, 
                           2 * BORDER):
            incorrects.append((path, coord))
            continue
        
        save(join(PATH['ENHANCED_CROPPED'], 
                  name + '_'.join([str(coord[0]), 
                                   str(coord[1]), 
                                   str(coord[2])])), 
             patch.astype(int16))
    return incorrects


def get_remind_files():
    file_list = set(glob(join(PATH['DATA'], '*')))
    file_list = file_list.difference(set([join(PATH['DATA'], basename(path).split('.npy')[0][:32]) 
                            for path in glob(join(PATH['ENHANCED_CROPPED'], '*.npy'))]))

    return sorted(list(file_list))


incorrects = list()
file_list = get_remind_files()
for counter in range(len(file_list) // BATCH_SIZE + 1):
    
    batch_files = file_list[BATCH_SIZE * counter:
                            BATCH_SIZE * (counter + 1)]
    with Pool(CPU) as pool:
        incorrect = pool.map(operate, batch_files)
    incorrects += incorrect
  
    pickle.dump(incorrects, 
                open(join(PATH['WEIGHTS'], 
                          'incorrects'), 
                     'wb'))