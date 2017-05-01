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
from scipy.ndimage import morphology
from skimage import filters
from skimage import exposure
from tqdm import tqdm
from skimage import morphology as skm
from skimage.morphology import watershed
from scipy.ndimage import label
from scipy.ndimage import generate_binary_structure
from skimage import measure
from multiprocessing import Pool
import pickle
from pure_ws_segmentation import *
from numpy import *
# import warnings
# warnings.filterwarnings('ignore')

import SimpleITK as sitk
from paths import * 
from skimage.transform import resize


erroneus = sorted(list(set(pickle.load(open(join(PATH['STAGE_MASKS'], 'erroneus'), 'rb')))))
erroneus = [join(PATH['STAGE_DATA'],err) for err in erroneus]

def operate(path, out_dir='STAGE_<ASKS'):
    ct_scan = read_ct_scan(path)
    ct_scan_px = get_pixels_hu(ct_scan)

    with Pool(34) as pool:
        ct_excluded = pool.map(exclude_lungs, ct_scan_px)

    # end = time.time()
    # print(end - start)

    lung_filter = asarray(ct_excluded)
    a128 = lung_filter.min()
    a255 = lung_filter.max()
    lung_filter[lung_filter==a128] = 0
    lung_filter[lung_filter==a255] = 1
    left, right = separate_lungs3d(lung_filter)
    save(join(join('/home/a.dobrenkii/Projects/Kaggle/DataScienceBowl2K17/data/NEW_STAGE/MASKS','FIXED'),basename(path)),left+2*right)

for err in erroneus:
    operate(err)
