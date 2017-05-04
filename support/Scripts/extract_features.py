import sys
sys.path.append('../')
sys.path.append('../support/')
from scipy.ndimage.measurements import label
from scipy.ndimage import interpolation
from time import time
from glob import glob
import timeit
from os.path import join, basename, isfile
from tqdm import tqdm
from paths import *
from ct_reader import *
import dicom
from scipy.misc import imresize
from multiprocessing import Pool
import pickle
from paths import *
from scipy.ndimage import morphology
# import seaborn as sns
import pandas as pd
from numpy import *


# In[2]:

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

def label_nodules(enhanced):
    isolated = enhanced.copy()
    isolated[(isolated == 4)
            |(isolated == 2)
            |(isolated == 6)] = 0
    isolated, _ = label(isolated)

    vascular = enhanced.copy()
    vascular[(vascular == 1)
            |(vascular == 2)
            |(vascular == 3)] = 0
    vascular, _ = label(vascular)

    plural = enhanced.copy()
    plural[(plural == 1)
          |(plural == 4)
          |(plural == 5)] = 0
    plural, _ = label(plural)
    return isolated, vascular, plural


# In[4]:

def mask_features(mask,sp_mask):
    volumes = bincount(mask.flatten())
    zone_volumes = bincount(sp_mask.flatten())
    ans = dict()
    for i in range(16):
        try:
            ans['volume' + str(i)] = volumes[i]
        except:
            ans['volume' + str(i)] = 0 
    for i in range(7):
        ans['z_volume' + str(i)] = zone_volumes[i]
    ans['l//r'] = volumes[1]  / volumes[2] if(volumes[2]) else 0.0
    ans['lungoverlap//l'] = volumes[3] / volumes[1] if(volumes[1]) else 0.0
    ans['br_overlap//l'] = volumes[5] / volumes[1] if(volumes[1]) else 0.0
    ans['br_overlap//r'] = volumes[6] / volumes[2] if(volumes[2]) else 0.0
    ans['tr_overlap//l'] = volumes[9] / volumes[1] if(volumes[1]) else 0.0
    ans['tr_overlap//r'] = volumes[10] / volumes[2] if(volumes[2]) else 0.0
    ans['br_tr_overlap//tr'] = volumes[12] / volumes[8] if(volumes[8]) else 0.0
    ans['z_volume_1//2'] = zone_volumes[1] / zone_volumes[2]
    ans['z_volume_2//3'] = zone_volumes[2] / zone_volumes[3]
    ans['z_volume_4//5'] = zone_volumes[4] / zone_volumes[5]
    ans['z_volume_5//6'] = zone_volumes[5] / zone_volumes[6]
    return ans


# In[5]:

def if_left(mask):
    return in1d(mask,[1,3,5,7,9,11,13,15]).reshape(mask.shape)
            
def if_right(mask):
    return in1d(mask,[2,3,6,7,10,11,14,15]).reshape(mask.shape)

def split_mask(mask):
    mn1 = where(if_left(mask))[0].min()
    mx1 = where(if_left(mask))[0].max()
    mn2 = where(if_right(mask))[0].min()
    mx2 = where(if_right(mask))[0].max()
    height1 = int((mx1-mn1)/3.0)
    height2 = int((mx2-mn2)/3.0)
    mask_zones = zeros(mask.shape)
    mask_zones[mn1:mn1+height1,:,:] = 1 
    mask_zones[mn1+height1:mn1+2*height1,:,:] = 2
    mask_zones[mn1+2*height1:mx1,:,:] = 3
    mask_l = if_left(mask)*mask_zones
    mask_zones = zeros(mask.shape)
    mask_zones[mn2:mn2+height2,:,:] = 4
    mask_zones[mn2+height2:mn2+2*height2,:,:] = 5
    mask_zones[mn2+2*height2:mx2,:,:] = 6
    return (mask_l + if_right(mask) * mask_zones).astype('int8')


# In[6]:

def merge(enhanced, mask):
    return 8 * mask + enhanced


# In[7]:

def collect_stats(enhanced,mask,sp_mask):
    prev_time = time()
    l_enhanced = enhanced * if_left(mask)
    r_enhanced = enhanced * if_right(mask)
 
    
#     print('split_mask ',time()-prev_time)
#     prev_time = time()
    
    enh_areas = bincount(enhanced.flatten())[1:]
    enh_l_areas = bincount(l_enhanced.flatten())[1:]
    enh_r_areas = bincount(r_enhanced.flatten())[1:]
    
    enh_areas_zones = list()
    for i in range(1,7):
        enh_areas_zones.append(bincount((enhanced * (sp_mask == i)).flatten())[1:])
#     enh_l_areas = concatenate((enh_areas_zones[1][enh_areas_zones[1]>0],
#                               enh_areas_zones[2][enh_areas_zones[2]>0],
#                               enh_areas_zones[0][enh_areas_zones[0]>0]))
#     enh_r_areas = concatenate((enh_areas_zones[4][enh_areas_zones[4]>0],
#                               enh_areas_zones[5][enh_areas_zones[5]>0],
#                               enh_areas_zones[3][enh_areas_zones[3]>0]))
#     enh_areas = concatenate((enh_l_areas,enh_r_areas))
#     print('bincounts ',time()-prev_time)
#     prev_time = time()
    
    if not enh_areas.shape[0]:
        max_areas = dict()
        for i in range(5):
            max_areas['max'+str(i)] = 0
            max_areas['max_l'+str(i)] = 0
            max_areas['max_r'+str(i)] = 0
        zone_feats = dict()
        for i in range(6):
            zone_feats['amoun_z' + str(i+1)] = 0
            zone_feats['sumarea_z' + str(i+1)] = 0
        enh_comps_after_dil = dict()
        for i in range(20):
            enh_comps_after_dil['comps_dil'+str(i)] = 0
            enh_comps_after_dil['comps_dil_l'+str(i)] = 0
            enh_comps_after_dil['comps_dil_r'+str(i)] = 0
        ans = dict((('areas', 0), ('amoun', 0), 
                     ('mean', 0), ('std', 0), ('median', 0), 
                     ('mean_not_min', 0), 
                     ('median_not_min', 0), 
                     ('modes', [0] * 9)))
        ans.update(max_areas)
        ans.update(enh_comps_after_dil)
        ans.update(mask_features(mask,sp_mask))
        ans.update(zone_feats)
        return ans
    
    enh_amoun = enh_areas[enh_areas > 0].shape[0]
    enh_amoun_l = enh_l_areas[enh_l_areas > 0].shape[0]
    enh_amoun_r = enh_r_areas[enh_r_areas > 0].shape[0]
    enh_amoun_zones = [x[x > 0].shape[0] for x in enh_areas_zones]
    enh_area_sum_zones = [x[x > 0].sum() for x in enh_areas_zones]
    
    zone_feats = dict()
    for i in range(6):
        zone_feats['amoun_z' + str(i+1)] = enh_amoun_zones[i]
        zone_feats['sumarea_z' + str(i+1)] = enh_area_sum_zones[i]
    
    enh_mean = mean(enh_areas)
    enh_std = std(enh_areas)
    enh_sort_areas = sorted(enh_areas[enh_areas > 0],reverse=True)
    enh_sort_areas_l = sorted(enh_l_areas[enh_l_areas > 0],reverse=True)
    enh_sort_areas_r = sorted(enh_r_areas[enh_r_areas > 0],reverse=True)
    max_areas = dict()
    for i in range(5):
        try:
            max_areas['max'+str(i)] = enh_sort_areas[i]
        except:
            max_areas['max'+str(i)] = 0 
        try:
            max_areas['max_l'+str(i)] = enh_sort_areas_l[i]
        except:
            max_areas['max_l'+str(i)] = 0    
        try:
            max_areas['max_r'+str(i)] = enh_sort_areas_r[i]
        except:
            max_areas['max_l'+str(i)] = 0
    
    enh_median = median(enh_areas)
    enh_mean_not_min = enh_areas[enh_areas != enh_areas.min()].mean()
    enh_median_not_min = median(enh_areas[enh_areas != enh_areas.min()])
    modes = [2, 3, 4, 5, 6, 9, 12, 19, 37, 1e7]
    enh_modes = [sum((enh_areas >= modes[i - 1]) 
                 & (modes[i] > enh_areas))
                for i in range(1, len(modes))]
    
#     print('stats ',time()-prev_time)
#     prev_time = time()
    
    img = enhanced.copy()
    enh_comps_after_dil = dict()
    iter_num = 1
    for i in range(iter_num):
        labeled,label_num = label(img)
        enh_comps_after_dil['comps_dil'+str(i)] = label_num
        enh_comps_after_dil['comps_dil_l'+str(i)] = len(unique(labeled*if_left(mask)))
        enh_comps_after_dil['comps_dil_r'+str(i)] = len(unique(labeled*if_right(mask)))
        img = morphology.binary_dilation(img,structure=ones((5,5,5)))
    labeled,label_num = label(img)
    enh_comps_after_dil['comps_dil'+str(iter_num)] = label_num
    enh_comps_after_dil['comps_dil_l'+str(iter_num)] = len(unique(labeled*if_left(mask)))
    enh_comps_after_dil['comps_dil_r'+str(iter_num)] = len(unique(labeled*if_right(mask)))

#     print('dil ',time()-prev_time)
#     prev_time = time()
    
    
    ans = dict((('areas', sum(enh_areas)), ('amoun', enh_amoun), 
                 ('mean', enh_mean), ('std', enh_std), ('median', enh_median), 
                 ('mean_not_min', enh_mean_not_min), 
                 ('median_not_min', enh_median_not_min),
                 ('modes', enh_modes)))
    ans.update(max_areas)
    ans.update(enh_comps_after_dil)
    ans.update(mask_features(mask,sp_mask))
    ans.update(zone_feats)

#     print('mask_feats ',time()-prev_time)
#     prev_time = time()
    
    return ans


# In[9]:

def operate(path):
    try:
        enhanced = load(join(PATH['STAGE_ENHANCED'], 
                             path + '.npy'))
        mask = load(join(PATH['STAGE_MASKS'], 
                             path + '.npy'))

        zoomfactor = [w / float(f) for w, f in zip(enhanced.shape, mask.shape)]
        mask = interpolation.zoom(mask, zoom=zoomfactor, order = 0, mode = 'nearest')
        isolated, vascular, plural = label_nodules(enhanced)
        sp_mask = split_mask(mask)
        save(join(PATH['STAGE_MASKS'], path), merge(enhanced,mask))
        return (path, collect_stats(isolated,mask,sp_mask)),\
                (path, collect_stats(vascular,mask,sp_mask)),\
                (path, collect_stats(plural,mask,sp_mask))
    except:
        pass
        return ((path, None), (path, None), (path, None))


# In[ ]:

patients = set([basename(path)[:32] for path in glob(join(PATH['STAGE_ENHANCED'], '*'))])
patients = patients.difference(pickle.load(open(join(PATH['STAGE_MASKS'], 'still_erroneus_ncrash'), 'rb')))
stats = list()
CPU = 8
#print('Start. ', len(patients))
with Pool(CPU) as pool:
    stats = pool.map(operate, list(patients))
    
#print('Done.')
path = join(PATH['STAGE_MASKS'], 'DATAFRAMES')
pickle.dump(stats, open(join(path, 'merged_stats_100'), 'wb'))

