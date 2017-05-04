import sys
sys.path.append('../')
sys.path.append('../support/')
sys.path.append('../Nets/')
sys.path.append('../lung_segmentation/')
from glob import glob
import timeit
from skimage.transform import resize
from os.path import join, basename, isfile
from tqdm import tqdm
from paths import *
from ct_reader import *
import dicom
from scipy.misc import imresize
from skimage import exposure
from paths import *
from ResNet import *
from numpy import *

# In[2]:

def load_mask(path):
    if len(path.split('.mhd')) == 2:
        return load(join(PATH['LUNA_OUT'], basename(path).split('.mhd')[0] + '.npy'))
    else:
        return load(join(PATH['DATA_OUT'], basename(path) + '.npy'))


# In[3]:

def read_ct(path, ret_xy_spacing=False):
    patient = read_ct_scan(path)
    image = get_pixels_hu(patient)
    
    if ret_xy_spacing:
        return image, patient.GetSpacing()[0]
    
    return image


# In[4]:

def crop_lungs(lungs, mask):
    mask[(mask == 4) 
         | (mask == 12) 
         | (mask == 8)] = 0

    mask[(mask == 1) 
         | (mask == 5) 
         | (mask == 9)
         | (mask == 13)] = 1

    mask[(mask == 2) 
         | (mask == 6) 
         | (mask == 10)
         | (mask == 14)] = 1

    mask[(mask == 3) 
         | (mask == 7) 
         | (mask == 15)] = 0
    
    x, y, z = where(mask)
    min_x, min_y, min_z = min(x), min(y), min(z)
    max_x, max_y, max_z = max(x), max(y), max(z)
    return lungs[min_x: max_x, 
                 min_y: max_y, 
                 min_z: max_z]


# In[5]:

def get_extractor():
    model, bottle_neck = ResNet50()
    bottle_neck = load_weighs(bottle_neck, include_top=False)
    return bottle_neck


# In[6]:

def select_slice(patient, i, axes=0):
    if axes == 0:
        return patient[i]
    if axes == 1:
        return patient[:, i]
    if axes == 2:
        return patient[:, :, i]
    


# In[7]:

def get_data(path, axes=0):
    patient = read_ct(path)
    mask = load_mask(path)
    patient = crop_lungs(patient, mask)
    patient = clip(patient, -1000, 700)

    batch = list()
    for i in range(0, patient.shape[axes] - 9, 3):
        tmp = list()
        for j in range(3):
            
            img = select_slice(patient, i + 3 * j, axes)
        
            for k in range(1, 3):
                img = vstack([expand_dims(select_slice(patient, 
                                                       i + j + 3 * j, 
                                                       axes), 
                                          0), 
                              expand_dims(img, 0)]).max(axis=0)
                
            img = 255.0 / amax(img) * img
#             img = exposure.equalize_hist(img.astype(np.uint8))
            img = imresize(img, 
                           (224, 224), 
                           interp='bicubic')
            tmp.append(img)

        batch.append(array(tmp))

    return array(batch)


# In[90]:

#imshow(swapaxes(swapaxes(c[19], 0, 2), 1, 0));


# In[107]:

#imshow(swapaxes(swapaxes(c[4], 0, 2), 1, 0));


# In[11]:

def calc_features(patients, net=None, batch_size=1, post_dir='MAX_TOP', axes=0):
    if net is None:
        net = get_extractor()
        
    for folder in patients:
        batch = get_data(folder, axes=axes)
        feats = list()
        for i in range(len(batch) // batch_size + 1):
            tmp = net.predict(batch[i * batch_size: 
                                    (i + 1) * batch_size])
            if len(tmp):
                feats.append(tmp)
        feats = vstack(feats)
        save(join(PATH['DATA_MXNET'], 
                  post_dir, 
                  '.'.join(basename(folder).split('.')[:-1])), 
             feats)



# In[12]:

def get_remined_files(post_dir='MAX_TOP'):
    file_list = set(basename(path) 
                    for path in glob(join(PATH['DATA'], '*')))

    file_list = file_list.difference([basename(path).split('.npy')[0] 
                                      for path in glob(join(PATH['DATA_MXNET'], 
                                                            post_dir, '*.npy'))])
    file_list = [join(PATH['DATA'], base_name) 
                 for base_name in file_list]
    return file_list


# In[13]:

net = get_extractor()


# In[ ]:

file_list = get_remined_files()
while len(file_list):
#     try:
        calc_features(file_list, net, 
                      batch_size=32, 
                      post_dir='MAX_TOP', 
                      axes=0)
        file_list = get_remined_files(post_dir='MAX_TOP')
        
#     except:
#         pass
#         continue
