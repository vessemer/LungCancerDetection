import sys
sys.path.append('../Nets/')
from glob import glob
from os.path import join
from multiprocessing import Pool
from scipy.ndimage.interpolation import rotate
from keras.callbacks import ModelCheckpoint
from tqdm import tqdm
from functools import partial
from Nodule import *
from numpy import *


PATH = {
    'DATA': '/home/a.dobrenkii/Projects/Kaggle/DataScienceBowl2K17/data/TRAIN',
    'DATA_OUT': '/home/a.dobrenkii/Projects/Kaggle/DataScienceBowl2K17/data/TRAIN_OUT',
    'CSV': '/home/a.dobrenkii/Projects/Kaggle/DataScienceBowl2K17/data/CSV',
    'LABELS': '/home/a.dobrenkii/Projects/Kaggle/DataScienceBowl2K17/data/CSV/stage1_labels.csv',

    'LUNA': '/fasthome/a.dobrenkii/LUNA',
    'LUNA_DATA': '/fasthome/a.dobrenkii/LUNA/DATA',
    'LUNA_SOBEL': '/fasthome/a.dobrenkii/LUNA/SOBEL_IMG',
    'LUNA_LUNGS': '/fasthome/a.dobrenkii/LUNA/LUNGS_IMG',
    'LUNA_MASKS': '/fasthome/a.dobrenkii/LUNA/MASKS',
    'LUNA_CSV': '/fasthome/a.dobrenkii/LUNA/CSVFILES',
    'LUNA_PRED': '/fasthome/a.dobrenkii/LUNA/PRED',
    'PATCH_PATHS': '/home/a.dobrenkii/Projects/Kaggle/DataScienceBowl2K17/data/LUNA/OUT/PATCHES',
    'LUNA_NODULES': '/home/a.dobrenkii/Projects/Kaggle/DataScienceBowl2K17/data/LUNA/OUT/PATCHES/NODULES',
    'LUNA_VESSELS': '/home/a.dobrenkii/Projects/Kaggle/DataScienceBowl2K17/data/LUNA/OUT/PATCHES/VESSELS',

    'WEIGHTS': '/home/a.dobrenkii/Projects/Kaggle/DataScienceBowl2K17/data/WEIGHTS',
    'CHECKPOINTS': '/home/a.dobrenkii/Projects/Kaggle/DataScienceBowl2K17/data'
}


CPU = 10
ANGLE = 35
SHIFT = 4
SHAPE = (8, 64, 64)
TRAIN_TEST = .2
NB_EPOCH = 50


model, bottle_neck = dim_concentration()
model.compile('adam', 'mse')


def augment_patch(patch, shape, angle=15, shift=4):
    
    if angle:
        shift = random.randint(-shift, shift, 3)
        patch = rotate(patch, random.uniform(-angle, angle), axes=[1,2])
        patch = rotate(patch, random.uniform(-angle, angle), axes=[0,1])
        patch = rotate(patch, random.uniform(-angle, angle), axes=[0,2])
    
    center = (array(patch.shape) // 2) + shift
    
    left = array(shape) // 2
    right = array(shape) - left
    patch = patch[center[0] - left[0]:center[0] + right[0], 
                  center[1] - left[1]:center[1] + right[1], 
                  center[2] - left[2]:center[2] + right[2]]
    
    mn = patch.min()
    mx = patch.max()
    if (mx - mn) != 0:
        patch = (patch - mn) / (mx - mn)
    else:
        patch[:, :, :] = 0.
    
    return patch


def batch_generator(patch_paths, batch_size, shape=(8, 64, 64), angle=15, shift=4, CPU=4):
    number_of_batches = ceil(len(patch_paths) / batch_size)
    counter = 0
    
    random.shuffle(patch_paths)
    
    while True:
        batch_files = patch_paths[batch_size * counter:batch_size * (counter + 1)]        
        
        patch_list = [load(patch_path) for patch_path in batch_files]

        augment = partial(augment_patch, shape=shape, angle=angle, shift=shift)
        with Pool(CPU) as pool:
            patch_list = pool.map(augment, patch_list)
            
        counter += 1
        yield expand_dims(array(patch_list), 1), expand_dims(array(patch_list), 1)
        
        if counter == number_of_batches:
            random.shuffle(patch_paths)
            counter = 0
            
            
# patch_paths = glob(join(PATH['LUNA_NODULES'], '*_patch.npy')) 
# patch_paths += glob(join(PATH['LUNA_VESSELS'], '*_patch.npy')) 
# shuffle(patch_paths)
# save(join(PATH['PATCH_PATHS'], 'LUNA'), array(patch_paths))


patch_paths = load(join(PATH['PATCH_PATHS'], 'LUNA.npy'))

train = patch_paths[int(len(patch_paths) * .2):]
valid = patch_paths[:int(len(patch_paths) * .2)]
SAMPLES_PER_EPOCH = len(train) 
NB_VAL_SAMPLES = len(valid)



train_generator = batch_generator(train, 
                                  batch_size=32, 
                                  shape=SHAPE, 
                                  angle=ANGLE, 
                                  shift=SHIFT, 
                                  CPU=CPU)

valid_generator = batch_generator(valid, 
                                  batch_size=32, 
                                  shape=SHAPE, 
                                  angle=0, 
                                  shift=0, 
                                  CPU=CPU)



checkpoint = ModelCheckpoint(filepath=join(PATH['WEIGHTS'], '3DCAE_nodule_model'), 
                             verbose=1, 
                             save_best_only=True)


model.fit_generator(train_generator, 
                    samples_per_epoch=1853, 
                    nb_epoch=NB_EPOCH, 
                    callbacks=[checkpoint], 
                    validation_data=valid_generator, 
                    class_weight=None, 
                    nb_val_samples=463,
                    nb_worker=1)