import sys
sys.path.append('../')
sys.path.append('../Nets/')
from glob import glob
from os.path import join
from multiprocessing import Pool
from scipy.ndimage.interpolation import rotate
from keras.callbacks import ModelCheckpoint, TensorBoard
from tqdm import tqdm
from functools import partial
from Nodule import *
from numpy import *
from paths import *

CPU = 24
TEST_FOLD = str(0)
VALID_FOLD = str(1)
ANGLE = 180
SHIFT = 8
NODULE_OVERSAMPLING = 81
BATCH_SIZE = 50
SHAPE = (16, 48, 48)
TRAIN_TEST = .2
NB_EPOCH = 20


def cae_augment_patch(patch, shape, angle=15, shift=4):
#     patch = load(patch)
    center = (array(patch.shape) // 2) 
    
    if angle:
        shift = random.randint(-shift, shift, 3)
        center += shift
        angle = random.uniform(-angle, angle, 3)
        patch = rotate(patch, angle[0], axes=[1,2])
        patch = rotate(patch, angle[1], axes=[0,1])
        patch = rotate(patch, angle[2], axes=[0,2])
    
    
    
    left = array(shape) // 2
    right = array(shape) - left
    patch = patch[center[0] - left[0]:center[0] + right[0], 
                  center[1] - left[1]:center[1] + right[1], 
                  center[2] - left[2]:center[2] + right[2]]
    
    
    return patch


def cae_batch_generator(patch_paths, 
                        batch_size, shape=(16, 48, 48), 
                        angle=15, shift=4, CPU=24):
    
    number_of_batches = ceil(len(patch_paths) / batch_size)
    counter = 0
    
    random.shuffle(patch_paths)
    
    while True:
        batch_files = patch_paths[batch_size * counter:
                                  batch_size * (counter + 1)]
        
        with Pool(CPU) as pool:
            patch_list = pool.map(load, batch_files) 
        
        augment = partial(cae_augment_patch, shape=shape, angle=angle, shift=shift)
        with Pool(CPU) as pool:
            patch_list = pool.map(augment, patch_list)
            
        patch_list = expand_dims(array(patch_list), -1)
        patch_list = clip(patch_list, -2000, 400)
        patch_list = (patch_list - patch_list.min()) / (patch_list.max() - patch_list.min())
        
        counter += 1
        
        yield patch_list, patch_list
        
        if counter == number_of_batches:
            random.shuffle(patch_paths)
            counter = 0


model, bottle_neck = dim_concentration()
model.compile('adam', 'mse')
model.load_weights(join(PATH['WEIGHTS'], '3DCAE_nodule'))


file_list = set([path for path in glob(join(PATH['LUNA_NODULES'], 'subset*', '*.npy'))] 
              + [path for path in glob(join(PATH['LUNA_VESSELS'], 'subset*', '*.npy'))])

test = [path for path in file_list if ''.join(['subset', TEST_FOLD]) in path]
# file_list = list(file_list.difference(test))
valid = [path for path in file_list if ''.join(['subset', VALID_FOLD]) in path]
train = list()
for path in list(set(file_list).difference(valid[len(valid) // 2:])):
    if 'nodule' in path.lower():
        train += [path] * NODULE_OVERSAMPLING
    else:
        train.append(path)
valid = valid[: len(valid) // 2]



train_generator = cae_batch_generator(train, 
                                      batch_size=BATCH_SIZE, 
                                      shape=SHAPE, 
                                      angle=ANGLE, 
                                      shift=SHIFT, 
                                      CPU=CPU)

valid_generator = cae_batch_generator(valid, 
                                      batch_size=BATCH_SIZE, 
                                      shape=SHAPE, 
                                      angle=0, 
                                      shift=0, 
                                      CPU=CPU)


checkpoint = ModelCheckpoint(filepath=join(PATH['WEIGHTS'], '3DCAE_nodule'), 
                             verbose=1, 
                             save_best_only=False)

tensorboard = TensorBoard(log_dir=PATH['LOGDIR'], 
                          histogram_freq=1, 
                          write_graph=True, 
                          write_images=False)

history = model.fit_generator(train_generator, 
                              samples_per_epoch=len(train) // (4 * BATCH_SIZE) * BATCH_SIZE , 
                              nb_epoch=NB_EPOCH, 
                              callbacks=[checkpoint, tensorboard], 
                              validation_data=valid_generator, 
                              class_weight=None, 
                              nb_val_samples=len(valid),
                              nb_worker=1)
    
model.save_weights(join(PATH['WEIGHTS'], '3DCAE_nodule_final'))





