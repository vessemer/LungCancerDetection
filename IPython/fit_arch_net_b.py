import sys
sys.path.append('../')
sys.path.append('../Nets/')
from glob import glob
from os.path import join, basename, isfile
from multiprocessing import Pool
from scipy.ndimage.interpolation import rotate
from keras.callbacks import ModelCheckpoint
from tqdm import tqdm
import pickle
from functools import partial
from Nodule import *
from numpy import *
from NoduleClf import*
from SegNod import*
from paths import *


class LoggingCallback(Callback):
    """Callback that logs message at end of epoch.
    """

    def __init__(self, print_fcn=print):
        Callback.__init__(self)
        self.print_fcn = print_fcn

    def on_epoch_end(self, epoch, logs={}):
        msg = "Epoch: %i, %s" % (epoch, ", ".join("%s: %f" % (k, v) for k, v in logs.iteritems()))
        self.print_fcn(msg)
       
    
def cae_augment_patch(patch_mute, shape, shift=2):
    patch, mute = patch_mute
    if mute > 0:
        shift = random.randint(-shift, shift, 3)
        angle = mute % 4 * 90
        if angle != 0:
            patch = rotate(patch, angle, axes=[1,2])
            
    
    center = array(patch.shape) // 2 
    if mute > 0:
        center += shift
        
    left = array(shape) // 2
    right = array(shape) - left
    patch = patch[center[0] - left[0]:center[0] + right[0], 
                  center[1] - left[1]:center[1] + right[1], 
                  center[2] - left[2]:center[2] + right[2]]
    
    return patch


def cae_batch_generator(patch_paths, 
                        batch_size, shape=(6, 20, 20), 
                        angle=15, shift=4, CPU=4):
    
    number_of_batches = ceil(len(patch_paths) / batch_size)
    counter = 0
    
    random.shuffle(patch_paths)
    
    while True:
        batch_files = patch_paths[batch_size * counter:batch_size * (counter + 1)]
        
        shifts = [patch_path.split('_shift')[-1] for patch_path in batch_files] 
        shifts = [int(shift) if shift in ' '.join(arange(NODULE_OVERSAMPLING).astype(str)) else -1
                  for shift in shifts]
        
        batch_files = [patch_path.split('_shift')[0] for patch_path in batch_files] 
        patch_list = [load(patch_path) for patch_path in batch_files]
        
        labels = [[1, 0] if 'NODULE' in patch_path else [0, 1] 
                  for patch_path in batch_files]
        
        augment = partial(cae_augment_patch, shape=shape, shift=shift)
        with Pool(CPU) as pool:
            patch_list = pool.map(augment, list(zip(patch_list, shifts)))
            
        patch_list = array(patch_list)
        patch_list = clip(patch_list, -1000, 400)
        patch_list = (patch_list - patch_list.min()) / (patch_list.max() - patch_list.min())
        
        counter += 1
        
        yield expand_dims(patch_list, 1), asarray(labels)
        
        if counter == number_of_batches:
            random.shuffle(patch_paths)
            counter = 0
            
            
            
            
valid_fold = '0'
file_list = [path + '_shift' + str(i) 
             for path in glob(join(PATH['LUNA_NODULES_WOR'], 'subset*', '*.npy')) 
             for i in range(NODULE_OVERSAMPLING)]
file_list = set(file_list
             + [path for path in glob(join(PATH['LUNA_VESSELS_WOR'], 'subset*', '*.npy'))])

valid = [path if '_shift' not in path else path.split('_shift')[0] 
         for path in file_list if 'subset' + valid_fold in path]
train = list(file_list.difference(valid))



if sys.argv[1].lower() == 'c':
    SHAPE = (26, 40, 40)
    archmodel = arch_C()

if sys.argv[1].lower() == 'b':
    SHAPE = (10, 30, 30)
    archmodel = arch_B()
    
if sys.argv[1].lower() == 'a':
    SHAPE = (6, 20, 20)
    archmodel = arch_A()


CPU = 20
ANGLE = 90
NODULE_OVERSAMPLING = 81
SHIFT = 2
VALID_FOLD = '0'
NB_EPOCH = 50
BATCH_SIZE = 200
POSTFIX = sys.argv[1]

def print_fcn(msg):
    msg = [msg]
    if isfile(join('/home/a.dobrenkii/', 'archlog_' + POSTFIX)):
        msg += pickle.load(join('/home/a.dobrenkii/', 'archlog_' + POSTFIX))
    pickle.dump(join('/home/a.dobrenkii/', 'archlog_' + POSTFIX), msg)

    
sgd = SGD(lr=.3, decay=1e-5, momentum=.9, nesterov=True)
archmodel.compile(loss='binary_crossentropy', optimizer=sgd)      
        

train_generator = cae_batch_generator(train, 
                                      batch_size=BATCH_SIZE, 
                                      shape=SHAPE, 
                                      shift=SHIFT, 
                                      CPU=CPU)

valid_generator = cae_batch_generator(valid, 
                                      batch_size=BATCH_SIZE, 
                                      shape=SHAPE, 
                                      shift=0, 
                                      CPU=CPU)   


checkpoint = ModelCheckpoint(filepath=join(PATH['WEIGHTS'], 
                                           'CUMedVis_arch_' + sys.argv[1]), 
                             verbose=1, 
                             save_best_only=True)

logger = LoggingCallback(print_fcn)

pickle.dump(join('/home/a.dobrenkii/', 'pre_archlog_' + POSTFIX), 'up to this point all was fine!')

archmodel.fit_generator(train_generator, 
                        samples_per_epoch=len(train), 
                        nb_epoch=NB_EPOCH, 
                        callbacks=[checkpoint, logger], 
                        validation_data=valid_generator, 
                        class_weight=None, 
                        nb_val_samples=len(val),
                        nb_worker=1)
    
    
model.save_weights(join(PATH['WEIGHTS'], 
                        'CUMedVis_arch_' + sys.argv[1].lower() + '_final'))
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
   