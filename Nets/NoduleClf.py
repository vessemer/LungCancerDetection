from __future__ import print_function

import numpy as np
import warnings

from keras.layers import merge, Input
from keras.layers import Dense, Activation, Flatten
from keras.layers import Convolution3D, MaxPooling3D, ZeroPadding3D, AveragePooling3D, UpSampling3D
from keras.layers import BatchNormalization
from keras.models import Model
import keras.backend as K
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file



def identity_block(input_tensor, kernel_size, filters, bn_axis=1):

    x = Convolution3D(filters[0], 1, 1, 1)(input_tensor)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)

    x = Convolution3D(filters[1], 
                      kernel_size, 
                      kernel_size, 
                      kernel_size,
                      border_mode='same')(x)
    
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)

    x = Convolution3D(filters[2], 1, 1, 1)(x)
    x = BatchNormalization(axis=bn_axis)(x)

    x = merge([x, input_tensor], mode='sum')
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, bn_axis=1):

    x = Convolution3D(filters[0], 1, 1, 1, subsample=(1, 1, 1),  border_mode='same')(input_tensor)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    
    x = Convolution3D(filters[1], 
                      kernel_size, 
                      kernel_size, 
                      kernel_size, 
                      subsample=(1, 1, 1),  
                      border_mode='same')(x)
    
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    
    x = Convolution3D(filters[2], 1, 1, 1, 
                      subsample=(1, 1, 1),  
                      border_mode='same')(x)
    
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    
    shortcut = Convolution3D(filters[2], 1, 1, 1, subsample=(1, 1, 1))(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis)(shortcut)

    x = merge([x, shortcut], mode='sum')
    x = Activation('relu')(x)
    return x


def coder(out=(1, 8, 64, 64), 
          shape=(1, 8, 64, 64)):
   
    # Determine proper input shape
    if K.image_dim_ordering() != 'th':
        print('Wrong dim ordering: should be TH')
        
    img_input = Input(shape=shape)
    bn_axis = 1
    
#   shape:  1, 8, 64, 64
    x = Convolution3D(16, 3, 7, 7, subsample=(1, 1, 1),  border_mode='same')(img_input)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    x = MaxPooling3D((1, 2, 2))(x)
    
#   shape:  32, 8, 32, 32
    x = conv_block(x, 3, [16, 16, 32])
    x = identity_block(x, 3, [16, 16, 32])
    x = identity_block(x, 3, [16, 16, 32])
    x = MaxPooling3D((1, 2, 2))(x)

#   shape:  64, 8, 16, 16
    x = conv_block(x, 3, [32, 32, 64])
    x = identity_block(x, 3, [32, 32, 64])
    x = identity_block(x, 3, [32, 32, 64])
    x = identity_block(x, 3, [32, 32, 64])
    x = AveragePooling3D((1, 2, 2))(x)

#   shape:  128, 8, 8, 8
    bottle_neck = conv_block(x, 3, [16, 16, 32])
    bottle_neck = identity_block(bottle_neck, 3, [16, 16, 32])
    bottle_neck = identity_block(bottle_neck, 3, [16, 16, 32])
    bottle_neck = identity_block(bottle_neck, 3, [16, 16, 32])
    
    return img_input, bottle_neck


def predictor(channels=3,
              out=1, 
              shape=(32, 8, 8, 8), 
              input_tensor=None,
              coder_weights=None):
   
    # Determine proper input shape
    if K.image_dim_ordering() != 'th':
        print('Wrong dim ordering: should be TH')
        
    coders = [coder() for i in range(3)]
    x = merge([coder[1] for coder in coders], mode='concat')
    
#   shape:  3 ч* 128, 8, 8, 8
    x = conv_block(x, 3, [32, 32, 64])
    x = identity_block(x, 3, [32, 32, 64])
    x = identity_block(x, 3, [32, 32, 64])
    x = identity_block(x, 3, [32, 32, 64])
    
    x = AveragePooling3D((2, 2, 2))(x)

    x = Flatten()(x)
    x = Dense(out, activation='sigmoid', name='IsNodule')(x)
    
    coders_model = [Model(coder[0], coder[1]) for coder in coders]
    clf_model = Model([coder[0] for coder in coders], x)
    
    return clf_model, coders_model