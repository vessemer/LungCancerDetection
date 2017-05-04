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

    x_merged = merge([x, input_tensor], mode='sum')
    x = Activation('relu')(x_merged)
    return x, x_merged


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

    x_merged = merge([x, shortcut], mode='sum')
    x = Activation('relu')(x_merged)
    return x, x_merged


def seg_nod(out=(1, 8, 64, 64), 
            shape=(1, 8, 64, 64)):
   
    # Determine proper input shape
    if K.image_dim_ordering() != 'th':
        print('Wrong dim ordering: should be TH')
        
    img_input = Input(shape=shape)
    bn_axis = 1
    
#   shape:  1, 8, 64, 64
    x = Convolution3D(16, 3, 7, 7, subsample=(1, 1, 1),  border_mode='same')(img_input)
    #   shape:  16, 8, 64, 64
    x_merged1 = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x_merged1)
    x = MaxPooling3D((1, 2, 2))(x)
    
#   shape:  16, 8, 32, 32
    x, _ = conv_block(x, 3, [16, 16, 32])
    x, _ = identity_block(x, 3, [16, 16, 32])
    #   shape:  32, 8, 32, 32
    x, x_merged2 = identity_block(x, 3, [16, 16, 32])
    x = MaxPooling3D((1, 2, 2))(x)

#   shape:  32, 8, 16, 16
    x, _ = conv_block(x, 3, [32, 32, 64])
    x, _ = identity_block(x, 3, [32, 32, 64])
    x, _ = identity_block(x, 3, [32, 32, 64])
    #   shape:  64, 8, 16, 16
    x, x_merged3 = identity_block(x, 3, [32, 32, 64])
    x = AveragePooling3D((1, 2, 2))(x)

#   shape:  64, 8, 8, 8
    bottle_neck, _ = conv_block(x, 3, [64, 64, 128])
    bottle_neck, _ = identity_block(bottle_neck, 3, [64, 64, 128])
    bottle_neck, _ = identity_block(bottle_neck, 3, [64, 64, 128])
    bottle_neck, _ = identity_block(bottle_neck, 3, [64, 64, 128])
    #   shape:  128, 8, 8, 8
    x = UpSampling3D((1, 2, 2))(x)

#   shape:  32, 8, 16, 16
    x, _ = conv_block(x, 3, [32, 32, 64])
    x, _ = identity_block(x, 3, [32, 32, 64])
    x, _ = identity_block(x, 3, [32, 32, 64])
    _, x = identity_block(x, 3, [32, 32, 64])  
    #   shape:  64, 8, 16, 16
    x = merge([x_merged3, x], mode='sum')
    x = UpSampling3D((1, 2, 2))(x)

#   shape:  16, 8, 32, 32
    x, _ = conv_block(x, 3, [16, 16, 32])
    x, _ = identity_block(x, 3, [16, 16, 32])
    _, x = identity_block(x, 3, [16, 16, 32])
    #   shape:  32, 8, 32, 32
    x = merge([x_merged2, x], mode='sum')
    x = UpSampling3D((1, 2, 2))(x)

#   shape:  32, 8, 64, 64
    x = Convolution3D(16, 3, 3, 3, subsample=(1, 1, 1),  border_mode='same')(x)
    x = BatchNormalization(axis=bn_axis)(x)
    #   shape:  16, 8, 64, 64
    x = merge([x_merged1, x], mode='sum')
    x = Activation('relu')(x)
    
    x = Convolution3D(1, 3, 3, 3, subsample=(1, 1, 1),  border_mode='same')(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('sigmoid')(x)
    
    bottle_neck = Model(img_input, bottle_neck)
    model = Model(img_input, x)
    
    return model, bottle_neck