from __future__ import print_function

import numpy as np
import warnings

from keras.layers import merge, Input
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Convolution3D, MaxPooling3D, ZeroPadding3D, AveragePooling3D, UpSampling3D
from keras.layers import BatchNormalization
from keras.models import Model
import keras.backend as K
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file


def identity_block(input_tensor, kernel_size, 
                   filters, bn_axis=4, trainable=True):

    
    x = BatchNormalization(axis=bn_axis)(input_tensor)
    x = Activation('relu')(x)
    x = Convolution3D(filters[0], 
                      kernel_size, 
                      kernel_size, 
                      kernel_size,
                      border_mode='same', 
                      trainable=trainable)(x)

    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)

    x = Convolution3D(filters[1], 
                      kernel_size, 
                      kernel_size, 
                      kernel_size,

                      border_mode='same', 
                      trainable=trainable)(x)
    x = merge([x, input_tensor], mode='sum')
    return x


def conv_block(input_tensor, kernel_size, 
               filters, bn_axis=4, 
               trainable=True, stride=(1, 1, 1)):
    
    x = BatchNormalization(axis=bn_axis)(input_tensor)
    x = Activation('relu')(x)
    x = Convolution3D(filters[0], 
                      kernel_size, 
                      kernel_size, 
                      kernel_size, 
                      subsample=stride,
                      border_mode='same', 
                      trainable=trainable)(x)
    
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    
    x = Convolution3D(filters[1], 
                      kernel_size, 
                      kernel_size, 
                      kernel_size, 
                      subsample=(1, 1, 1),  
                      border_mode='same', 
                      trainable=trainable)(x)
    
    shortcut = Convolution3D(filters[1], 
                             kernel_size, 
                             kernel_size, 
                             kernel_size, 
                             subsample=stride,
                             border_mode='same', 
                             trainable=trainable)(input_tensor)
    
    x = merge([x, shortcut], mode='sum')
    return x


def dim_concentration(out=(20, 40, 40, 1), 
                      shape=(20, 40, 40, 1)):
   
    # Determine proper input shape
    if K.image_dim_ordering() != 'tf':
        print('Wrong dim ordering: should be TF')
        
    img_input = Input(shape=shape)
    x = Dropout(.3)(img_input)
    bn_axis = 4
    
    
    x = Convolution3D(64, 5, 5, 5, 
                      subsample=(1, 1, 1),  
                      border_mode='same')(x)     

    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    x = MaxPooling3D((1, 2, 2))(x)
    
    #   shape:  32, 20, 20, 20
    x = conv_block(x, 3, [64, 64])
    x = identity_block(x, 3, [64, 64])  
    x = MaxPooling3D((2, 2, 2))(x)

    #   shape:  64, 10, 10, 10
    x = conv_block(x, 3, [128, 128]) 
    x = identity_block(x, 3, [128, 128])  
    x = MaxPooling3D((2, 2, 2))(x)

#   shape:  128, 5, 5, 5
    x = Dropout(.2)(x)
    bottle_neck = conv_block(x, 3, [256, 256])
    x = Dropout(.2)(x)
    bottle_neck = identity_block(bottle_neck, 3, [256, 256])
    x = Dropout(.2)(x)
    bottle_neck = identity_block(bottle_neck, 3, [256, 256])
#    x = Dropout(.3)(x)
    x = UpSampling3D((2, 2, 2))(x)
    

#   shape:  64, 10, 10, 10
    x = conv_block(x, 3, [128, 128])
    x = UpSampling3D((2, 2, 2))(x)

#   shape:  32, 20, 20, 20
    x = conv_block(x, 3, [64, 64])
    x = UpSampling3D((1, 2, 2))(x)

#   shape:  1, 20, 40, 40

    x = Convolution3D(1, 3, 3, 3, subsample=(1, 1, 1),  border_mode='same')(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('sigmoid')(x)

    bottle_neck = Model(img_input, bottle_neck)
    model = Model(img_input, x)
    
    return model, bottle_neck
