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
                   filters, bn_axis=4, 
                   dropout=None, trainable=True):

    x = Convolution3D(filters[0], 
                      kernel_size, 
                      kernel_size, 
                      kernel_size,
                      border_mode='same', 
                      trainable=trainable)(input_tensor)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    if dropout is not None:
        x = Dropout(dropout)(x)
    x = Convolution3D(filters[1], 
                      kernel_size, 
                      kernel_size, 
                      kernel_size,
                      border_mode='same', 
                      trainable=trainable)(x)
    x = merge([x, input_tensor], mode='sum')
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    if dropout is not None:
        x = Dropout(dropout)(x)
    return x


def conv_block(input_tensor, kernel_size, 
               filters, bn_axis=4, 
               trainable=True, dropout=None,
               stride=(1, 1, 1)):
    
    x = Convolution3D(filters[0], 
                      kernel_size, 
                      kernel_size, 
                      kernel_size, 
                      subsample=stride,
                      border_mode='same', 
                      trainable=trainable)(input_tensor)
    
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    if dropout is not None:
        x = Dropout(dropout)(x)
    
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
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    if dropout is not None:
        x = Dropout(dropout)(x)
    return x

def identity_block_api(kernel_size, filters, 
                       bn_axis=4, trainable=True):
    interm_layers = [BatchNormalization(axis=bn_axis),
                     Activation('relu'),

                     Convolution3D(filters[1],
                                   kernel_size,
                                   kernel_size,
                                   kernel_size,
                                   border_mode='same',
                                   trainable=trainable),

                     BatchNormalization(axis=bn_axis),
                     Activation('relu'),

                     Convolution3D(filters[1],
                                   kernel_size,
                                   kernel_size,
                                   kernel_size,
                                   border_mode='same', 
                                   trainable=trainable)]
    
    return interm_layers


def conv_block_api(kernel_size, filters, 
                   bn_axis=4, trainable=True,
                   stride=(1, 1, 1)):
    
    interm_layers = [BatchNormalization(axis=bn_axis),
                     Activation('relu'),

                     Convolution3D(filters[1],
                                   kernel_size,
                                   kernel_size,
                                   kernel_size,
                                   subsample=stride,
                                   border_mode='same', 
                                   trainable=trainable),

                     BatchNormalization(axis=bn_axis),
                     Activation('relu'),

                     Convolution3D(filters[1],
                                   kernel_size,
                                   kernel_size,
                                   kernel_size,
                                   subsample=(1, 1, 1),
                                   border_mode='same', 
                                   trainable=trainable)]

    shortcut = Convolution3D(filters[1], 
                             kernel_size, 
                             kernel_size, 
                             kernel_size, 
                             subsample=stride,
                             border_mode='same', 
                             trainable=trainable)

    return interm_layers, shortcut


def apply_block(input_tensor, layer):
    if layer[0] == 'usual':
        return layer[1](input_tensor) 
    
    if layer[0] == 'conv_blok':
        x = input_tensor
        for lay in layer[1][0]:
            x = lay(x)
        shortcut = layer[1][1](input_tensor)
        return merge([x, shortcut], mode='sum')
    
    if layer[0] == 'ident_blok':
        x = input_tensor
        for lay in layer[1]:
            x = lay(x)
        return merge([x, input_tensor], mode='sum')