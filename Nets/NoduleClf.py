<<<<<<< HEAD
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
=======
from Nodule import *
from keras.layers.advanced_activations import LeakyReLU


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


def coder(shape=(24, 48, 48, 1), 
          layers=None, 
          poolings=None,
          input_tensor=None,
          bn_axis = 4):
    
    if input_tensor is None:
        input_tensor = Input(shape=shape)
        
    x = Convolution3D(64, 5, 5, 5, 
                      subsample=(1, 2, 2),  
                      border_mode='same', 
                      trainable=True)(input_tensor)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    x = MaxPooling3D((3, 3, 3), strides=(2, 2, 2))(x)
                     
    x = identity_block(x, 3, [64, 64])
    x = identity_block(x, 3, [64, 64])

    x = conv_block(x, 3, [128, 128], stride=(2, 2, 2))
    x = identity_block(x, 3, [128, 128])
    x = identity_block(x, 3, [128, 128])
    
    
    x = conv_block(x, 3, [256, 256], stride=(2, 2, 2))
    x = identity_block(x, 3, [256, 256])
    x = identity_block(x, 3, [256, 256])
    
    x = conv_block(x, 3, [512, 512], stride=(1, 1, 1))
    x = identity_block(x, 3, [512, 512])
    x = identity_block(x, 3, [512, 512])
    bottle_neck = AveragePooling3D((3, 3, 3))(x)
    return bottle_neck


def predictor(channels=3,
              out=2,
              shape=(20, 48, 48, 1),
              dropout_conv=None,
              dropout_dence=None,
              bn_axis = 4):
       
    if K.image_dim_ordering() != 'tf':
        print('Wrong dim ordering: should be TF')
    
    inputs = [Input(shape=shape) 
              for i in range(channels)]
           
    coders = [coder(input_tensor=tnsr, layers=None) 
              for tnsr in inputs]
    
    x = merge(coders, mode='concat')
    
    bottle_neck = Model(inputs, x)
        
    if dropout_conv is not None:
        x = Dropout(dropout_conv)(x)
    
    x = Flatten()(x)
    
    x = Dense(256)(x)
    x = LeakyReLU(.3)(x)
    
    if dropout_dence is not None:
        x = Dropout(dropout_dence)(x)
        
    x = Dense(out, 
              activation='softmax', 
              name='IsNodule')(x)

    coders_model = [Model(inp, coder) 
                    for inp, coder in zip(inputs, coders)]
    
    clf_model = Model(inputs, x)

    return clf_model, coders_model, bottle_neck
>>>>>>> 1362926d9120cec3f7e15c0c2bfb790e6ac8f408
