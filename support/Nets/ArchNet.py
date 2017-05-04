from __future__ import print_function

from keras.layers import merge, Input
from keras.layers.merge import Concatenate
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Convolution3D, MaxPooling3D, AveragePooling3D
from keras.layers import BatchNormalization
from keras.models import Model
import keras.backend as K



def arch_a(out=2,
           shape=(6, 20, 20, 1)):
    # Determine proper input shape
    if K.image_dim_ordering() != 'th':
        print('Wrong dim ordering: should be TH')

    patch_input = Input(shape=shape)
    x = Dropout(.2)(patch_input)
    bn_axis = 1

    #   shape:  1, 6, 20, 20
    x = Convolution3D(64, 3, 5, 5,
                      border_mode='same')(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)

    #   shape:  64, 6, 20, 20
    x = Convolution3D(64, 3, 5, 5,
                      border_mode='same')(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    
    #   shape:  64, 6, 20, 20
    x = Convolution3D(64, 1, 5, 5,
                      border_mode='same')(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    
    x = Flatten()(x)
    
    x = Dense(150, activation='relu')(x)
    x = Dense(2, activation='softmax')(x)

    return Model(patch_input, x)



def arch_b(out=2,
           shape=(10, 30, 30, 1)):
    # Determine proper input shape
    if K.image_dim_ordering() != 'th':
        print('Wrong dim ordering: should be TH')

    patch_input = Input(shape=shape)
    x = Dropout(.2)(patch_input)
    bn_axis = 1

    #   shape:  1, 10, 30, 30
    x = Convolution3D(64, 3, 5, 5,
                      border_mode='same')(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    x = MaxPooling3D((1, 2, 2))(x)

    #   shape:  64, 5, 15, 15
    x = Convolution3D(64, 3, 5, 5, 
                      border_mode='same')(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    
    #   shape:  64, 5, 15, 15
    x = Convolution3D(64, 3, 5, 5,
                      border_mode='same')(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    
    x = Flatten()(x)
    
    x = Dense(250, activation='relu')(x)
    x = Dense(2, activation='softmax')(x)

    return Model(patch_input, x)



def arch_c(out=2,
           shape=(1, 26, 40, 40)):
    # Determine proper input shape
    if K.image_dim_ordering() != 'th':
        print('Wrong dim ordering: should be TH')

    patch_input = Input(shape=shape)
    x = Dropout(.2)(patch_input)
    bn_axis = 1

    #   shape:  1, 26, 40, 40
    x = Convolution3D(64, 3, 5, 5,
                      border_mode='same')(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    x = MaxPooling3D((2, 2, 2))(x)

    #   shape:  64, 13, 20, 20
    x = Convolution3D(64, 3, 5, 5,
                      border_mode='same')(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    
    #   shape:  64, 13, 20, 20
    x = Convolution3D(64, 3, 5, 5,
                      border_mode='same')(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    
    x = Flatten()(x)
    
    x = Dense(250, activation='relu')(x)
    x = Dense(2, activation='softmax')(x)

    return Model(patch_input, x)




def predictor(channels=3,
              out=2,
              dropout_conv=None,
              dropout_dence=None,
              input_tensor=None,
              coder_weights=None):
    
    # Determine proper input shape
    if K.image_dim_ordering() != 'tf':
        print('Wrong dim ordering: should be TF')
    
    inputs = [Input(shape=shape) for i in range(channels)]
    coders = [arch_a(inputs[0]), 
              arch_a(inputs[1]), 
              arch_a(inputs[2])]
    
    x = merge(coders, mode='concat', concat_axis=-1)
    
    if dropout_dence is not None:
        x = Dropout(dropout_conv)(x)

    x = Dense(256)(x)
    x = LeakyReLU(.3)(x)
    
        
    x = Dense(out, 
              activation='softmax', 
              name='IsNodule')(x)

    coders_model = [Model(inp, coder) 
                    for inp, coder in zip(inputs, coders)]
    clf_model = Model(inputs, x)

    return clf_model, coders_model, bottle_neck




