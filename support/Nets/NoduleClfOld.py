from keras.layers import merge, Input
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Convolution3D, MaxPooling3D, AveragePooling3D
from keras.layers import BatchNormalization
from keras.models import Model
import keras.backend as K


def identity_block(input_tensor, kernel_size, filters, bn_axis=4, trainable=True):
    x = Convolution3D(filters[0], 1, 1, 1, 
                      trainable=trainable)(input_tensor)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)

    x = Convolution3D(filters[1],
                      kernel_size,
                      kernel_size,
                      kernel_size,
                      border_mode='same',
                      trainable=trainable)(x)

    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)

    x = Convolution3D(filters[2], 1, 1, 1, 
                      trainable=trainable)(x)
    x = BatchNormalization(axis=bn_axis)(x)

    x = merge([x, input_tensor], mode='sum')
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, bn_axis=4, trainable=True):
    x = Convolution3D(filters[0], 1, 1, 1,
                      subsample=(1, 1, 1),
                      border_mode='same',
                      trainable=trainable)(input_tensor)

    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)

    x = Convolution3D(filters[1],
                      kernel_size,
                      kernel_size,
                      kernel_size,
                      subsample=(1, 1, 1),
                      border_mode='same',
                      trainable=trainable)(x)

    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)

    x = Convolution3D(filters[2], 1, 1, 1,
                      subsample=(1, 1, 1),
                      border_mode='same',
                      trainable=trainable)(x)

    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)

    shortcut = Convolution3D(filters[2], 1, 1, 1,
                             subsample=(1, 1, 1),
                             trainable=trainable)(input_tensor)

    shortcut = BatchNormalization(axis=bn_axis)(shortcut)

    x = merge([x, shortcut], mode='sum')
    x = Activation('relu')(x)
    return x


def identity_block_api(kernel_size, filters, bn_axis=4):
    interm_layers = [Convolution3D(filters[0], 1, 1, 1),
                     BatchNormalization(axis=bn_axis),
                     Activation('relu'),

                     Convolution3D(filters[1],
                                   kernel_size,
                                   kernel_size,
                                   kernel_size,
                                   border_mode='same'),

                     BatchNormalization(axis=bn_axis),
                     Activation('relu'),

                     Convolution3D(filters[2], 1, 1, 1),
                     BatchNormalization(axis=bn_axis)]
    
    return interm_layers


def conv_block_api(kernel_size, filters, bn_axis=4):
    interm_layers = [Convolution3D(filters[0], 1, 1, 1,
                                   subsample=(1, 1, 1),
                                   border_mode='same'),

                     BatchNormalization(axis=bn_axis),
                     Activation('relu'),

                     Convolution3D(filters[1],
                                   kernel_size,
                                   kernel_size,
                                   kernel_size,
                                   subsample=(1, 1, 1),
                                   border_mode='same'),

                     BatchNormalization(axis=bn_axis),
                     Activation('relu'),

                     Convolution3D(filters[2], 1, 1, 1,
                                   subsample=(1, 1, 1),
                                   border_mode='same'),

                     BatchNormalization(axis=bn_axis),
                     Activation('relu')]

    shortcut = [Convolution3D(filters[2], 
                              1, 1, 1,
                              subsample=(1, 1, 1)),
                BatchNormalization(axis=bn_axis)]

    return interm_layers, shortcut


def apply_block(input_tensor, layers):
    if len(layers) == 2:
        x = layers[0][0](input_tensor)
        for layer in layers[0][1:]:
            x = layer(x)

        shortcut = layers[1][0](input_tensor)
        for layer in layers[1][1:]:
            shortcut = layer(shortcut)
        x = merge([x, shortcut], mode='sum')
    else:
        x = layers[0](input_tensor)
        for layer in layers[1:]:
            x = layer(x)
        x = merge([x, input_tensor], mode='sum')

    return Activation('relu')(x)
 

def coder(patch_input=None,
          shape=(16, 48, 48, 1), 
          layers=None, 
          poolings=None):
   
    
    # Determine proper input shape
    if K.image_dim_ordering() != 'tf':
        print('Wrong dim ordering: should be TF')
        
    if patch_input is None:
        patch_input = Input(shape=shape)
        
    x = Dropout(.3)(patch_input)
    bn_axis = 4
    
#   shape:  1, 16, 48, 48
    if (layers is None) or (poolings is None):
        x = Convolution3D(16, 5, 7, 7, 
                          subsample=(1, 1, 1),  
                          border_mode='same', 
                          trainable=False)(x) #trainable=False
        
        x = BatchNormalization(axis=bn_axis)(x)
        x = Activation('relu')(x)
        x = MaxPooling3D((1, 2, 2))(x)

    #   shape:  32, 16, 24, 24
        x = conv_block(x, 3, [16, 16, 32], trainable=False) #trainable=False
        x = identity_block(x, 3, [16, 16, 32], trainable=False) #trainable=False
        x = MaxPooling3D((2, 2, 2))(x)

    #   shape:  64, 8, 12, 12
        x = conv_block(x, 3, [32, 32, 64], trainable=False) #trainable=False
        x = identity_block(x, 3, [32, 32, 64], trainable=False) #trainable=False
        x = identity_block(x, 3, [32, 32, 64], trainable=False) #trainable=False
        x = AveragePooling3D((1, 2, 2))(x)
        
    else:
        for layer1 in layers[0]:
            x = layer1(x)
        x = poolings[0](x)
        #x = Dropout(.2)(x)

        for i in range(1, len(layers)):
            for layer2 in layers[i]:
                x = apply_block(x, layer2)
            x = poolings[i](x)
            #x = Dropout(.2)(x)
            
        if len(poolings) == 2:
            #   shape:  64, 8, 12, 12
            x = conv_block(x, 3, [32, 32, 64])
            x = identity_block(x, 3, [32, 32, 64])
            x = identity_block(x, 3, [32, 32, 64])
            x = MaxPooling3D((1, 2, 2))(x)
            #x = Dropout(.3)(x)
            
#   shape:  128, 8, 6, 6
    bottle_neck = conv_block(x, 3, [64, 64, 128])
    bottle_neck = identity_block(bottle_neck, 3, [64, 64, 128])
    bottle_neck = identity_block(bottle_neck, 3, [64, 64, 128])

    return bottle_neck


def predictor(channels=3,
              out=2,
              shape=(16, 48, 48, 1),
              dropout_conv=None,
              dropout_dence=None,
              input_tensor=None,
              coder_weights=None,
              shared_layers=False,
              half_sized_shared=False):
    
    bn_axis = 4
    layers1 = [
        [
            Convolution3D(16, 5, 7, 7, 
                          subsample=(1, 1, 1),  
                          border_mode='same'),
            
            BatchNormalization(axis=bn_axis),
            Activation('relu'),
        ], 
        [
            conv_block_api(3, [16, 16, 32]),
            identity_block_api(3, [16, 16, 32])
        ],
        [
            conv_block_api(3, [32, 32, 64]),
            identity_block_api(3, [32, 32, 64]),
            identity_block_api(3, [32, 32, 64])
        ]
    ]
        
        
    poolings1 = [MaxPooling3D((1, 2, 2)), 
                 MaxPooling3D((2, 2, 2)), 
                 MaxPooling3D((1, 2, 2))]
    
    
    layers2 = [
        [
            Convolution3D(16, 5, 7, 7, 
                          subsample=(1, 1, 1),  
                          border_mode='same'),
            
            BatchNormalization(axis=bn_axis),
            Activation('relu'),
        ], 
        [
            conv_block_api(3, [16, 16, 32]),
            identity_block_api(3, [16, 16, 32])
       
        ]
    ]
        
    poolings2 = [MaxPooling3D((1, 2, 2)), 
                 MaxPooling3D((2, 2, 2))]
    
        
    # Determine proper input shape
    if K.image_dim_ordering() != 'tf':
        print('Wrong dim ordering: should be TF')
    
    inputs = [Input(shape=shape) 
              for i in range(channels)]
           
    layers = layers1
    poolings = poolings1
            
    if half_sized_shared:
        layers = layers2
        poolings = poolings2
            
    if shared_layers:
        coders = [coder(tnsr, 
                        layers=layers, 
                        poolings=poolings) 
                  for tnsr in inputs]
    else:
        coders = [coder(tnsr) 
                  for tnsr in inputs]
    
    x = merge(coders, mode='sum')
    
    if dropout_conv is not None:
        x = Dropout(dropout_conv)(x)
        
    #   shape:  128, 8, 6, 6
    x = conv_block(x, 3, [128, 128, 256])
    x = identity_block(x, 3, [128, 128, 256])
    x = AveragePooling3D((2, 2, 2))(x)
    
    #   shape:  256, 4, 3, 3
    x = conv_block(x, 3, [128, 128, 256])
    
    botneck = x
    
    if dropout_conv is not None:
        x = Dropout(dropout_conv)(x)
        
    x = Flatten()(x)
    x = Dense(256)(x)
    x = LeakyReLU(.3)(x)
    intermediate = x
    if dropout_dence is not None:
        x = Dropout(dropout_dence)(x)
        
    x = Dense(out, 
              activation='softmax', 
              name='IsNodule')(x)

    coders_model = [Model(inp, coder) 
                    for inp, coder in zip(inputs, coders)]
    
    clf_model = Model(inputs, x)
    bottle_neck = Model(inputs, [botneck, intermediate])
    
    return clf_model, coders_model, bottle_neck


