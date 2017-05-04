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