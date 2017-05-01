from NoduleClf import *
from keras.layers.advanced_activations import LeakyReLU


def coder(patch_input=None,
          shape=(18, 48, 48, 1), 
          trainable=[True, True, True]):
   
    
    # Determine proper input shape
    if K.image_dim_ordering() != 'tf':
        print('Wrong dim ordering: should be TF')
        
    if patch_input is None:
        patch_input = Input(shape=shape)
        
    x = Dropout(.3)(patch_input)
    bn_axis = 4
    
#   shape:  1, 16, 48, 48
    x = Convolution3D(16, 5, 7, 7, 
                      subsample=(1, 1, 1),  
                      border_mode='same', 
                      trainable=trainable[0])(x) 

    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    x = MaxPooling3D((1, 2, 2))(x)

    #   shape:  32, 16, 24, 24
    x = conv_block(x, 3, [16, 16, 32], trainable=trainable[1]) 
    x = identity_block(x, 3, [16, 16, 32], trainable=trainable[1]) 
    x = MaxPooling3D((2, 2, 2))(x)

    #   shape:  64, 8, 12, 12
    x = conv_block(x, 3, [32, 32, 64], trainable=trainable[2]) 
    x = identity_block(x, 3, [32, 32, 64], trainable=trainable[2]) 
    x = identity_block(x, 3, [32, 32, 64], trainable=trainable[2]) 
    x = AveragePooling3D((1, 2, 2))(x)
    
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
              trainable=[True, True, True]):
    
    bn_axis = 4
      
        
    # Determine proper input shape
    if K.image_dim_ordering() != 'tf':
        print('Wrong dim ordering: should be TF')
    
    inputs = [Input(shape=shape) 
              for i in range(channels)]

    coders = [coder(tnsr, trainable=trainable) 
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
    
    bottle_neck = Model(inputs, x)
    
    
    if dropout_conv is not None:
        x = Dropout(dropout_conv)(x)
        
    x = Flatten()(x)
    x = Dense(256)(x)
    x = LeakyReLU(.3)(x)
    
    if dropout_dence is not None:
        x = Dropout(dropout_dence)(x)
        
    x = Dense(out, activation='softmax', 
              name='IsNodule')(x)

    coders_model = [Model(inp, coder) 
                    for inp, coder in zip(inputs, coders)]
    clf_model = Model(inputs, x)

    return clf_model, coders_model, bottle_neck