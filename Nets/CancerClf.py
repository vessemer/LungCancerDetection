from NoduleClfOld import *


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
    for layer1 in layers[0]:
        x = layer1(x)
    x = poolings[0](x)
    #x = Dropout(.2)(x)

    for i in range(1, len(layers)):
        for layer2 in layers[i]:
            x = apply_block(x, layer2)
        x = poolings[i](x)
        #x = Dropout(.2)(x)
            
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
              shared_layers=False):
    
    bn_axis = 4
    layers = [
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
        
        
    poolings = [MaxPooling3D((1, 2, 2)), 
                 MaxPooling3D((2, 2, 2)), 
                 MaxPooling3D((1, 2, 2))]
    
        
    # Determine proper input shape
    if K.image_dim_ordering() != 'tf':
        print('Wrong dim ordering: should be TF')
    
    inputs = [Input(shape=shape) 
              for i in range(channels)]
           
    coders = [coder(tnsr, 
                    layers=layers, 
                    poolings=poolings) 
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
    
    bottle_neck = x
    
    
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
    bottle_neck = Model(inputs, [bottle_neck, intermediate])

    return bottle_neck