from Nodule import *
from keras.layers.advanced_activations import LeakyReLU


def coder(input_tensor=None,
          shape=(18, 48, 48, 1)):
   
    
    # Determine proper input shape
    if K.image_dim_ordering() != 'tf':
        print('Wrong dim ordering: should be TF')
        
    if input_tensor is None:
        input_tensor = Input(shape=shape)
        
    x = Dropout(.3)(input_tensor)
    bn_axis = 4
    
#   shape:  1, 16, 48, 48
    x = Convolution3D(16, 5, 7, 7, 
                      subsample=(1, 1, 1),  
                      border_mode='same')(x) 
    
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    x = MaxPooling3D((1, 2, 2))(x)

#   shape:  32, 16, 24, 24
    x = conv_block(x, 3, [16, 16, 32]) 
    x = identity_block(x, 3, [16, 16, 32]) 
    x = MaxPooling3D((2, 2, 2))(x)

#   shape:  64, 8, 12, 12
    x = conv_block(x, 3, [32, 32, 64]) 
    x = identity_block(x, 3, [32, 32, 64]) 
    x = identity_block(x, 3, [32, 32, 64]) 
    x = MaxPooling3D((1, 2, 2))(x)
        
            
#   shape:  128, 8, 6, 6
    bottle_neck = conv_block(x, 3, [64, 64, 128])
    bottle_neck = identity_block(bottle_neck, 3, [64, 64, 128])
    bottle_neck = identity_block(bottle_neck, 3, [64, 64, 128])
    x = UpSampling3D((1, 2, 2))(x)
    

#   shape:  64, 8, 12, 12
    x = conv_block(x, 3, [32, 32, 64])
    x = identity_block(x, 3, [32, 32, 64])
    x = UpSampling3D((2, 2, 2))(x)

#   shape:  32, 16, 24, 24
    x = conv_block(x, 3, [16, 16, 32])
    x = identity_block(x, 3, [16, 16, 32])
    x = UpSampling3D((1, 2, 2))(x)

#   shape:  1, 16, 48, 48
    x = Convolution3D(1, 3, 3, 3, subsample=(1, 1, 1),  border_mode='same')(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('sigmoid')(x)

    bottle_neck = Model(input_tensor, bottle_neck)

    return bottle_neck, x


def tripple(patch_input=None,
            channels=3,
            shape=(18, 48, 48, 1)):
    
    inputs = [Input(shape=shape) 
              for i in range(channels)]
           
    # consist of (bottle_neck, x)
    coders = [coder(input_tensor=tnsr) 
              for tnsr in inputs]
    
    final_model = Model(inputs, [coder[1] for coder in coders])
    return final_model, [coder[0] for coder in coders]




