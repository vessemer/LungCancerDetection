from ResNetAPI import *


def coder(in_tensor,
          stride=(1, 2, 2),
          bn_axis = 4):
    
    x = Convolution3D(32, 5, 5, 5,
                      border_mode='same')(in_tensor)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    x = MaxPooling3D((2, 2, 2))(x)
                     
    x = identity_block(x, 3, [32, 32])
    x = identity_block(x, 3, [32, 32])

    x = conv_block(x, 3, [64, 64], stride=stride)
    x = identity_block(x, 3, [64, 64])
    x = identity_block(x, 3, [64, 64])
    
    
    x = conv_block(x, 3, [128, 128])
    x = identity_block(x, 3, [128, 128])
    bottle_neck = identity_block(x, 3, [128, 128])
    return bottle_neck



def dim_concentration(in_shape=[(18, 42, 42, 1), 
                                (36, 20, 42, 1), 
                                (36, 42, 20, 1)],
                      strides=[(1, 2, 2), 
                               (2, 1, 2), 
                               (2, 2, 1)],
                      out_shape=(36, 40, 40, 1),
                      bn_axis=4):
   
    # Determine proper input shape
    if K.image_dim_ordering() != 'tf':
        print('Wrong dim ordering: should be TF')
    inputs = [Input(shape=inx) for inx in in_shape]
    coders = [coder(in_tensor=tnsr, stride=stride) 
              for tnsr, stride in zip(inputs, strides)]
    
    x = merge(coders, mode='ave')
    bottle_neck = Model(inputs, x)
    
    #   shape:  256, 9, 10, 10   
    
    x = conv_block(x, 3, [128, 128]) 
    x = identity_block(x, 3, [128, 128])  
    bottle_neck = Model(inputs, x)
    
    x = UpSampling3D((2, 2, 2))(x)

    #   shape:  256, 18, 20, 20
    x = conv_block(x, 3, [32, 32]) 
    x = identity_block(x, 3, [32, 32])  
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    x = Convolution3D(1, 3, 3, 3, 
                      border_mode='same')(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('sigmoid')(x)

    model = Model(inputs, x)
    
    return model, bottle_neck