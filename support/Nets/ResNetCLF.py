from ResNetAPI import *
from keras.layers import LeakyReLU

def coder(in_tensor,
          stride=(1, 2, 2),
          bn_axis=4,
          dropout=.2,
          trainable=True):
    
    x = Dropout(dropout)(in_tensor)
    x = Convolution3D(32, 5, 5, 5,
                      border_mode='same',
                      trainable=trainable)(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    x = Dropout(dropout)(x)
    x = MaxPooling3D((2, 2, 2))(x)
                     
    x = identity_block(x, 3, [32, 32], trainable=trainable, dropout=dropout)

    x = conv_block(x, 3, [64, 64], stride=stride, trainable=trainable, dropout=dropout)
    x = identity_block(x, 3, [64, 64], trainable=trainable, dropout=dropout)
    
    x = conv_block(x, 3, [128, 128], dropout=dropout)
    x = identity_block(x, 3, [128, 128], dropout=dropout)
    x = identity_block(x, 3, [128, 128], dropout=dropout)
    return x



def predictor(in_shape=[(18, 42, 42, 1), 
                        (36, 20, 42, 1), 
                        (36, 42, 20, 1)],
              strides=[(1, 2, 2), 
                       (2, 1, 2), 
                       (2, 2, 1)],
              out_shape=(36, 40, 40, 1),
              dropout_conv=None,
              dropout_dence=None,
              trainable=True,
              bn_axis=4):
   
    # Determine proper input shape
    if K.image_dim_ordering() != 'tf':
        print('Wrong dim ordering: should be TF')
    inputs = [Input(shape=inx) for inx in in_shape]
    coders = [coder(in_tensor=tnsr, stride=stride, 
                    trainable=trainable, dropout=dropout_conv) 
              for tnsr, stride in zip(inputs, strides)]
    
    x = merge(coders, mode='ave')
    x = Dropout(dropout_conv)(x)
    
    #   shape:  256, 9, 10, 10   
    x = Convolution3D(256, 3, 3, 3,
                      border_mode='same')(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_conv)(x)
    x = MaxPooling3D((2, 2, 2))(x)
    
    #   shape:  256, 4, 5, 5
    x = Convolution3D(512, 3, 3, 3,
                      border_mode='same')(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_conv)(x)
    
    x = Flatten()(x)
    
    x = Dense(512)(x)
    x = LeakyReLU(.3)(x)
    x = Dropout(dropout_dence)(x)
        
    x = Dense(2, 
              activation='softmax', 
              name='is_nodule')(x)

    model = Model(inputs, x)
    
    return model, bottle_neck