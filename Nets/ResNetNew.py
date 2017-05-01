from ResNetAPI import *
from keras.layers import LeakyReLU

def coder(in_tensor,
          stride=(1, 2, 2),
          bn_axis=4,
          dropout=.2,
          kernel=(3, 5, 5),
          trainable=True):
    
    #x = Dropout(dropout)(in_tensor)
    x = Convolution3D(64, kernel[0], kernel[1], kernel[2],
                      border_mode='same',
                      trainable=trainable)(in_tensor)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    x = Dropout(dropout)(x)
    x = MaxPooling3D((2, 2, 2))(x)
                     
    x = identity_block(x, 3, [64, 64], trainable=trainable, dropout=dropout)

    x = conv_block(x, 3, [64, 64], trainable=trainable, dropout=dropout)
    x = MaxPooling3D(stride)(x)
    
    x = conv_block(x, 3, [64, 64], trainable=trainable, dropout=dropout)
    x = identity_block(x, 3, [64, 64], trainable=trainable, dropout=dropout)
    x = AveragePooling3D((3, 3, 3))(x)
    x = Flatten()(x)
    x = Dense(250, activation='relu')(x)
    return x


def predictor(in_shape=[(18, 42, 42, 1), 
                        (36, 20, 42, 1), 
                        (36, 42, 20, 1)],
              
              strides=[(1, 2, 2), 
                       (2, 1, 2), 
                       (2, 2, 1)],
              
              kernels=[(3, 5, 5), 
                       (5, 3, 5), 
                       (5, 5, 3)],
              out_shape=(36, 40, 40, 1),
              dropout_conv=None,
              dropout_dence=None,
              trainable=True,
              bn_axis=4):
   
    # Determine proper input shape
    if K.image_dim_ordering() != 'tf':
        print('Wrong dim ordering: should be TF')
    inputs = [Input(shape=inx) for inx in in_shape]
    coders = [coder(in_tensor=tnsr, stride=stride, kernel=kernel, 
                    trainable=trainable, dropout=dropout_conv) 
              for tnsr, stride, kernel in zip(inputs, strides, kernels)]
    
    x = merge(coders, mode='concat')
    #   shape:  256, 9, 10, 10   
    
    x = Dropout(dropout_dence)(x)
    #x = Dense(256, activation='relu')(x)
    x = Dense(2, 
              activation='softmax', 
              name='is_nodule')(x)

    model = Model(inputs, x)
    
    return model