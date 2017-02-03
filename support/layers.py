import tensorflow as tf


class Dense():
    """Fully-connected layer"""
    def __init__(self, scope="dense_layer", size=None, dropout=1.,
                 nonlinearity=tf.identity):
        # (str, int, (float | tf.Tensor), tf.op)
        assert size, "Must specify layer size (num nodes)"
        self.scope = scope
        self.size = size
        self.dropout = dropout # keep_prob
        self.nonlinearity = nonlinearity

    def __call__(self, x):
        """Dense layer currying, to apply layer to any input tensor `x`"""
        # tf.Tensor -> tf.Tensor
        # Flatten input if not
        
        with tf.name_scope(self.scope):
            pred = tf.equal(tf.shape(x), tf.constant([1]))
            x = tf.select(pred, x, tf.reshape(x, (tf.shape(x)[0], -1)))
            while True:
                try: # reuse weights if already initialized
                    return self.nonlinearity(tf.matmul(x, self.w) + self.b)
                except(AttributeError):
                    self.w, self.b = wbVars(x.get_shape()[1].value, self.size)
                    self.w = tf.nn.dropout(self.w, self.dropout)

                    
class Conv():
    """Convolution layer"""
    def __init__(self, scope="conv_layer", shape=None, dropout=1.,
                 nonlinearity=tf.identity, strides=[1,1,1,1], padding="SAME", 
                 use_cudnn_on_gpu=True, data_format=None, reshape_input=None):
        # (str, int, (float | tf.Tensor), tf.op)
        assert shape, "Must specify layer shape (dim1, dim2, in, out)"
        assert len(shape) != 4, "Must specify layer shape with 4D (dim1, dim2, in, out)"
        if reshape_input:
            assert len(reshape_input), "Must specify input shape with 4D (batch, dim1, dim2, channels)"
        
        self.scope = scope
        self.shape = shape
        self.dropout = dropout # keep_prob
        self.nonlinearity = nonlinearity
        self.stride = stride
        self.strides = strides
        self.padding = padding
        self.use_cudnn_on_gpu = use_cudnn_on_gpu
        self.data_format = data_format
        self.reshape_input = reshape_input
        

    def __call__(self, x):
        """Convolution layer currying, to apply layer to any input tensor `x`"""
        # tf.Tensor -> tf.Tensor
        if self.reshape_input:
            x = tf.reshape(x, self.reshape_input)
        
        
        with tf.name_scope(self.scope):
            while True:
                try: # reuse weights if already initialized
                    return self.nonlinearity(tf.nn.bias_add(
                        tf.nn.conv2d(x, self.w, strides=self.strides, 
                                     padding=self.padding, 
                                     use_cudnn_on_gpu=self.use_cudnn_on_gpu, 
                                     data_format=self.data_format, 
                                     name=self.scope) + self.b),
                        self.b)
                
                except(AttributeError):
                    self.w, self.b = wbVars(self.shape)
                    self.w = tf.nn.dropout(self.w, self.dropout)

                    
#def wbVars(shape, dtype=tf.float32, stddev=5e-2):
#    initial_w = tf.random_normal_initializer(stddev=stddev)
#    initial_b = tf.constant_initializer(0.0)
#    return (tf.get_variable(name="weights", shape=shape, initializer=initial_w, dtype=dtype),
#            tf.get_variable(name="biases", shape=[shape[-1]], initializer=initial_b, dtype=dtype))
                    
def wbVars(fan_in, fan_out):
        """Helper to initialize weights and biases, via He's adaptation
        of Xavier init for ReLUs: https://arxiv.org/abs/1502.01852
        """
        # (int, int) -> (tf.Variable, tf.Variable)
        stddev = tf.cast((2 / fan_in)**0.5, tf.float32)

        initial_w = tf.random_normal([fan_in, fan_out], stddev=stddev)
        initial_b = tf.zeros([fan_out])

        return (tf.Variable(initial_w, trainable=True, name="weights"),
                tf.Variable(initial_b, trainable=True, name="biases"))
    
