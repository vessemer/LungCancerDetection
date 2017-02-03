import tensorflow as tf


class Dense:
    def __init__(self, shape, nonlinearity, name="Dense"):
        self.nonlinearity = nonlinearity
        self.W = tf.Variable(tf.zeros(shape), name="weigh")
        self.b = tf.Variable(tf.zeros([shape[-1]]), name="bias")

    def __call__(self, x):

        if len(x.get_shape().as_list()) > 2:
            x = tf.reshape(x, (tf.shape(x)[0], -1))

        return self.nonlinearity(tf.add(
            tf.matmul(x, self.W),
            self.b))


class Conv:
    def __init__(self, shape, nonlinearity, strides=[1],
                 padding="SAME", name="Conv"):
        self.nonlinearity = nonlinearity
        self.padding = padding
        if len(strides) == 1:
            self.strides = strides * len(shape)

        assert len(self.strides) == len(shape), "Strides in conflict with shape"

        self.W = tf.Variable(tf.zeros(shape), name="weigh")
        self.b = tf.Variable(tf.zeros([shape[-1]]), name="bias")

    def __call__(self, x):
        return self.nonlinearity(tf.add(tf.nn.conv2d(input=x,
                                                     filter=self.W,
                                                     strides=self.strides,
                                                     padding=self.padding),
                                        self.b))
