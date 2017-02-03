import tensorflow as tf


class Losses:
    @staticmethod
    def rmse_loss(x, y):
        """RMSE metric"""
        with tf.name_scope("RMSE"):
            return tf.sqrt(tf.reduce_mean(tf.square(tf.sub(x, y))))

    @staticmethod
    def mse(x, y):
        """MSE metric"""
        with tf.name_scope("MSE"):
            return tf.reduce_mean(tf.square(tf.sub(x, y)))

    @staticmethod
    def KL_loss(mu, log_sigma):
        """Kullback-Leibler divergence KL(q||p) for Gaussian distribution"""
        with tf.name_scope("KL_divergence"):
            return -0.5 * tf.reduce_sum(1 + 2 * log_sigma - mu ** 2 -
                                        tf.exp(2 * log_sigma), 1)