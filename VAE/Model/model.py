import numpy as np

from input import DataHandler
import tensorflow as tf
import matplotlib.pyplot as plt
from layers import Dense, Conv
from losses import Losses


class VAE:
    def __init__(self):
        self.sess = tf.InteractiveSession()
        self.dh = DataHandler()
        self._build_graph()
        self.train()
        self.validation()
        self.check()
        self.dh.close_all_queues()

    def _build_graph(self):
        self.x_input = tf.placeholder(tf.float32, shape=[None, 3000])
        self.encoded = Dense([3000, 600], tf.nn.elu)(self.x_input)
        self.encoded = Dense([600, 400], tf.nn.elu)(self.encoded)
        self.encoded = Dense([400, 100], tf.nn.elu)(self.encoded)

        self.imper_mean = Dense([100, 100], tf.identity)(self.encoded)
        self.imper_log_sigma = Dense([100, 100], tf.identity)(self.encoded)

        self.imper = self.sample_Gaussian(self.imper_mean, self.imper_log_sigma)

        self.decoded = Dense([100, 400], tf.nn.elu)(self.imper)
        self.decoded = Dense([400, 600], tf.nn.elu)(self.decoded)
        self.x_reconstructed = Dense([600, 3000], tf.nn.elu)(self.decoded)

        self.rmse = Losses.rmse_loss(self.x_input, self.x_reconstructed)
        self.kl_div = Losses.KL_loss(self.imper_mean, self.imper_log_sigma)
        self.loss = tf.reduce_mean(800 * self.rmse + self.kl_div, name="vae_cost")

    def train(self):

        train_step = tf.train.AdamOptimizer(learning_rate=.5).minimize(self.loss)
        self.sess.run([tf.initialize_all_variables(),
                       tf.initialize_local_variables()])

        self.dh.init_all_queues()
        for i in range(300):
            img = self.sess.run([self.dh.train_x])[0]
            img = tf.reshape(img, (tf.shape(img)[0], -1))
            img = img.eval()
            if i % 50 == 0:
                err = self.validation()
                print('current batch: ' + str(i), "\tvalidation loss: " + str(err))

            train_step.run(feed_dict={self.x_input: img})

    def validation(self):
        accuracy = []
        for i in range(100):
            img = self.sess.run([self.dh.valid_x])[0]
            img = tf.reshape(img, (tf.shape(img)[0], -1))
            img = img.eval()

            accuracy.append(self.loss.eval(feed_dict={self.x_input: img}))
        return sum(accuracy) / len(accuracy)

    def check(self):
        img = self.sess.run([self.dh.valid_x])[0]
        # plt.figure()
        # plt.subplot(1, 2, 1)
        # plt.imshow(img[0, :, :, 0])

        img = tf.reshape(img, (tf.shape(img)[0], -1))
        img = img.eval()
        img = self.x_reconstructed.eval(feed_dict={self.x_input: img})
        print(img.shape)
        # plt.subplot(1, 2, 2)
        # plt.imshow(np.reshape(img, [-1, 75, 40])[0])
        # plt.show()

    @staticmethod
    def sample_Gaussian(mu, log_sigma):
        """Sample from Gaussian distribution"""
        with tf.name_scope("sample_gaussian"):
            epsilon = tf.random_normal(tf.shape(log_sigma), name="epsilon")
            return mu + epsilon * tf.exp(log_sigma)


VAE()
