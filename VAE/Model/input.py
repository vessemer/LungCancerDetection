import tensorflow as tf
import random

from support.dcmx import DICOMReader
from config import *


class DataHandler:
    def __init__(self, split_convert_data=False):
        if split_convert_data:
            self.split_convert_data()

        self.train_queue, self.valid_queue = self.build_queue([PATH['TRAIN'],
                                                               PATH['VALID']])
        self.train_x = self.read_and_decode(self.train_queue)
        self.valid_x = self.read_and_decode(self.valid_queue)

    @staticmethod
    def split_convert_data():
        """Convert .DICOM data from config.DATA_PATH into .TFRecords
        and split with respect to config.VALIDATION_PART
        resulting dataset will be stored in config.TFRECORDS_PATH_TRAIN
        """

        dcm_reader = DICOMReader()
        files = dcm_reader.collect_dcm_filenames(PATH['DATA'])
        random.shuffle(files)
        split = int(len(files) * TRAIN_PARAMS['VALIDATION_PART'])
        dcm_reader.dcm_to_tfr(files[split:], PATH['TRAIN'], DATA_TRANSFORMATION['SCALE'])
        dcm_reader.dcm_to_tfr(files[:split], PATH['VALID'], DATA_TRANSFORMATION['SCALE'])

    @staticmethod
    def build_queue(paths):
        queues = []
        for path in paths:
            queues.append(tf.train.string_input_producer(
                [path],
                shuffle=TRAIN_PARAMS['SHUFFLE'],
                num_epochs=TRAIN_PARAMS['NUM_EPOCHS']
            ))
        return queues

    @staticmethod
    def data_transformation(image):
        """Put here all data transformation:
        resize image w.r.t. config.LUNG_HEIGHT, config.LUNG_WIDTH

        :return: transformed data
        """
        image_size_const = tf.constant((LUNG_SHAPE['HEIGHT'],
                                        LUNG_SHAPE['WIDTH'], 1),
                                       dtype=tf.int32)

        resized_image = tf.image.resize_image_with_crop_or_pad(image=image,
                                                               target_height=LUNG_SHAPE['HEIGHT'],
                                                               target_width=LUNG_SHAPE['WIDTH'])
        return tf.reshape(resized_image,
                          image_size_const)

    @staticmethod
    def read_and_decode(filename_queue):
        """ Read from filename_queue, parse images
        make data augmentation in data_transformation

        :return: batch augmented input data
        """
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        features = tf.parse_single_example(
            serialized_example,
            features={
                'height': tf.FixedLenFeature([], tf.int64),
                'width': tf.FixedLenFeature([], tf.int64),
                'image_raw': tf.FixedLenFeature([], tf.string),
            })

        image = tf.decode_raw(features['image_raw'], tf.uint8)
        height = tf.cast(features['height'], tf.int32)
        width = tf.cast(features['width'], tf.int32)

        image_shape = tf.pack([height, width, 1])
        image = tf.reshape(image, image_shape)
        image = DataHandler.data_transformation(image)

        return tf.train.shuffle_batch([image],
                                      batch_size=2,
                                      capacity=30,
                                      num_threads=1,
                                      min_after_dequeue=2)

    def init_all_queues(self):
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(coord=self.coord)

    def close_all_queues(self):
        self.coord.request_stop()
        self.coord.join(self.threads)
