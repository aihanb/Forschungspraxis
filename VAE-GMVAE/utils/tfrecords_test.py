#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
read the data from tfrecords (/train_20k/dev_pdf_20k_splice_1f_cmn/data_1.tfrecords)
And test the dimension of the tfrecords vector, also the data and label in this tfrecords

'''

import tensorflow as tf
import numpy as np


# def read_test(input_file):
#
#     # use dataset to load the tfrecords
#     dataset = tf.data.TFRecordDataset(input_file)
#     print('dataset: ', dataset)
#     # dataset = dataset.map(_parse_record)
#     # iterator = dataset.make_one_shot_iterator()
#     #
#     # with tf.Session() as sess:
#     #     features = sess.run(iterator.)
#
#
#
# read_test('test.tfrecords')
# read_test('data_1.tfrecords')
#
# # record_iterator = tf.python_io.tf_record_iterator()

# for example in tf.python_io.tf_record_iterator("data_1.tfrecords"):
#     print(tf.train.Example.FromString(example))

# def _parse_function(self, example_proto):
#     """
#
#     Creates parse function for loading TFRecords files
#
#
#
#     :param example_proto:   prototype coming from a TFRecords file
#
#     :return:                data from TFRecords files
#
#     """
#
#     # with tf.variable_scope('DataFeedingHelper/parse_function'):
#     keys_to_features = {'x': tf.FixedLenFeature(self._dim_features, tf.float32),
#
#                         'y': tf.FixedLenFeature(self._dim_hard, tf.float32)}
#     print('keys_to_features: ', keys_to_features)
#
#     parsed_features = tf.parse_single_example(example_proto, keys_to_features)
#
#     print("parsed_features['x']: ", parsed_features['x'])
#     print("parsed_features['y']: ", parsed_features['y'])
#
#     return parsed_features['x'], parsed_features['y']
#
# # filenames = ["data_1.tfrecords"]
# # dataset = tf.data.TFRecordDataset(filenames)
# # dataset = dataset.map(_parse_function)
#
# def _input_fn(self):
#     """
#
#     Create Dataset using TF-API and iterate through the dict
#
#     """
#
#     with tf.variable_scope('DataFeedingHelper/input_fn'):
#         # get self.train, self.test and selt.dev references
#
#         for key, item in self._dict_lists.items():
#             dataset = tf.data.TFRecordDataset(item)
#
#             # Parse the record into tensors.
#
#             dataset = dataset.map(self._parse_function)
#
#             dataset = dataset.shuffle(100000, reshuffle_each_iteration=False)
#
#             dataset = dataset.batch(self._batch_size, drop_remainder=True)
#
#             # dict_ref[key] = dataset.make_initializable_iterator()
#
#             setattr(self, key, dataset.make_initializable_iterator())


def dataset_input_fn():
    filenames = ["data_1.tfrecords"]
    dataset = tf.data.TFRecordDataset(filenames)

    def _parse_function(record):
        """

        Creates parse function for loading TFRecords files



        :param example_proto:   prototype coming from a TFRecords file

        :return:                data from TFRecords files

        """

        # with tf.variable_scope('DataFeedingHelper/parse_function'):
        keys_to_features = {'x': tf.FixedLenFeature(self._dim_features, tf.float32),

                            'y': tf.FixedLenFeature(self._dim_hard, tf.float32)}
        print('keys_to_features: ', keys_to_features)

        parsed_features = tf.parse_single_example(record, keys_to_features)

        print("parsed_features['x']: ", parsed_features['x'])
        print("parsed_features['y']: ", parsed_features['y'])

        return parsed_features['x'], parsed_features['y']

    dataset = dataset.map(_parse_function)
    dataset = dataset.shuffle(100000, reshuffle_each_iteration=False)
    dataset = dataset.batch(32)
    dataset = dataset.repeat(10)

    return dataset

dataset = dataset_input_fn()
print('dataset: ', dataset)