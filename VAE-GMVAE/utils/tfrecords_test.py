#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
read the data from tfrecords (/train_20k/dev_pdf_20k_splice_1f_cmn/data_1.tfrecords)
And test the dimension of the tfrecords vector, also the data and label in this tfrecords

'''

import tensorflow as tf
import numpy as np


def _parse_function(example_proto):
    """

    Creates parse function for loading TFRecords files



    :param example_proto:   prototype coming from a TFRecords file

    :return:                data from TFRecords files

    """

    # with tf.variable_scope('DataFeedingHelper/parse_function'):
    keys_to_features = {'x': tf.FixedLenFeature([39], tf.float32),

                        'y': tf.FixedLenFeature([1], tf.float32)}

    parsed_features = tf.parse_single_example(example_proto, keys_to_features)

    # print("parsed_features['x']: ", parsed_features['x'])
    # print("parsed_features['y']: ", parsed_features['y'])

    return parsed_features['x'], parsed_features['y']


def get_dataset(fname):
    """

    Create Dataset using TF-API


    :param fname:   filenames

    :return:        dataset from TFRecords files

    """
    dataset = tf.data.TFRecordDataset(fname)
    dataset = dataset.map(_parse_function)
    # dataset = dataset.shuffle(buffer_size=100000)
    dataset = dataset.batch(138420)
    # dataset = dataset.batch(10000)
    # dataset = dataset.batch(1)
    # dataset = dataset.repeat(1)
    # dataset = dataset.shuffle(buffer_size=1000).batch(32).repeat(10)
    return dataset


dataset = get_dataset('data_1.tfrecords')
iterator = dataset.make_initializable_iterator()
x, y = iterator.get_next()

with tf.Session() as sess:
    sess.run(iterator.initializer)
    try:
        while True:
            data, label = sess.run([x, y])
            # print("type(data): ", type(data))
            # print("data.shape: ", data.shape)
            # print("type(label): ", type(label))
            # print("label.shape: ", label.shape)
    except tf.errors.OutOfRangeError:
        print("END!")

print("type(data): ", type(data))
print("data.shape: ", data.shape)
print("type(label): ", type(label))
print("label.shape: ", label.shape)
# print("data: ", data)
# print("label: ", label)
data_dim = data.shape[1]
num_data = data.shape[0]
train_size = int(num_data*0.8)
valid_size = int(num_data*0.1)
test_size = num_data - train_size - valid_size
print("train_size: ", train_size)
print("valid_size: ", valid_size)
print("test_size: ", test_size)

x_train = data[:train_size]
x_valid = data[train_size:(train_size+valid_size)]
x_test = data[(train_size+valid_size):]

x_train = np.reshape(x_train, [-1, 39, 1, 1])
x_valid = np.reshape(x_valid, [-1, 39, 1, 1])
x_test = np.reshape(x_test, [-1, 39, 1, 1])

print("////// After np.reshape //////")
print("x_train.shape: ", x_train.shape)
print("x_valid.shape: ", x_valid.shape)
print("x_test.shape: ", x_test.shape)

# train_dataset = Dataset(x_train, data.train.labels)
# valid_dataset = Dataset(x_valid, data.train.labels)
# test_dataset = Dataset(x_test, data.test.labels)
