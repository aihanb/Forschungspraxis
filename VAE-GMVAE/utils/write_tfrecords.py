#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
to create the basic tfrecords by tf.train.example

'''

'''
tf.train.Example Protocol Buffer

message Example {
 Features features = 1;
};

message Features{
 map<string,Feature> featrue = 1;
};

message Feature{
    oneof kind{
        BytesList bytes_list = 1;
        FloatList float_list = 2;
        Int64List int64_list = 3;
    }
};

'''

import tensorflow as tf
import numpy as np

# create the writer to write the data in test.tfrecords
# writer = tf.python_io.TFRecordWriter('test.tfrecords')
# for i in range(0, 2):
#     a = 0.618 + i
#     b = [2016 + i, 2017 + i]
#     c = np.array([[0, 1, 2], [3, 4, 5]]) + i
#     c = c.astype(np.uint8)
#     c_raw = c.tostring()
#     print('i', i)
#     print('a', a)
#     print('b', b)
#     print('c', c)
#     print('c_raw', c_raw)
#     example = tf.train.Example(
#         features=tf.train.Features(
#             feature={'a': tf.train.Feature(float_list=tf.train.FloatList(value=[a])),
#                      'b': tf.train.Feature(int64_list=tf.train.Int64List(value=b)),
#                      'c': tf.train.Feature(bytes_list=tf.train.BytesList(value=[c_raw]))}))
#     serialized = example.SerializeToString()
#     writer.write(serialized)
#     print('write', i, 'DOWN!')
# writer.close()

# class A():
#     def __init__(self):
#         print('enter A')
#         print('leave A')
#
#
# class B(A):
#     def __init__(self):
#         print('enter B')
#         super().__init__()
#         print('leave B')
#
#
# class C(A):
#     def __init__(self):
#         print('enter C')
#         super().__init__()
#         print('leave C')
#
#
# class D(B, C):
#     def __init__(self):
#         print('enter D')
#         super().__init__()
#         print('leave D')
#
#
# d = D()
labels = np.array([[40], [70], [60], [1], [27], [37], [57], [67], [87], [9]])
# labels = np.array([4, 7, 0, 1, 2, 3, 5, 6, 8, 9])
# labels = labels.reshape(-1)
labels = labels.flatten()
print(labels)
var_2d = np.array([[12.361733, -19.492949, 18.531893, 0.8916614, 1.1481463, 28.068153, -15.72083, -1.6366373],
                   [2, 2, 3, 3, 4, 4, 6, 6],
                   [3, 2, 3, 3, 4, 4, 6, 6],
                   [4, 2, 3, 3, 4, 4, 6, 6],
                   [5, 2, 3, 3, 4, 4, 6, 6],
                   [6, 2, 3, 3, 4, 4, 6, 6],
                   [7, 2, 3, 3, 4, 4, 6, 6],
                   [8, 2, 3, 3, 4, 4, 6, 6],
                   [9, 2, 3, 3, 4, 4, 6, 6],
                   [18.392069, -3.6869812, 22.69221, -10.05542, -4.371022, 14.532849, -37.512333, -13.53836]])
print(var_2d)
print(var_2d.shape)
print(labels.shape)
colors = {0:'black', 1:'grey', 2:'blue', 3:'cyan', 4:'lime', 5:'green', 6:'yellow', 7:'gold', 8:'red', 9:'maroon'}
if(labels is not None):
    for number, color in colors.items():
        print("number: ", number)
        print("colors: ", color)
        x = var_2d[labels == number, 0]
        print(x)


