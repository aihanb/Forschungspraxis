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
writer = tf.python_io.TFRecordWriter('test.tfrecords')
for i in range(0, 2):
    a = 0.618 + i
    b = [2016 + i, 2017 + i]
    c = np.array([[0, 1, 2], [3, 4, 5]]) + i
    c = c.astype(np.uint8)
    c_raw = c.tostring()
    print('i', i)
    print('a', a)
    print('b', b)
    print('c', c)
    print('c_raw', c_raw)
    example = tf.train.Example(
        features=tf.train.Features(
            feature={'a': tf.train.Feature(float_list=tf.train.FloatList(value=[a])),
                     'b': tf.train.Feature(int64_list=tf.train.Int64List(value=b)),
                     'c': tf.train.Feature(bytes_list=tf.train.BytesList(value=[c_raw]))}))
    serialized = example.SerializeToString()
    writer.write(serialized)
    print('write', i, 'DOWN!')
writer.close()
