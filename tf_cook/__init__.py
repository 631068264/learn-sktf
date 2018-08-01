#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2018/6/18 12:09
@annotation = ''
"""
import tensorflow as tf


def get_available_device(device_type='CPU'):
    from tensorflow.python.client import device_lib as _device_lib
    local_device_protos = _device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == device_type]

print(get_available_device())
# sess = tf.Session()
# with tf.device('/cpu:0'):
#     a = tf.constant([1.0, 3.0, 5.0], shape=[1, 3])
#     b = tf.constant([2.0, 4.0, 6.0], shape=[3, 1])
#     with tf.device('/gpu:0'):
#         c = tf.matmul(a, b)
#         c = tf.reshape(c, [-1])
#     with tf.device('/gpu:1'):
#         d = tf.matmul(b, a)
#         flat_d = tf.reshape(d, [-1])
#     combined = tf.multiply(c, flat_d)
# print(sess.run(combined))
