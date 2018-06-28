#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2018/6/26 10:11
@annotation = ''
"""
# import tensorflow as tf
#
# a = tf.Variable([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]])
# index_a = tf.Variable([0, 2])
#
# b = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# index_b = tf.Variable([2, 4, 6, 8])
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(sess.run(tf.gather(a, index_a)))
#     print(sess.run(tf.gather(b, index_b)))

"""timeline"""
# import tensorflow as tf
# from tensorflow.python.client import timeline
#
# a = tf.constant([1.0, 3.0, 5.0], shape=[1, 3])
# b = tf.constant([2.0, 4.0, 6.0], shape=[3, 1])
# c = tf.matmul(a, b)
# c = tf.reshape(c, [-1])
# d = tf.matmul(b, a)
# flat_d = tf.reshape(d, [-1])
# combined = tf.multiply(c, flat_d)
#
# with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
#     options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
#     run_metadata = tf.RunMetadata()
#     config = {
#         'options': options,
#         'run_metadata': run_metadata,
#     }
#     print(sess.run(combined, **config))
#
#     # Create the Timeline object, and write it to a json file
#     fetched_timeline = timeline.Timeline(run_metadata.step_stats)
#     chrome_trace = fetched_timeline.generate_chrome_trace_format()
#     with open('timeline_01.json', 'w') as f:
#         f.write(chrome_trace)

