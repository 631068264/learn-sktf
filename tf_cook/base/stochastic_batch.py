#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2018/5/28 10:08
@annotation = ''
"""
import numpy as np
import tensorflow as tf

import util

x_vals = np.random.normal(1, 0.1, 100)
y_vals = np.repeat(10., 100)
x_data = tf.placeholder(shape=[1], dtype=tf.float32)
y_target = tf.placeholder(shape=[1], dtype=tf.float32)

A = tf.Variable(tf.random_normal(shape=[1]))

my_output = tf.multiply(x_data, A)
loss = tf.square(my_output - y_target)
opt = tf.train.GradientDescentOptimizer(0.02)
train_step = opt.minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run(session=sess)
    for i in range(100):
        rand_index = np.random.choice(100)
        rand_x = [x_vals[rand_index]]
        rand_y = [y_vals[rand_index]]
        sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
        if (i + 1) % 25 == 0:
            print('Step #' + str(i + 1) + ' A = ' + str(sess.run(A)))
            print('Loss = ' + str(sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})))

util.reset_graph()
x_vals = np.random.normal(1, 0.1, 100)
y_vals = np.repeat(10., 100)
x_data = tf.placeholder(shape=[1], dtype=tf.float32)
y_target = tf.placeholder(shape=[1], dtype=tf.float32)

# Create variable (one model parameter = A)
A = tf.Variable(tf.random_normal(shape=[1]))

# Add operation to graph
my_output = tf.multiply(x_data, A)

# Add L2 loss operation to graph
loss = tf.square(my_output - y_target)

# Create Optimizer
my_opt = tf.train.GradientDescentOptimizer(0.02)
train_step = my_opt.minimize(loss)

loss_stochastic = []
init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run(session=sess)
    # Run Loop
    for i in range(100):
        rand_index = np.random.choice(100)
        rand_x = [x_vals[rand_index]]
        rand_y = [y_vals[rand_index]]
        sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
        if (i + 1) % 5 == 0:
            print('Step #' + str(i + 1) + ' A = ' + str(sess.run(A)))
            temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
            print('Loss = ' + str(temp_loss))
            loss_stochastic.append(temp_loss)

util.reset_graph()

# Create data
x_vals = np.random.normal(1, 0.1, 100)
y_vals = np.repeat(10., 100)
x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# Create variable (one model parameter = A)
A = tf.Variable(tf.random_normal(shape=[1, 1]))

# Add operation to graph
my_output = tf.matmul(x_data, A)

# Add L2 loss operation to graph
loss = tf.reduce_mean(tf.square(my_output - y_target))
my_opt = tf.train.GradientDescentOptimizer(0.02)
train_step = my_opt.minimize(loss)

batch_size = 20
loss_batch = []
init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run(session=sess)
    # Run Loop
    for i in range(100):
        rand_index = np.random.choice(100, size=batch_size)
        rand_x = np.transpose([x_vals[rand_index]])
        rand_y = np.transpose([y_vals[rand_index]])
        sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
        if (i + 1) % 5 == 0:
            print('Step #' + str(i + 1) + ' A = ' + str(sess.run(A)))
            temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
            print('Loss = ' + str(temp_loss))
            loss_batch.append(temp_loss)

import matplotlib.pyplot as plt

plt.plot(range(0, 100, 5), loss_stochastic, 'b-', label='Stochastic Loss')
plt.plot(range(0, 100, 5), loss_batch, 'r--', label='Batch Loss, size=20')
plt.legend(loc='upper right', prop={'size': 11})
plt.show()
