#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2018/6/20 14:28
@annotation = ''
"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import util

x_vals = np.linspace(0, 10, 100)
y_vals = x_vals + np.random.normal(0, 1, 100)

x_vals_column = np.transpose(np.matrix(x_vals))
ones_column = np.transpose(np.matrix(np.repeat(1, 100)))

A = tf.constant(np.column_stack((x_vals_column, ones_column)))
b = tf.constant(np.transpose(np.matrix(y_vals)))

with tf.Session() as sess:
    tA_A = tf.matmul(tf.transpose(A), A)
    tA_A_1 = tf.matrix_inverse(tA_A)
    tA_A_1_tA = tf.matmul(tA_A_1, tf.transpose(A))
    x = tf.matmul(tA_A_1_tA, b)

    solution_eval = sess.run(x)

    # Extract coefficients
    slope = solution_eval[0][0]
    y_intercept = solution_eval[1][0]

    print('slope: ' + str(slope))
    print('y_intercept: ' + str(y_intercept))

    best_fit = []
    for i in x_vals:
        best_fit.append(slope * i + y_intercept)

    # Plot the results
    plt.plot(x_vals, y_vals, 'o', label='Data')
    plt.plot(x_vals, best_fit, 'r-', label='Best fit line', linewidth=3)
    plt.legend(loc='upper left')
    plt.show()


util.reset_graph()
A = tf.constant(np.column_stack((x_vals_column, ones_column)))
b = tf.constant(np.transpose(np.matrix(y_vals)))
with tf.Session() as sess:
    tA_A = tf.matmul(tf.transpose(A), A)
    L = tf.cholesky(tA_A)
    tA_b = tf.matmul(tf.transpose(A), b)
    sol1 = tf.matrix_solve(L, tA_b)
    sol2 = tf.matrix_solve(tf.transpose(L), sol1)

    solution_eval = sess.run(sol2)

    # Extract coefficients
    slope = solution_eval[0][0]
    y_intercept = solution_eval[1][0]

    print('slope: ' + str(slope))
    print('y_intercept: ' + str(y_intercept))

    best_fit = []
    for i in x_vals:
        best_fit.append(slope * i + y_intercept)

    # Plot the results
    plt.plot(x_vals, y_vals, 'o', label='Data')
    plt.plot(x_vals, best_fit, 'r-', label='Best fit line', linewidth=3)
    plt.legend(loc='upper left')
    plt.show()