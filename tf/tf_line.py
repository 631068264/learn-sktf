#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2018/5/17 16:21
@annotation = ''
"""

import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]

scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]
"""
Using the Normal Equation
"""
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if False:
    X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
    y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
    XT = tf.transpose(X)
    theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)

    with tf.Session() as sess:
        theta_value = theta.eval()

    print(theta_value)

    X = housing_data_plus_bias
    y = housing.target.reshape(-1, 1)
    theta_numpy = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    print(theta_numpy)

    from sklearn.linear_model import LinearRegression

    lin_reg = LinearRegression()
    lin_reg.fit(housing.data, housing.target.reshape(-1, 1))

    print(np.r_[lin_reg.intercept_.reshape(-1, 1), lin_reg.coef_.T])

if False:
    """
    Batch Gradient Descent
    """

    n_epochs = 1000
    learning_rate = 0.01

    X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
    y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
    theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
    y_pred = tf.matmul(X, theta, name="predictions")
    error = y_pred - y

    mse = tf.reduce_mean(tf.square(error), name="mse")
    # # gradients = 2 / m * tf.matmul(tf.transpose(X), error)
    # # 梯度计算优化
    # gradients = tf.gradients(mse, [theta])[0]
    # training_op = tf.assign(theta, theta - learning_rate * gradients)

    # 进一步简化
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                           momentum=0.9)

    training_op = optimizer.minimize(mse)

    init_var = tf.global_variables_initializer()

    with tf.Session() as sess:
        # sess.run(init_var)
        init_var.run()
        for epoch in range(n_epochs):
            if epoch % 100 == 0:
                print("Epoch", epoch, "MSE =", mse.eval())
            sess.run(training_op)

        best_theta = theta.eval()

    print(best_theta)

if False:
    """
    Mini-batch Gradient Descent
    """

    X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
    y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
    # n_epochs = 1000
    learning_rate = 0.01

    theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
    y_pred = tf.matmul(X, theta, name="predictions")
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="mse")
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(mse)

    init = tf.global_variables_initializer()
    n_epochs = 10
    batch_size = 100
    n_batches = int(np.ceil(m / batch_size))


    def fetch_batch(epoch, batch_index, batch_size):
        np.random.seed(epoch * n_batches + batch_index)  # not shown in the book
        indices = np.random.randint(m, size=batch_size)  # not shown
        X_batch = scaled_housing_data_plus_bias[indices]  # not shown
        y_batch = housing.target.reshape(-1, 1)[indices]  # not shown
        return X_batch, y_batch


    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(n_epochs):
            for batch_index in range(n_batches):
                X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

        best_theta = theta.eval()

    print(best_theta)

if False:
    """
    保存sess 检查点

    Saver saves and restores all variables under their own name
    saver = tf.train.Saver({"weights": theta})
    """
    n_epochs = 1000
    learning_rate = 0.01

    X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")  # not shown
    y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")  # not shown
    theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
    y_pred = tf.matmul(X, theta, name="predictions")  # not shown
    error = y_pred - y  # not shown
    mse = tf.reduce_mean(tf.square(error), name="mse")  # not shown
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)  # not shown
    training_op = optimizer.minimize(mse)  # not shown

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    path = 'tmp/my_model.ckpt'
    final_path = 'tmp/my_model_final.ckpt'
    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(n_epochs):
            if epoch % 100 == 0:
                print("Epoch", epoch, "MSE =", mse.eval())
                save_path = saver.save(sess, path)
            sess.run(training_op)

        best_theta = theta.eval()
        save_path = saver.save(sess, final_path)
    print(best_theta)

    with tf.Session() as sess:
        saver.restore(sess, final_path)
        best_theta_restored = theta.eval()
    print(np.allclose(best_theta, best_theta_restored))
