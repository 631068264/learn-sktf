#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2018/5/18 09:42
@annotation = ''
"""
from datetime import datetime

import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

if False:
    """
    tensorboard --logdir tf_logs/
    """
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs"
    logdir = "{}/run-{}/".format(root_logdir, now)

    housing = fetch_california_housing()
    m, n = housing.data.shape
    housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]
    scaler = StandardScaler()
    scaled_housing_data = scaler.fit_transform(housing.data)
    scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]

    n_epochs = 1000
    learning_rate = 0.01

    X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
    y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
    theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
    y_pred = tf.matmul(X, theta, name="predictions")

    with tf.name_scope("loss") as scope:
        error = y_pred - y
        mse = tf.reduce_mean(tf.square(error), name="mse")

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(mse)

    init = tf.global_variables_initializer()

    mse_summary = tf.summary.scalar('MSE', mse)
    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

    n_epochs = 10
    batch_size = 100
    n_batches = int(np.ceil(m / batch_size))


    def fetch_batch(epoch, batch_index, batch_size):
        np.random.seed(epoch * n_batches + batch_index)  # not shown in the book
        indices = np.random.randint(m, size=batch_size)  # not shown
        X_batch = scaled_housing_data_plus_bias[indices]  # not shown
        y_batch = housing.target.reshape(-1, 1)[indices]  # not shown
        return X_batch, y_batch


    with tf.Session() as sess:  # not shown in the book
        sess.run(init)  # not shown

        for epoch in range(n_epochs):  # not shown
            for batch_index in range(n_batches):
                X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
                if batch_index % 10 == 0:
                    summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
                    step = epoch * n_batches + batch_index
                    file_writer.add_summary(summary_str, step)
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

        best_theta = theta.eval()  # not shown

    file_writer.close()
    print(best_theta)
    print(error.op.name)
    print(mse.op.name)
if False:
    """
    Name Scopes
    """

    a1 = tf.Variable(0, name="a")  # name == "a"
    a2 = tf.Variable(0, name="a")  # name == "a_1"

    with tf.name_scope("param"):  # name == "param"
        a3 = tf.Variable(0, name="a")  # name == "param/a"
        a4 = tf.Variable(0, name="a")  # name == "param/a"

    with tf.name_scope("param"):  # name == "param_1"
        a5 = tf.Variable(0, name="a")  # name == "param_1/a"

    for node in (a1, a2, a3, a4, a5):
        print(node.op.name)
if True:
    def relu(X):
        # with tf.name_scope("relu"):
        with tf.variable_scope("relu", reuse=True):
            """
            get_variable() function to create the shared variable 
            if it does not exist yet, or reuse it if it already exists
            
            If you want to reuse a variable, 
            you need to explicitly say so by setting the variable scope’s reuse attribute to True
            """

            threshold = tf.get_variable("threshold")
            w_shape = (int(X.get_shape()[1]), 1)  # not shown in the book
            w = tf.Variable(tf.random_normal(w_shape), name="weights")  # not shown
            b = tf.Variable(0.0, name="bias")  # not shown
            z = tf.add(tf.matmul(X, w), b, name="z")  # not shown
            return tf.maximum(z, threshold, name="max")  # not shown


    n_features = 3
    X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
    with tf.variable_scope("relu"):
        threshold = tf.get_variable("threshold", shape=(),
                                    initializer=tf.constant_initializer(0.0))
    relus = [relu(X) for i in range(5)]
    output = tf.add_n(relus, name="output")

    file_writer = tf.summary.FileWriter("logs/relu", tf.get_default_graph())
    file_writer.close()
