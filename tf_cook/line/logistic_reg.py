#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2018/6/21 10:54
@annotation = ''
"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets

iris = datasets.load_iris()
x_vals = iris.data
y_vals = iris.target

# birthdata_url = 'https://www.umass.edu/statdata/statdata/data/lowbwt.dat'
# birth_file = requests.get(birthdata_url)
# birth_data = birth_file.text.split('\r\n')[5:]
# birth_header = [x for x in birth_data[0].split(' ') if len(x) >= 1]
# birth_data = [[float(x) for x in y.split(' ') if len(x) >= 1] for y in birth_data[1:] if len(y) >= 1]
# # Pull out target variable
# y_vals = np.array([x[1] for x in birth_data])
# # Pull out predictor variables (not id, not target, and not birthweight)
# x_vals = np.array([x[2:9] for x in birth_data])

train_indices = np.random.choice(len(x_vals), round(len(x_vals) * 0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]


def normalize_cols(m):
    col_max = m.max(axis=0)
    col_min = m.min(axis=0)
    return (m - col_min) / (col_max - col_min)


x_vals_train = np.nan_to_num(normalize_cols(x_vals_train))
x_vals_test = np.nan_to_num(normalize_cols(x_vals_test))

# Declare batch size
batch_size = 25

# Initialize placeholders
x_data = tf.placeholder(shape=[None, x_vals.shape[1]], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# Create variables for linear regression
A = tf.Variable(tf.random_normal(shape=[x_vals.shape[1], 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))

# Declare model operations
# y = sigmoid(Ax + b)
model_output = tf.add(tf.matmul(x_data, A), b)

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=y_target))
init = tf.global_variables_initializer()

# Declare optimizer
my_opt = tf.train.GradientDescentOptimizer(0.01)
train_step = my_opt.minimize(loss)

# Actual Prediction
prediction = tf.round(tf.sigmoid(model_output))
predictions_correct = tf.cast(tf.equal(prediction, y_target), tf.float32)
accuracy = tf.reduce_mean(predictions_correct)

with tf.Session() as sess:
    init.run()
    # Training loop
    loss_vec = []
    train_acc = []
    test_acc = []
    for i in range(1500):
        rand_index = np.random.choice(len(x_vals_train), size=batch_size)
        rand_x = x_vals_train[rand_index]
        rand_y = np.transpose([y_vals_train[rand_index]])
        sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

        temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
        loss_vec.append(temp_loss)
        temp_acc_train = sess.run(accuracy, feed_dict={x_data: x_vals_train, y_target: np.transpose([y_vals_train])})
        train_acc.append(temp_acc_train)
        temp_acc_test = sess.run(accuracy, feed_dict={x_data: x_vals_test, y_target: np.transpose([y_vals_test])})
        test_acc.append(temp_acc_test)
        if (i + 1) % 300 == 0:
            print('Loss = ' + str(temp_loss))

    # Plot loss over time
    plt.plot(loss_vec, 'k-')
    plt.title('Cross Entropy Loss per Generation')
    plt.xlabel('Generation')
    plt.ylabel('Cross Entropy Loss')
    plt.show()

    # Plot train and test accuracy
    plt.plot(train_acc, 'k-', label='Train Set Accuracy')
    plt.plot(test_acc, 'r--', label='Test Set Accuracy')
    plt.title('Train and Test Accuracy')
    plt.xlabel('Generation')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()

    # A_ = sess.run(A)
    # B_ = sess.run(b)

    # print()
