#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2018/5/13 17:23
@annotation = ''
"""

import numpy as np
from matplotlib import pyplot as plt


def plot_learning_curves(model, X, y):
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    train_errors, test_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_test_predict = model.predict(X_test)
        train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
        test_errors.append(mean_squared_error(y_test_predict, y_test))
    plt.xlabel('data set size')
    plt.ylabel('mse')
    plt.plot(np.sqrt(train_errors), "r", linewidth=2, label="train")
    plt.plot(np.sqrt(test_errors), "b", linewidth=3, label="test")
    plt.legend()
    plt.show()
