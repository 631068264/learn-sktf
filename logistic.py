#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2018/5/13 18:27
@annotation = ''
"""
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()
# petal width
X = iris["data"][:, 3:]
# 1 if Iris-Virginica, else 0
y = (iris["target"] == 2).astype(np.int)
from sklearn.linear_model import LogisticRegression

"""
LogisticRegression
Softmax回归分类器一次只能预测一个类（即，它是多类的，而不是多输出的），
所以它只能用于互斥类，如不同类型的植物。你不能用它来识别一张照片中的多个人。
"""
if False:
    log_reg = LogisticRegression()
    log_reg.fit(X, y)

    X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
    y_proba = log_reg.predict_proba(X_new)
    plt.plot(X_new, y_proba[:, 1], "g-", label="Iris-Virginica")
    plt.plot(X_new, y_proba[:, 0], "b--", label="Not Iris-Virginica")
    plt.legend()
    plt.show()
"""
多项

Scikit-Learn's LogisticRegression在两个以上的类上进行训练时默认使用“一对多”，
但您可以将multi_class超参数设置为“多项”来将其切换为Softmax回归 随机平均梯度下降解算器。
您还必须指定一个支持Softmax回归的解算器，例如“lbfgs”求解器
"""
X = iris["data"][:, (2, 3)]
y = iris["target"]
softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10)
softmax_reg.fit(X, y)
print(softmax_reg.predict([[5, 2]]))
print(softmax_reg.predict_proba([[5, 2]]))
