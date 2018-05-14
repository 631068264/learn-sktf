#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2018/5/13 21:13
@annotation = ''
"""

"""
能够执行线性或非线性分类，回归，甚至异常值检测。它是机器学习中最受欢迎的模型之一

scale非常重要
"""

import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

iris = datasets.load_iris()
X = iris["data"][:, (2, 3)]  # petal length, petal width
y = (iris["target"] == 2).astype(np.float64)  # Iris-Virginica
if False:
    """
    
    线性svm
    overfit reduce C
    loss hyperparameter to "hinge"
    Prefer dual=False when n_samples > n_features.
    """

    svm_clf = Pipeline((
        ("scaler", StandardScaler()),
        ("linear_svc", LinearSVC(C=1, loss="hinge")),
    ))
    svm_clf.fit(X, y)
    print(svm_clf.predict([[5.5, 1.7]]))

if False:
    """
    Adding polynomial features is simple to implement 
    and can work great with all sorts of Machine Learning algorithms (not just SVMs)
    
    with a high polynomial degree it creates a huge number of features, making the model too slow
    
    
    Use kernel
    解决非线性问题的另一种技术是添加使用相似性函数计算的特征，该函数可以测量每个实例与特定地标相似的程度。
    
    gamma like C
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures

    polynomial_svm_clf = Pipeline((
        ("poly_features", PolynomialFeatures(degree=3)),
        ("scaler", StandardScaler()),
        ("svm_clf", LinearSVC(C=10, loss="hinge"))
    ))
    polynomial_svm_clf.fit(X, y)

    from sklearn.svm import SVC

    poly_kernel_svm_clf = Pipeline((
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))
    ))
    poly_kernel_svm_clf.fit(X, y)

    rbf_kernel_svm_clf = Pipeline((
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="rbf", gamma=5, C=0.001))
    ))
    rbf_kernel_svm_clf.fit(X, y)

    """
    LinearSVC比SVC快得多（ker nel =“linear”）），特别是如果训练集非常大或者它有很多特征。
    如果训练集不太大，则应该尝试高斯RBF内核;它在大多数情况下运作良好。
    
    """

if False:
    from sklearn.svm import LinearSVR
    """
    epsilon -> street width
    C large regularization small
    
    """
    svm_reg = LinearSVR(epsilon=1.5)
    svm_reg.fit(X, y)
    """
    SVR类是SVC类的回归等价物，LinearSVR类是LinearSVC类的回归等价物。 
    LinearSVR类与训练集的大小成线性关系（就像LinearSVC类一样），而当训练集变大时SVR类变得太慢（就像SVC类一样）
    """
    from sklearn.svm import SVR

    svm_poly_reg = SVR(kernel="poly", degree=2, C=100, epsilon=0.1)
    svm_poly_reg.fit(X, y)