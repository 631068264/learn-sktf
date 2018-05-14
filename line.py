#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2018/5/11 15:08
@annotation = ''
"""
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import SGDRegressor, LinearRegression

import util

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
# add x0 = 1 to each instance
X_b = np.c_[np.ones((100, 1)), X]
X_new = np.array([[0], [2]])
if False:
    """
    Use theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y) 直接
    when the number of features grows large  The Normal Equation gets very slow
    """

    # X_b = X
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    print(theta_best)

    # X_new_b = X_new
    X_new_b = np.c_[np.ones((2, 1)), X_new]  # add x0 = 1 to each instance
    y_predict = X_new_b.dot(theta_best)
    print(y_predict)

    plt.plot(X_new, y_predict, "r-")
    plt.plot(X, y, "b.")
    plt.axis([0, 2, 0, 15])
    plt.show()
if False:
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    print(lin_reg.intercept_, lin_reg.coef_)
    print(lin_reg.predict(X_new))
if False:
    """
    Batch Gradient Descent
    
    它使用整个训练集来计算每一步的梯度，这使得训练集很大时非常缓慢
    """
    eta = 0.1  # learning rate
    n_iterations = 1000
    m_instance_num = 100
    theta = np.random.randn(2, 1)  # random initialization
    for iteration in range(n_iterations):
        gradients = 2 / m_instance_num * X_b.T.dot(X_b.dot(theta) - y)
        theta = theta - eta * gradients

    print(theta)
if False:
    """
    Stochastic Gradient Descent
    随机梯度下降只是在每个步骤中在训练集中选取一个随机实例，并仅基于该单个实例计算梯度
    由于其随机性，该算法比批处理梯度成本函数会上下反弹 只会平均减少 随着时间的推移，
    它将最终接近最小值，但一旦它到达那里，它将继续反弹，永远不会稳定下来。
    所以一旦算法停止，最终的参数值是好的，但不是最优的
    
    
    当成本函数非常不规则时，这实际上可以帮助算法跳出局部最小值，
    因此，随机性很好地摆脱局部最优。但不好，因为这意味着该算法永远无法最小化
    
    方法是
    逐渐降低学习率。这些步骤开始较大（这有助于快速进展并避免局部最小值），
    然后变得越来越小，从而使算法在全局最小值处达到最小。（simulated annealing模拟退火）
    """
    n_epochs = 50
    t0, t1 = 5, 50  # learning schedule hyperparameters


    def learning_schedule(t):
        return t0 / (t + t1)


    theta = np.random.randn(2, 1)  # random initialization
    for epoch in range(n_epochs):
        for i in range(m_instance_num):
            random_index = np.random.randint(m_instance_num)
            xi = X_b[random_index:random_index + 1]
            yi = y[random_index:random_index + 1]
            gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
            eta = learning_schedule(epoch * m_instance_num + i)
            theta = theta - eta * gradients
    print(theta)
    """
    Mini-batch Gradient Descent
    小批量的小随机实例集上的梯度

    进展比SGD更不稳定，特别是在相当大的小批量时。
    因此，小批量GD最终会走得比SGD更接近最小值。但另一方面，它可能难以摆脱局部最小值
    """

if False:
    sgd_reg = SGDRegressor(max_iter=50, penalty=None, eta0=0.1)
    sgd_reg.fit(X, y.ravel())
    print(sgd_reg.intercept_, sgd_reg.coef_)
    util.plot_learning_curves(sgd_reg, X, y.ravel())

if False:
    """
    Polynomial Regression
    
    y = ax2 + bx + c
    
    two features a and b, PolynomialFeatures with degree=3 would not only add the features a2, a3, b2, and b3, 
    but also the combinations ab, a2b, and ab2.
    """
    m = 100
    X = 6 * np.random.rand(m, 1) - 3
    y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)

    from sklearn.preprocessing import PolynomialFeatures

    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly_features.fit_transform(X)
    lin_reg = LinearRegression()
    lin_reg.fit(X_poly, y)
    print(lin_reg.intercept_, lin_reg.coef_)
    coef = lin_reg.coef_[0]
    print(str(coef[1]) + ' X^2 + ' + str(coef[0]) + 'X + ' + str(lin_reg.intercept_[0]))

    from sklearn.pipeline import Pipeline

    polynomial_regression = Pipeline((
        ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
        ("sgd_reg", LinearRegression()),
    ))
    util.plot_learning_curves(polynomial_regression, X, y)

if False:
    """
    Ridge Regression 
    超参数α控制你想要规范模型的程度。
    如果α= 0，则岭回归就是线性回归。
    如果α非常大，则所有权重非常接近零
    
    对于正则化模型 scale非常重要
    """
    from sklearn.linear_model import Ridge

    # Alternatively you can use the Ridge class with the "sag" solver. Stochastic Average GD is a variant of SGD.
    # Scikit-Learn using a closed-form solu‐ tion (a variant of Equation 4-9 using a matrix factorization technique by André-Louis Cholesky):
    # ridge_reg = Ridge(alpha=1, solver="cholesky")
    ridge_reg = Ridge(alpha=1, solver="saga")
    ridge_reg.fit(X, y)
    print(ridge_reg.predict([[1.5]]))
    util.plot_learning_curves(ridge_reg, X, y)

    sgd_reg = SGDRegressor(penalty="l2")
    sgd_reg.fit(X, y.ravel())
    print(sgd_reg.predict([[1.5]]))
    util.plot_learning_curves(sgd_reg, X, y.ravel())

if False:
    """
    Lasso Regression
    所有权重归零
    """
    from sklearn.linear_model import Lasso

    lasso_reg = Lasso(alpha=0.1)
    lasso_reg.fit(X, y)
    print(lasso_reg.predict([[1.5]]))
    util.plot_learning_curves(lasso_reg, X, y)

    sgd_reg = SGDRegressor(penalty="l1")
    sgd_reg.fit(X, y.ravel())
    print(sgd_reg.predict([[1.5]]))
    util.plot_learning_curves(sgd_reg, X, y.ravel())

if False:
    """
    When r = 0, Elastic Net is equivalent to Ridge Regression
    when r = 1, it is equivalent to Lasso Regression
    l1_ratio = r
    
    Ridge is a good default 
    Elastic Net比Lasso更受欢迎  
    
    Lasso 不稳定
    feature number > instance number 
    or several features are strongly correlated
    """
    from sklearn.linear_model import ElasticNet

    elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
    elastic_net.fit(X, y)
    print(elastic_net.predict([[1.5]]))
    util.plot_learning_curves(elastic_net, X, y)
