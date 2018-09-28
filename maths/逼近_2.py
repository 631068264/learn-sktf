#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2018/8/1 14:21
@annotation = ''
"""
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits .mplot3d import Axes3D
import matplotlib as mpl
import statsmodels.api as sm

def f(x, y):
    return np.sin(x) + 0.25 * x + np.sqrt(y) + 0.05 * y ** 2


def np_deg(x, y, deg=1):
    """线性单项式x x^2 x^3"""
    p = np.polyfit(x, y, deg=deg)
    """
    N = len(p)
    p[0]*x**(N-1) + p[1]*x**(N-2) + ... + p[N-2]*x + p[N-1]
    """
    return np.polyval(p, x)


def mse(origin_y, test_y):
    return np.sum((origin_y - test_y) ** 2) / len(test_y)


x = np.linspace(0, 10, 20)
y = np.linspace(0, 10, 20)
X, Y = np.meshgrid(x, y)

Z = f(X, Y)
x = X.flatten()
y = Y.flatten()





matrix = np.zeros((len(x), 6 + 1))
matrix[:, 6] = np.sqrt(y)
matrix[:, 5] = np.sin(x)
matrix[:, 4] = y ** 2
matrix[:, 3] = x ** 2
matrix[:, 2] = y
matrix[:, 1] = x
matrix[:, 0] = 1



model = sm.OLS(f(x, y), matrix).fit()
print(model.summary())


def reg_func(a, x, y):
    f6 = a[6] * np.sqrt(y)
    f5 = a[5] * np.sin(x)
    f4 = a[4] * y ** 2
    f3 = a[3] * x ** 2
    f2 = a[2] * y
    f1 = a[1] * x
    f0 = a[0] * 1
    return (f6 + f5 + f4 + f3 + f2 + f1 + f0)


RZ = reg_func(model.params, X, Y)

fig = plt.figure(figsize=(9, 6))
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, rstride=2, cstride=2, cmap=mpl.cm.coolwarm, linewidth=0.5, antialiased=True)
surf1 = ax.plot_surface(X, Y, RZ, rstride=2, cstride=2, cmap=mpl.cm.coolwarm, linewidth=0.5, antialiased=True)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

