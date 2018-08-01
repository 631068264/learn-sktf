#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2018/8/1 15:52
@annotation = ''
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl

def fm(x, y):
    z = np.sin(x) + 0.05 * x ** 2 + np.sin(y) + 0.05 * y ** 2
    return z

# def fo((x, y)):
#     z = np.sin(x) + 0.05 * x ** 2 + np.sin(y) + 0.05 * y ** 2
#     print('%8.4f %8.4f %8.4f' % (x, y, z))
#     return z
output = False
def fo(x):
    z = np.sin(x[0]) + 0.05 * x[0] ** 2 + np.sin(x[1]) + 0.05 * x[1] ** 2
    if output == True:
        print('%8.4f %8.4f %8.4f' % (x[0], x[1], z))
    return z

x = np.linspace(-10, 10, 50)
y = np.linspace(-10, 10, 50)
X, Y = np.meshgrid(x, y)
Z = fm(X, Y)

fig = plt.figure(figsize=(9, 6))
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, rstride=2, cstride=2, cmap=mpl.cm.coolwarm, linewidth=0.5, antialiased=True)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x,y)')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

import scipy.optimize as spo
# -10到10step0.1 两两组合 求最小
global_opt = spo.brute(fo, ((-10, 10.1, 0.1), (-10, 10.1, 0.1)), finish=None)
# global_opt = spo.brute(fo, ((-10, 10.1, 0.1), (-10, 10.1, 0.1)), )
print(fm(global_opt[0],global_opt[1]))
# 局部
output=True
"""
fmin函数的输人是需要最小化的函数和起始参数值。此外，可以定义输入参数宽容度和函数值宽容度， 以及最大迭代及函数调用次数

在许多凸优化问题中， 建议在局部优化之前进行全局优化。 
主要原因是局部凸优化算法很容易陷人某个局部最小值（所谓的 “盆地跳跃 ” （ basin hopping））， 
而忽略“更好”的局部最小值和全局最小值。
"""
opt2 = spo.fmin(fo, global_opt, xtol=0.001, ftol=0.001, maxiter=15, maxfun=20)
print(fm(opt2[0], opt2[1]))