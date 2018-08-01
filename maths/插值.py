#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2018/8/1 15:33
@annotation = ''
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as spi


def f(x):
    return np.sin(x) + 0.5 * x


def spi_k(x, y, k=1):
    """
    在两个相邻数据点之 间进行回归，
    不仅产生的分段插值函数完全匹配数据点，而且函数在数据点上连续可 微分。

    须有排序(且"无噪声" )的数据，该方 法仅限于低维度问题.样条插值的计算要求也更高，在某些用例中可 能导致花费的时间比回归方法长得多.
    """
    p = spi.splrep(x, y, k=k)
    return spi.splev(x, p)


x = np.linspace(-2 * np.pi, 2 * np.pi, 50)
plt.plot(x, f(x), 'b.')
plt.plot(x, spi_k(x, f(x), 1), 'r')
plt.legend(loc=0)
plt.grid(True)
plt.xlabel('x')
plt.ylabel('f(x)')

plt.show()
