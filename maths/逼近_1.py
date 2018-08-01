#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2018/8/1 14:21
@annotation = ''
"""
import matplotlib.pyplot as plt
import numpy as np


def f(x):
    x = x + 0.15 * np.random.standard_normal(len(x))
    return np.sin(x) + 0.5 * x + 0.25 * np.random.standard_normal(len(x))


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


x = np.linspace(-2 * np.pi, 2 * np.pi, 50)
plt.plot(x, f(x), 'bo')
plt.plot(x, np_deg(x, f(x), 7), 'r')
plt.grid(True)
plt.xlabel('x')
plt.ylabel('f(x)')

plt.show()

print(mse(f(x), np_deg(x, f(x), 7)))
