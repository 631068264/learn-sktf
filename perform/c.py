#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2018/7/22 18:11
@annotation = ''
"""
from math import *

import numba as nb
import numpy as np
I = 500000
# a_py = np.arange(I)
a_py = range(I)


@nb.jit()
def f(x):
    return abs(cos(x)) ** 0.5 + sin(2 + 3 * x)


@nb.jit()
def f1(a):
    res = []
    for x in a:
        res.append(f(x))
    return res


@nb.jit()
def f2(a):
    return [f(x) for x in a]


print(f1(a_py))
print(f2(a_py))
