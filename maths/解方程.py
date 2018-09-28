#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2018/8/3 11:20
@annotation = ''
"""
# 二元一次方程组

"""
x + 2y = 3
4x ＋ 5y = 6
"""
import numpy as np

A = np.array([[1, 2], [4, 5]])
b = np.array([3, 6]).T
r = np.linalg.solve(A, b)
print(r)
