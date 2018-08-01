#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2018/8/1 15:52
@annotation = ''
"""
from math import *

import scipy.optimize as spo


def profit(s):
    a,b = s
    return -(0.5 * sqrt(a * 15 + b * 5) + 0.5 * sqrt(a * 5 + b * 12))


cons = ({'type': 'ineq', 'fun': lambda s: 100 - s[0] * 10 - s[1] * 10})
bnds = ((0, 1000), (0, 1000))
result = spo.minimize(profit, [5, 5], method='SLSQP', bounds=bnds, constraints=cons)
print(result['x'])
print(-result['fun'])
