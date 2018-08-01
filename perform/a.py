#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2018/7/22 18:11
@annotation = ''
"""
# import numpy as np
#
# a = np.array([[1, 2], [3, 4]])
# # b = np.expand_dims(a, axis=2)
# # print(b)
# print(a)
# print(a.shape)
# print()
# print(a.transpose())
# print()
# print(a.transpose((0, 1)))


# win_size = 2、、
# X = []
#
# y = []
# for i in range(100):
#     X.append((i, i + win_size))
#     y.append((i + win_size, i + 2 * win_size))
#
# print(X)
# print(y)
#

import pandas as pd
import statsmodels.api as sm

df = pd.DataFrame({"A": [10, 20, 30, 40, 50], "B": [20, 30, 10, 40, 50], "C": [32, 234, 23, 23, 42523]})
result = sm.OLS(df['A'], sm.add_constant(df[['B', 'C']])).fit()
# print(result.summary())
print(df['B'] * result.params['B'] + df['C'] * result.params['C'] + result.params['const'])

from sklearn import linear_model

reg = linear_model.LinearRegression()
reg.fit(df[['B', 'C']], df['A'])
# print(reg.coef_)
# print(reg.intercept_)
print(df['B'] * reg.coef_[0] + df['C'] * reg.coef_[1] + reg.intercept_)
