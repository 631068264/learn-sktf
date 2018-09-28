#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2018/8/3 17:13
@annotation = ''
"""
import datetime as dt

import numpy as np


# 获取年分数
def get_year_deltas(date_list, day_count=365.):
    """
    Return vector of floats with day deltas in years.
    Initial value normalized to zero.
    :param date_list: list or array
    collection of datetime objects
    :param day_count: float
    number of days for a year
    :return:
    delta_list:array
    year fractions
    """
    start = date_list[0]
    delta_list = [(date - start).days / day_count for date in date_list]
    return np.array(delta_list)


class constant_short_rate(object):
    """
    Class for constant short rate discounting
    """

    def __init__(self, name, short_rate):
        """

        :param name:string
         name of the object
        :param short_rate:float(positive)
         constant rate for discounting
        """
        self.name = name
        self.short_rate = short_rate
        if short_rate < 0:
            raise ValueError('Short rate negative.')

    def get_discount_factors(self, date_list, dtobjects=True):
        """
        get discount factors given a list/array of datetime objects or year fractions
        """
        if dtobjects is True:
            dlist = get_year_deltas(date_list)
        else:
            dlist = np.array(date_list)
        dflist = np.exp(self.short_rate * np.sort(-dlist))
        return np.array((date_list, dflist)).T


dates = [dt.datetime(2018, 1, 1), dt.datetime(2018, 7, 1), dt.datetime(2019, 1, 1)]
csr = constant_short_rate('csr', 0.05)
print(csr.get_discount_factors(dates))
# array([[datetime.datetime(2018, 1, 1, 0, 0), 0.951229424500714],
#        [datetime.datetime(2018, 7, 1, 0, 0), 0.9755103387657228],
#        [datetime.datetime(2019, 1, 1, 0, 0), 1.0]], dtype=object)

deltas = get_year_deltas(dates)
print(csr.get_discount_factors(deltas, dtobjects=False))
