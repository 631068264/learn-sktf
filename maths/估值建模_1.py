#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2018/8/3 17:13
@annotation = ''
"""
import datetime as dt

import numpy as np

from maths.估值建模 import constant_short_rate


class market_environment(object):
    def __init__(self, name, pricing_date):
        self.name = name
        self.pricing_date = pricing_date
        self.constants = {}
        self.lists = {}
        self.curves = {}

    def add_constant(self, key, constant):
        self.constants[key] = constant

    def get_constant(self, key):
        return self.constants[key]

    def add_list(self, key, list_object):
        self.lists[key] = list_object

    def get_list(self, key):
        return self.lists[key]

    def add_curve(self, key, curve):
        self.curves[key] = curve

    def get_curve(self, key):
        return self.curves[key]

    def add_environment(self, env):
        for key in env.constants:
            self.constants[key] = env.constants[key]
        for key in env.lists:
            self.lists[key] = env.lists[key]
        for key in env.curves:
            self.curves[key] = env.curves[key]



dates = [dt.datetime(2018, 1, 1), dt.datetime(2018, 7, 1), dt.datetime(2019, 1, 1)]
csr = constant_short_rate('csr', 0.05)
me_1 = market_environment('me_1', dt.datetime(2018, 1, 1))
me_1.add_list('symbols', ['AAPL', 'MSFT', 'FB'])
me_1.get_list('symbols')
# ['AAPL', 'MSFT', 'FB']

me_2 = market_environment('me_2', dt.datetime(2018, 1, 1))
me_2.add_constant('volatility', 0.2)
me_2.add_curve('short_rate', csr)
me_2.get_curve('short_rate')
# <__main__.constant_short_rate at 0x1a226781cc0>

me_1.add_environment(me_2)
me_1.get_curve('short_rate')
# <__main__.constant_short_rate at 0x1a226781cc0>

print(me_1.constants)
# {'volatility': 0.2}

print(me_1.lists)
# {'symbols': ['AAPL', 'MSFT', 'FB']}

print(me_1.curves)
# {'short_rate': <__main__.constant_short_rate at 0x1a226781cc0>}

print(me_1.get_curve('short_rate').short_rate)
# 0.05