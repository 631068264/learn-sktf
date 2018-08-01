#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2018/7/22 18:11
@annotation = ''
"""
import operator
from math import *
from numba import jit
# 使用range函数，我们可以高效地生成一个包含 50 万个数值的列表对象
I = 500000
a_py = range(I)


# 很容易转换为一个Python函数
@jit()
def f(x):
    return abs(cos(x)) ** 0.5 + sin(2 + 3 * x)


# 包含显式循环的标准Python函数
def f1(a):
    res = []
    for x in a:
        res.append(f(x))
    return res


# 包含隐含循环的迭代子方法
def f2(a):
    return [f(x) for x in a]


# 包含隐含循环、使用eval的选代子方法
def f3(a):
    ex = 'abs(cos(x))**0.5+sin(2+3*x)'
    return [eval(ex) for x in a]


# Numy向量化实现
import numpy as np

a_np = np.arange(I)


def f4(a):
    return (np.abs(np.cos(a)) ** 0.5 + np.sin(2 + 3 * a))


# 专用库numexpr求数值表达式的值。 这个库内建了多线程执行支持
# numexpr单线程实现
import numexpr as ne


def f5(a):
    ex = 'abs(cos(a))**0.5+sin(2+3*a)'
    ne.set_num_threads(1)
    return ne.evaluate(ex)


# nwexpr多线程实现
def f6(a):
    ex = 'abs(cos(a))**0.5+sin(2+3*a)'
    ne.set_num_threads(16)
    return ne.evaluate(ex)


def perf_comp_data(func_list, data_list, rep=3, number=1):
    from timeit import repeat
    res_list = {}
    for name in enumerate(func_list):
        stmt = name[1] + '(' + data_list[name[0]] + ')'
        setup = "from __main__ import " + name[1] + ', ' + data_list[name[0]]
        results = repeat(stmt=stmt, setup=setup, repeat=rep, number=number)
        res_list[name[1]] = sum(results) / rep
    # res_sort = sorted(res_list.items(), key=lambda item: item[1])
    res_sort = sorted(res_list.items(), key=operator.itemgetter(1))
    for item in res_sort:
        rel = item[1] / res_sort[0][1]
        print('function:' + item[0] + ', avg sec: %9.5f, ' % item[1] + 'relative: %6.1f' % rel)

import numba as nb

f_nb1 = nb.jit(f1)
# f_nb2 = nb.jit(f2)
# f_nb3 = nb.jit(f3)

f_nb4 = nb.jit(f4)

func_list = ['f1',
             'f2', 'f3',
             'f4',
             'f5', 'f6',
             'f_nb1','f_nb4',
             ]
data_list = ['a_py',
             'a_py', 'a_py',
             'a_np',
             'a_np', 'a_np',
                'a_np', 'a_np',
             ]
perf_comp_data(func_list, data_list)



# func_list = [
#              'f1', 'f4',
#              'f_nb4']
# data_list = ['a_py',
#              'a_py', 'a_np',
#              'a_np']
# perf_comp_data(func_list, data_list)
