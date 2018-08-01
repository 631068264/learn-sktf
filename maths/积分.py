#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2018/8/1 17:50
@annotation = ''
"""
import numpy as np


def f(x):
    return np.sin(x) + 0.5 * x


import scipy.integrate as sci

a = 0.5
b = 9.5

print(sci.fixed_quad(f, a, b)[0])
# 24.366995967084602
print(sci.quad(f, a, b)[0])
# 24.374754718086752
print(sci.romberg(f, a, b))
# 24.374754718086713


xi = np.linspace(0.5, 9.5, 25)
sci.trapz(f(xi), xi)
# 24.352733271544516
sci.simps(f(xi), xi)
# 24.374964184550748

if False:
    # 我们感兴趣的是［0.5, 9.5]区间内的积分
    a = 0.5
    b = 9.5
    x = np.linspace(0, 10)
    y = f(x)

    from matplotlib.patches import Polygon

    fig, ax = plt.subplots(figsize=(7, 5))
    plt.plot(x, y, 'b', linewidth=2)
    plt.ylim(ymin=0)

    # area unde the function
    # between lower and upper limit
    Ix = np.linspace(a, b)
    Iy = f(Ix)
    verts = [(a, 0)] + list(zip(Ix, Iy)) + [(b, 0)]
    poly = Polygon(verts, facecolor='0.7', edgecolor='0.5')
    ax.add_patch(poly)

    # labels
    plt.text(0.75 * (a + b), 1.5, r"$\int_a^b f(x)dx$", horizontalalignment='center', fontsize=20)
    plt.figtext(0.9, 0.075, '$x$')
    plt.figtext(0.075, 0.9, '$f(x)$')

    ax.set_xticks((a, b))
    ax.set_xticklabels(('$a$', '$b$'))
    ax.set_yticks([f(a), f(b)])
    plt.show()
