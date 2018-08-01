#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2018/8/1 18:03
@annotation = ''
"""
import sympy as sy

x = sy.Symbol('x')
y = sy.Symbol('y')

if False:
    sy.sqrt(x)
    # sqrt(x)

    3 + sy.sqrt(x) - 4 ** 2
    # sqrt(x) - 13

    f = x ** 2 + 3 + 0.5 * x ** 2 + 3 / 2
    s = sy.simplify(f)
    print(s)
    # 求解
    s = sy.solve(f)
    print(s)

    s = sy.solve(x ** 2 + y ** 2)
    print(s)

"""
f(x) => f(0.5)
f.subs(x, 0.5).evalf()

f(x,y) => f(xo,yo)
f.subs({x : xo, y : yo}).evalf() 
"""
# 积分
a, b = sy.symbols('a b')
print(sy.pretty(sy.Integral(sy.sin(x) + 0.5 * x, (x, a, b))))
f = sy.sin(x) + 0.5 * x

print(f.integrate((x, 0.5, 9.5)))
print(f.integrate().subs(x, 0.5, 9.5).evalf())
s = sy.integrate(f, (x, 0.5, 9.5))
print(s)

# 反导数
int_func = sy.integrate(sy.sin(x) + 0.5 * x, x)
print(sy.pretty(int_func))

# 导数 微分
f = x ** 2 + 3 + 0.5 * x ** 2 + 3 / 2
print(sy.pretty(f.diff()))

# 偏微分
f = sy.sin(x) + 0.05 * x ** 2 + sy.sin(y) + 0.05 * y ** 2
del_x = f.diff(x)
del_y = f.diff(y)
print(del_x)
print(del_y)
