#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2018/5/17 10:39
@annotation = ''
"""
import tensorflow as tf

import util

"""
TensorFlow程序通常分为两部分
构建 built ML model and train it
执行 evaluate a training step repeatedly and improve the model parameters

TensorFlow操作（也称为操作简称）可以接受任意数量的输入并生成任意数量的输出
输入和输出是多维数组，称为张量tensor
"""

if False:
    x = tf.Variable(3, name="x")
    y = tf.Variable(4, name="y")
    f = x * x * y + y + 2

    # with tf.Session() as sess:
    #     sess.run(x.initializer)
    #     sess.run(y.initializer)
    #     result = sess.run(f)
    #     print(result)

    # 初始化变量
    init_var = tf.global_variables_initializer()
    with tf.Session() as sess:
        # x.initializer.run()
        # y.initializer.run()
        init_var.run()
        result = f.eval()
        print(result)

"""
Managing Graphs

Any node you create is automatically added to the default graph
"""
if False:
    util.reset_graph()
    x1 = tf.Variable(1)
    print(x1.graph is tf.get_default_graph())

    # 新开一图 creating a new Graph and temporarily making it the default graph inside a with block
    graph = tf.Graph()
    with graph.as_default():
        x2 = tf.Variable(2)

    print(x2.graph is graph)
    print(x2.graph is tf.get_default_graph())

    w = tf.constant(3)
    x = w + 2
    y = x + 5
    z = x * 3
if False:
    with tf.Session() as sess:
        """
        TensorFlow会自动检测到y依赖于w，这取决于x，所以它首先评估w，然后x，然后y，并返回y的值。
        最后，代码运行图来评估z。 再一次检测到它必须首先评估w和x。
        重要的是要注意，它不会重用先前评估w和x的结果。总之，前面的代码评估w和x两次。
        所有节点值都在图运行之间被删除，除了变量值
        """
        print(y.eval())
        print(z.eval())

    with tf.Session() as sess:
        """
        想要高效地评估y和z，而不需要像前面的代码那样对w和x进行两次评估 ,在一次图形运行中评估y和z
        
        单进程TensorFlow中,每个会话都有自己的每个变量副本 多个会话不共享任何状态，即使它们重复使用同一个图
        """
        y_val, z_val = sess.run([y, z])
        print(y_val)  # 10
        print(z_val)  # 15

"""
占位符

它们通常用于在训练期间将训练数据传递给TensorFlow
实际上不执行任何计算
"""
if False:
    A = tf.placeholder(tf.float32, shape=(None, 3))
    B = A + 5
    with tf.Session() as sess:
        B_val_1 = B.eval(feed_dict={A: [[1, 2, 3]]})
        B_val_2 = B.eval(feed_dict={A: [[4, 5, 6], [7, 8, 9]]})

    print(B_val_1)
