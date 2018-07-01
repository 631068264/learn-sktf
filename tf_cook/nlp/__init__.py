#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2018/6/25 14:32
@annotation = ''
"""
from collections import Counter

"""
对于大文本，我们可以选择要保留多少个单词，并且通常保留最常用的单词，
用零的索引标记其他所有内容。假设数字代表种类不是数字关系

两个句子的长度不同 希望创建的模型具有相同的大小输入  
将每个句子创建为一个稀疏向量，如果该词出现在该索引中，则该词在特定索引中的值为1

这些向量的长度等于我们选择的词汇的大小  选择一个非常大的词汇表很常见，所以这些句子可能非常稀疏


"""
c = Counter('aaaaaaabracadabrabb')

print(c)

print(c.most_common(3))
