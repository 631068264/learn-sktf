#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2018/6/27 17:28
@annotation = ''
"""
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer,TfidfVectorizer

vectorizer = CountVectorizer()
corpus = [
    'This is the first document.',
    'This is the second second document.',
    'And the third one.',
    'Is this the first document?',
]
y = vectorizer.fit_transform(corpus)
# 词序
print(vectorizer.vocabulary_)
print()
feature_name = vectorizer.get_feature_names()
print(y)
print(feature_name)

print(len(feature_name))
print(y.toarray())
print(y.shape)

q = """
Tifidf
"""
print(q)

tfidf = TfidfVectorizer()
y = tfidf.fit_transform(corpus)
# 词序
print(vectorizer.vocabulary_)
print()
feature_name = vectorizer.get_feature_names()
print(y)
print(feature_name)

print(len(feature_name))
print(y.toarray())
print(y.shape)