#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2018/6/27 17:28
@annotation = ''
"""
import os

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from keras import Sequential
from keras.layers import Dense, Dropout
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

corpus = [
    'This is the first document.',
    'This is the second second document.',
    'And the third one.',
    'Is this the first document?',
]

if False:
    vectorizer = CountVectorizer()
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
if False:
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

if False:
    q = """
    Word2Vec 先分词
    """
    print(q)

    word = [['first', 'sentence'], ['second', 'sentence']]
    model = Word2Vec(word, min_count=1)
    print(model)
    vac = model.vocabulary
    print(vac)
    print(model.corpus_count)
    print(model['first'])
    print(model.most_similar('sentence'))


    # model = Doc2Vec(corpus,size=50, min_count=2, iter=10)
    # print(model['This'])

    def train_word2vec(filename, word2vec_file):
        # 模型文件不存在才处理
        if not os.path.exists(word2vec_file):
            sentences = LineSentence(filename)
            # sg=0 使用cbow训练, sg=1对低频词较为敏感
            model = Word2Vec(sentences,
                             size=300, window=5, min_count=2, sg=1, workers=4)
            model.save(word2vec_file)


    word2vec_file = 'temp/w2v.bin'
    train_word2vec('temp/news_sohusite_cutword.txt', word2vec_file)
    model = Word2Vec.load(word2vec_file)
    print(model.most_similar('健康'))
    print(model.similarity('健康', '血压'))
    print(model.similarity('血压', '健康'))
    print(model.similarity('血压', '血压'))
    print(model.similarity('血压', '牛奶'))


def baseline_model(max_features):
    model = Sequential()
    model.add(Dense(5, input_dim=max_features, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
