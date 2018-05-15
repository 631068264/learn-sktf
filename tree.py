#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2018/5/14 10:40
@annotation = ''
"""

"""
决策树也是多功能的机器学习算法，可以执行分类和回归任务，甚至可以执行多输出任务。

不需要feature scale
"""

from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data[:, 2:]
y = iris.target
if False:
    """
     为避免过度拟合训练数据，您需要在训练期间限制决策树的自由度。正则化超参数取决于所使用的算法，
     但通常您至少可以限制决策树的最大深度。增加min_ *超参数或减少max_ *超参数将规范模型
    """
    from sklearn.tree import DecisionTreeClassifier

    tree_clf = DecisionTreeClassifier(max_depth=2)
    tree_clf.fit(X, y)
    # from sklearn.tree import export_graphviz
    #
    # export_graphviz(
    #     tree_clf,
    #     out_file=util.image_path("iris_tree.dot"),
    #     feature_names=iris.feature_names[2:],
    #     class_names=iris.target_names,
    #     rounded=True,
    #     filled=True
    # )
    print(tree_clf.predict_proba([[5, 1.5]]))
    print(tree_clf.predict([[5, 1.5]]))

if False:
    from sklearn.tree import DecisionTreeRegressor

    tree_reg = DecisionTreeRegressor(max_depth=2)
    tree_reg.fit(X, y)


