#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2018/5/15 15:36
@annotation = ''
"""
import numpy as np
from sklearn.datasets import make_moons
from sklearn.ensemble import VotingClassifier, AdaBoostClassifier, RandomForestClassifier, BaggingClassifier, \
    GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import util

X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

"""
vote hard

If ‘hard’, uses predicted class labels for majority rule voting. 
Else if ‘soft’, predicts the class label based on the argmax of the sums of the predicted probabilities, 
which is recommended for an ensemble of well-calibrated classifiers.

"""
if False:
    log_clf = LogisticRegression()
    rnd_clf = RandomForestClassifier()
    svm_clf = SVC()
    voting_clf = VotingClassifier(
        estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
        voting='hard'
    )

    for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

"""
vote soft 比 vote hard
all classifiers are able to estimate class probabilities 
predict_proba()

This is called soft voting. It often achieves higher performance than hard voting 
because it gives more weight to highly confident votes. 
"""
if False:
    log_clf = LogisticRegression(random_state=42)
    rnd_clf = RandomForestClassifier(random_state=42)
    svm_clf = SVC(probability=True, random_state=42)

    voting_clf = VotingClassifier(
        estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
        voting='soft')

    for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

if False:
    """
    BaggingClassifier
    base_estimator 有 predict_proba auto 用 soft vote
    """

    bag_clf = BaggingClassifier(
        DecisionTreeClassifier(random_state=42), n_estimators=500,
        max_samples=100, bootstrap=True, n_jobs=-1, random_state=42)
    bag_clf.fit(X_train, y_train)
    y_pred = bag_clf.predict(X_test)
    print(accuracy_score(y_test, y_pred))

    tree_clf = DecisionTreeClassifier(random_state=42)
    tree_clf.fit(X_train, y_train)
    y_pred_tree = tree_clf.predict(X_test)
    print(accuracy_score(y_test, y_pred_tree))

    """
    out-of-bag (oob) instances 不被抽样 用于检验结果
    """
    bag_clf = BaggingClassifier(
        DecisionTreeClassifier(random_state=42), n_estimators=500,
        bootstrap=True, n_jobs=-1, oob_score=True, random_state=40)
    bag_clf.fit(X_train, y_train)
    print(bag_clf.oob_score_)
    y_pred_tree = tree_clf.predict(X_test)
    print(accuracy_score(y_test, y_pred_tree))

    """
    RandomForest
    Not find best feature to split a node
    But find best feature among a random subset of features 在随机里面找最好
    增加树的多样性 不容易过拟合
    
    rnd_clf 等价 bag_clf
    """
    rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
    bag_clf = BaggingClassifier(
        DecisionTreeClassifier(splitter="random", max_leaf_nodes=16),
        n_estimators=500, max_samples=1.0, bootstrap=True, n_jobs=-1
    )

    """
    feature importance
    """
    from sklearn.datasets import load_iris

    iris = load_iris()
    rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
    rnd_clf.fit(iris["data"], iris["target"])
    for name, score in zip(iris["feature_names"], rnd_clf.feature_importances_):
        print(name, score)

"""
Boost
"""
if False:
    """
    AdaBoost
    """
    ada_clf = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=1), n_estimators=200,
        algorithm="SAMME.R", learning_rate=0.5, random_state=42)
    ada_clf.fit(X_train, y_train)
    util.plot_decision_boundary(ada_clf, X, y)

if True:
    """
    Gradient Boosting
    """
    np.random.seed(42)
    X = np.random.rand(100, 1) - 0.5
    y = 3 * X[:, 0] ** 2 + 0.05 * np.random.randn(100)
    X_new = np.array([[0.8]])

    # tree_reg1 = DecisionTreeRegressor(max_depth=2, random_state=42)
    # tree_reg1.fit(X, y)
    # y2 = y - tree_reg1.predict(X)
    # tree_reg2 = DecisionTreeRegressor(max_depth=2, random_state=42)
    # tree_reg2.fit(X, y2)
    # y3 = y2 - tree_reg2.predict(X)
    # tree_reg3 = DecisionTreeRegressor(max_depth=2, random_state=42)
    # tree_reg3.fit(X, y3)
    # y_pred = sum(tree.predict(X_new) for tree in (tree_reg1, tree_reg2, tree_reg3))
    """
    learning_rate超参数缩放每棵树的贡献 learning_rate few n_estimators more 太多会overfit
    """

    gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=200, learning_rate=0.1, random_state=42)
    gbrt.fit(X, y)
    util.plot_predictions([gbrt], X, y, axes=[-0.5, 0.5, -0.1, 0.8])

    """
    Early stopping find best n_estimators
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=49)

    gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=120, random_state=42)
    gbrt.fit(X_train, y_train)
    util.plot_predictions([gbrt], X, y, axes=[-0.5, 0.5, -0.1, 0.8])

    errors = [mean_squared_error(y_test, y_pred) for y_pred in gbrt.staged_predict(X_test)]
    bst_n_estimators = np.argmin(errors)
    print(bst_n_estimators)
    gbrt_best = GradientBoostingRegressor(max_depth=2, n_estimators=bst_n_estimators, random_state=42)
    gbrt_best.fit(X_train, y_train)
    util.plot_predictions([gbrt_best], X, y, axes=[-0.5, 0.5, -0.1, 0.8])

    """
    提早停止训练来实现提早停止 不是先训练大量的树，然后再回头寻找最佳数目
    setting warm_start=True, which makes Scikit- Learn keep existing trees when the fit() method is called
    subsample 指定用于训练每棵树的随机选择训练实例的比例
    """
    gbrt = GradientBoostingRegressor(max_depth=2, warm_start=True)
    min_val_error = float("inf")
    error_going_up = 0
    for n_estimators in range(1, 120):
        gbrt.n_estimators = n_estimators
        gbrt.fit(X_train, y_train)
        y_pred = gbrt.predict(X_test)
        val_error = mean_squared_error(y_test, y_pred)
        if val_error < min_val_error:
            print(n_estimators, min_val_error)
            min_val_error = val_error
            error_going_up = 0
        else:
            error_going_up += 1
            if error_going_up == 5:
                break  # early stopping
    print(n_estimators)
    print(gbrt.n_estimators)
