#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2018/5/16 15:21
@annotation = ''
"""
import numpy as np
from sklearn.datasets import make_swiss_roll

np.random.seed(4)
m = 60
w1, w2 = 0.1, 0.3
noise = 0.1

angles = np.random.rand(m) * 3 * np.pi / 2 - 0.5
X = np.empty((m, 3))
X[:, 0] = np.cos(angles) + np.sin(angles) / 2 + noise * np.random.randn(m) / 2
X[:, 1] = np.sin(angles) * 0.7 + noise * np.random.randn(m) / 2
X[:, 2] = X[:, 0] * w1 + X[:, 1] * w2 + noise * np.random.randn(m)

from sklearn.decomposition import PCA

if False:
    pca = PCA(n_components=2)
    X2D = pca.fit_transform(X)
    # 特征空间中的主轴
    print(pca.components_)
    # 轴的差异比率
    print(pca.explained_variance_ratio_)
    print(1 - pca.explained_variance_.sum())
    # 解压缩 有损
    pca.inverse_transform(X2D)
"""
find good Dimensions
"""
if False:
    pca = PCA()
    pca.fit(X)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    d = np.argmax(cumsum >= 0.95) + 1
    print(d)

    pca = PCA(n_components=0.95)
    X_reduced = pca.fit_transform(X)
    print(pca.n_components_)
"""
增量pca 节省内存 慢
"""
if False:
    from sklearn.decomposition import IncrementalPCA

    n_batches = 5
    inc_pca = IncrementalPCA(n_components=2)
    for X_batch in np.array_split(X, n_batches):
        inc_pca.partial_fit(X_batch)
    X_reduced = inc_pca.transform(X)

    filename = 'x.data'
    X_mm = np.memmap(filename, dtype='float32', mode='write', shape=X.shape)
    X_mm[:] = X
    del X_mm
    X_mm = np.memmap(filename, dtype="float32", mode="readonly", shape=X.shape)
    batch_size = m // n_batches
    inc_pca = IncrementalPCA(n_components=2, batch_size=batch_size)
    inc_pca.fit(X_mm)
    X_reduced2 = inc_pca.transform(X)

if False:
    X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)
    y = t > 6.9
    from sklearn.decomposition import KernelPCA

    rbf_pca = KernelPCA(n_components=2, kernel="rbf", gamma=0.04)
    X_reduced = rbf_pca.fit_transform(X)

    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    clf = Pipeline([
        ("kpca", KernelPCA(n_components=2)),
        ("log_reg", LogisticRegression())
    ])

    param_grid = [{
        "kpca__gamma": np.linspace(0.03, 0.05, 10),
        "kpca__kernel": ["rbf", "sigmoid"]
    }]

    grid_search = GridSearchCV(clf, param_grid, cv=3)
    grid_search.fit(X, y)
    print(grid_search.best_estimator_)
    print(grid_search.best_params_)

if False:
    from sklearn.manifold import LocallyLinearEmbedding
    from matplotlib import pyplot as plt

    X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=41)
    lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=42)
    X_reduced = lle.fit_transform(X)
    plt.title("Unrolled swiss roll using LLE", fontsize=14)
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t, cmap=plt.cm.hot)
    plt.xlabel("$z_1$", fontsize=18)
    plt.ylabel("$z_2$", fontsize=18)
    plt.axis([-0.065, 0.055, -0.1, 0.12])
    plt.grid(True)

    plt.show()
