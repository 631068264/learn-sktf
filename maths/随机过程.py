#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2018/8/1 23:24
@annotation = ''
"""

import matplotlib.pyplot as plt
import numpy as np

if False:
    S0 = 100  # initial value
    r = 0.05  # constant short rate
    sigma = 0.25  # constant volatility
    T = 2.0  # in years
    I = 10000  # number of random draws

    ST1 = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * np.random.standard_normal(I))
    ST2 = S0 * np.random.lognormal((r - 0.5 * sigma ** 2) * T, sigma * np.sqrt(T), size=I)

    plot = False
    if plot:
        plt.hist(ST1, bins=50)
        plt.hist(ST2, bins=50)
        plt.xlabel('index level')
        plt.ylabel('frequency')
        plt.grid(True)
        plt.show()


def print_statistics(a1, a2):
    import scipy.stats as scs
    """
    比较分布
    """
    sta1 = scs.describe(a1)
    sta2 = scs.describe(a2)
    print('%14s %14s %14s' % ('statistic', 'data set 1', 'data set 2'))
    print(45 * '-')
    print('%14s %14.3f %14.3f' % ('size', sta1[0], sta2[0]))
    print('%14s %14.3f %14.3f' % ('min', sta1[1][0], sta2[1][0]))
    print('%14s %14.3f %14.3f' % ('max', sta1[1][1], sta2[1][1]))
    print('%14s %14.3f %14.3f' % ('mean', sta1[2], sta2[2]))
    print('%14s %14.3f %14.3f' % ('std', np.sqrt(sta1[3]), np.sqrt(sta2[3])))
    print('%14s %14.3f %14.3f' % ('skew', sta1[4], sta2[4]))
    print('%14s %14.3f %14.3f' % ('kurtosis', sta1[5], sta2[5]))


def gen_monte():
    dt = T / M
    S = np.zeros((M + 1, I), np.float64)
    S[0] = S0
    for t in range(1, M + 1):
        rand = np.random.standard_normal(I)
        rand = (rand - rand.mean()) / rand.std()
        S[t] = S[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * rand)


# print_statistics(ST1, ST2)
if False:
    # 随机过程
    T = 2.0  # in years
    I = 10000
    M = 50
    dt = T / M
    S = np.zeros((M + 1, I))
    S[0] = S0
    for t in range(1, M + 1):
        S[t] = S[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * np.random.standard_normal(I))

    # plt.hist(S[-1], bins=50)
    # plt.xlabel('index level')
    # plt.ylabel('frequency')
    # plt.grid(True)

    plt.plot(S[:, :10], lw=1.5)
    plt.xlabel('time')
    plt.ylabel('index level')

    plt.grid(True)

if False:
    # 平方根扩散的随机微分
    x0 = 0.05
    kappa = 3.0
    theta = 0.02
    sigma = 0.1

    T = 2.0
    I = 10000
    M = 50
    dt = T / M


    def srd_euler():
        xh = np.zeros((M + 1, I))
        x1 = np.zeros_like(xh)
        xh[0] = x0
        x1[0] = x0
        for t in range(1, M + 1):
            xh[t] = (xh[t - 1]
                     + kappa * (theta - np.maximum(xh[t - 1], 0)) * dt
                     + sigma * np.sqrt(np.maximum(xh[t - 1], 0)) * np.sqrt(dt)
                     * np.random.standard_normal(I))
        x1 = np.maximum(xh, 0)
        return x1


    x1 = srd_euler()

    plt.hist(x1[-1], bins=50)
    plt.xlabel('value')
    plt.ylabel('frequency')
    plt.grid(True)

if False:
    def gen_sn(M, I, anti_paths=True, mo_math=True):
        """
        Function to generate random numbers for simulation
        :param M: number of time  intervals for discretization
        :param I: number of paths to be simulated
        :param anti_paths: use of antithetic variates
        :param mo_math: use of moment matching
        :return:
        """
        if anti_paths is True:
            sn = np.random.standard_normal((M + 1, int(I / 2)))
            sn = np.concatenate((sn, -sn), axis=1)
        else:
            sn = np.random.standard_normal((M + 1, I))
        if mo_math is True:
            sn = (sn - sn.mean()) / sn.std()
        return sn


    S0 = 100.
    r = 0.05
    sigma = 0.25
    T = 1.0
    I = 50000
    M = 50


    def gbm_mcs_stat(K):
        """
        Valuation of European call option in Black-Scholes-Merton
        by Mont Carlo simulation ( of index level at maturity )
        :param k: float (positive) strike price of the option
        :return:
        """
        sn = gen_sn(1, I)
        # simulate index level at maturity
        ST = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * sn[1])
        # calculate payoff at maturity
        hT = np.maximum(ST - K, 0)
        # calculate MCS estimator
        C0 = np.exp(-r * T) * 1 / I * np.sum(hT)
        return C0


    def gbm_mcs_dyna(K, option='call'):
        """
        Valuation of European option in Black-Scholes-Merton by Monte Carlo simulation(of index level paths)
        :param K: （positive）strike price of the option
        :param option:
        :return:
        """
        dt = T / M
        # simulation of index level paths
        S = np.zeros((M + 1, I))
        S[0] = S0
        sn = gen_sn(M, I)
        for t in range(1, M + 1):
            S[t] = S[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * sn[t])
        # case-based calculation of payoff
        if option == 'call':
            hT = np.maximum(S[-1] - K, 0)
        else:
            hT = np.maximum(K - S[-1], 0)
        # calculation of MCS estimator
        C0 = np.exp(-r * T) * 1 / I * np.sum(hT)
        return C0


    # gbm_mcs_stat(K=105.)
    # gbm_mcs_dyna(K=110., option='call')
    # gbm_mcs_dyna(K=110., option='put')
    from maths.bsm_functions import bsm_call_value

    stat_res = []
    dyna_res = []
    anal_res = []
    k_list = np.arange(80., 120.1, 5.)
    np.random.seed(200000)
    for K in k_list:
        stat_res.append(gbm_mcs_stat(K))
        dyna_res.append(gbm_mcs_dyna(K))
        anal_res.append(bsm_call_value(S0, K, T, r, sigma))

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    ax1.plot(k_list, anal_res, 'b', label='analytical')
    ax1.plot(k_list, stat_res, 'ro', label='static')
    ax1.set_ylabel('European call option value')
    ax1.grid(True)
    ax1.legend(loc=0)
    ax1.set_ylim(ymin=0)
    wi = 1.0
    ax2.bar(k_list - wi / 2, (np.array(anal_res) - np.array(stat_res)) / np.array(anal_res) * 100, wi)
    ax2.set_xlabel('strike')
    ax2.set_ylabel('difference in %')
    ax2.set_xlim(left=75, right=125)
    ax2.grid(True)

plt.show()


def rand_w(split=5):
    w = np.random.random(split)
    w /= np.sum(w)
    return w


rand_w()
