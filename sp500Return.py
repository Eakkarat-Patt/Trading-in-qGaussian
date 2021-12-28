import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
import pandas as pd
import StockModels


numPaths = 10000
dt = 0.001
t0 = 1e-20
T = 0.6
numSteps = int(T / dt)
r = 0.01
sigma = 0.05
S0 = 10
q = 1.3
# #
# #
w1 = StockModels.WienerProcess()
w1.generateWiener(numPaths, numSteps, t0, T)
# # #
p1 = StockModels.GeometricBrownianMotion(w1)
p1.generateStockPath(r, sigma, S0)
#
# p2 = GeneralizedBrownianMotion(w1)
# p2.generateStockPath(r, sigma, S0, 1.011)
#
# p3 = GeneralizedBrownianMotion(w1)
# p3.generateStockPath(r, sigma, S0, 1.2)
#
p4 = StockModels.GeneralizedBrownianMotion(w1)
p4.generateStockPath(r, sigma, S0, q)


def spReturn(start, end=None):
    df = pd.read_csv('Data/spx_d.csv')[['Date', 'Close']]
    df = df.set_index('Date')
    if end == None:
        df = df.loc[start:]
    else:
        df = df.loc[start:end]
    df['Log return'] = np.log(df['Close']/df['Close'].shift(1))
    mu = df['Log return'].mean()
    sigma = df['Log return'].std()
    df['Log return'] = (df['Log return']-mu)/sigma  # Standardization
    print(df['Log return'].mean())
    print(df['Log return'].var())
    # x = np.linspace(np.min(df['Log return']), np.max(df['Log return']), 10000)
    # normalFit = norm.pdf(x, 0, 0.9)
    plt.figure(figsize=(8, 5), dpi=400)
    # plt.plot(x, normalFit, color='r', label='Gaussian dist: mu = {}, sigma = {}'.format(
    #     round(normalFit.mean(), 2), round(normalFit.std(),2)))
    sns.kdeplot(p4.GetOmg()[:, -1], color='r', log_scale=(False, True), label='Tsallis dist: $\mu = {}, \sigma = {}, q = {}$'.format(
        round(p4.GetOmg()[:, -1].mean(), 2), round(p4.GetOmg()[:, -1].std(), 2), p4.GetEntropyIndex()))
    sns.histplot(df['Log return'], stat='density', bins=80, binrange=[-5, 5], label='S&P500 index'
                 , log_scale=(False, True))
    plt.xlim(-4, 4)
    plt.ylim(10e-5, 10e-1)
    plt.xlabel('Daily log return')
    plt.legend()
    plt.show()


# spReturn('1993-08-10', '2001-07-11')

def spReturnTest(date,logScale = False):
    df = pd.read_csv('Data/sp500data.csv')
    tradingDay = df.loc[df.Date == date, :]
    tradingDay['Log return'] = np.log(tradingDay['Close']/tradingDay['Close'].shift(1))
    mu = tradingDay['Log return'].mean()
    std = tradingDay['Log return'].std()
    tradingDay['Log return'] = (tradingDay['Log return']-mu)/std  # Standardization
    print(tradingDay.head())
    print(tradingDay['Log return'].mean())
    print(tradingDay['Log return'].var())
    # x = np.linspace(np.min(df['Log return']), np.max(df['Log return']), 10000)
    # normalFit = norm.pdf(x, 0, 0.9)
    plt.figure(figsize=(8, 5), dpi=400)
    # plt.plot(x, normalFit, color='r', label='Gaussian dist: mu = {}, sigma = {}'.format(
    #     round(normalFit.mean(), 2), round(normalFit.std(),2)))
    sns.kdeplot(p4.GetOmg()[:, -1], color='r', log_scale=(False, logScale), label='Tsallis dist: $\mu = {}, \sigma = {}, q = {}$'.format(
        round(p4.GetOmg()[:, -1].mean(), 2), round(p4.GetOmg()[:, -1].std(), 2), p4.GetEntropyIndex()))
    sns.histplot(tradingDay['Log return'], stat='density', bins=80, binrange=[-5, 5], label='S&P500 index'
                 , log_scale=(False, logScale))
    plt.title('Date: ' + date)
    plt.xlim(-4, 4)
    plt.ylim(10e-5, 10e-1)
    plt.xlabel('Minutely log return')
    plt.legend()
    plt.show()


def spReturnTest2(start, stop, logScale = False):
    df = pd.read_csv('Data/sp500data.csv')
    df.set_index(['Date', 'Time'], inplace=True)
    tradingDay = df.loc[start:stop]
    tradingDay['Log return'] = np.log(tradingDay['Close']/tradingDay['Close'].shift(1))
    mu = tradingDay['Log return'].mean()
    std = tradingDay['Log return'].std()
    tradingDay['Log return'] = (tradingDay['Log return']-mu)/std  # Standardization
    print(tradingDay.head())
    print(tradingDay['Log return'].mean())
    print(tradingDay['Log return'].var())
    x = np.linspace(np.min(tradingDay['Log return']), np.max(tradingDay['Log return']), 10000)
    normalFit = norm.pdf(x, 0, 0.5)
    plt.figure(figsize=(8, 5), dpi=400)
    plt.plot(x, normalFit, color='r', label='Gaussian dist: mu = {}, sigma = {}'.format(
        round(normalFit.mean(), 2), round(normalFit.std(), 2)))
    sns.histplot(tradingDay['Log return'], stat='density', bins=80, binrange=[-5, 5], label='S&P500 index'
                 , log_scale=(False, logScale))
    plt.title('Minutely log return from' + start + 'to' + stop)
    plt.xlim(-4, 4)
    plt.ylim(10e-5, 10e-1)
    plt.xlabel('Minutely log return')
    plt.legend()
    plt.show()


spReturnTest2('2015-01-02', '2020-12-31')
