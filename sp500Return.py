import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
import pandas as pd
import StockModels

numPaths = 5000
dt = 0.005
T = 0.45
numSteps = int(T / dt)

q = 1.6

p2 = StockModels.NonGaussianBrownianMotion('qGaussian Process')
p2.generateWiener(numPaths, numSteps, T)
p2.generateOmega(q)


def spReturn(start, end=None):
    df = pd.read_csv('spx_d.csv')[['Date', 'Close']]
    df = df.set_index('Date')
    if end == None:
        df = df.loc[start:]
    else:
        df = df.loc[start:end]
    df['Log return'] = np.log(df['Close']/df['Close'].shift(1))
    mu = df['Log return'].mean()
    sigma = df['Log return'].std()
    df['Log return'] = (df['Log return']-mu)/sigma  # Standardization
    x = np.linspace(np.min(df['Log return']), np.max(df['Log return']), 10000)
    normalFit = norm.pdf(x, 0, 0.55)
    plt.figure(figsize=(8,5), dpi=400)
    plt.plot(x, normalFit, color='r', label='Gaussian dist: $\mu = {}, \sigma = {}$'.format(
        normalFit.mean(), normalFit.std()))
    sns.kdeplot(p2.getOmg()[:, -1], color='g', label='Tsallis dist: $\mu = {}, \sigma = {}$'.format(
        round(p2.getOmg()[:, -1].mean(), 2), round(p2.getOmg()[:, -1].std(), 2)))
    sns.histplot(df['Log return'], stat='density', binwidth=0.1, binrange=[-5, 5], label='S&P500 index')
    plt.xlim(-5, 5)
    plt.xlabel('Daily log return')
    plt.legend()
    plt.show()

spReturn('2000-01-01', end=None)