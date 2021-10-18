"""
This source code is modified from the lecture on Computational Finance taught by Dr.Lech A. Grzelak
"""


import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
plt.rc('text', usetex=True)

class WienerProcess(object):
    def __init__(self):
        self.paths = {}

    def getnumPaths(self):
        return self.paths['W'].shape[0]

    def getnumSteps(self):
        return self.paths['W'].shape[1]

    def getTime(self):
        return self.paths['t']

    def getW(self):
        return self.paths['W']

    def generateWiener(self, numPaths, numSteps, t0, T):
        t = np.linspace(t0, T, numSteps)
        #t[0] = 1e-20  # set initial t close to 0 to avoid ZeroDivisionError
        N = np.random.normal(0.0, 1.0, [numPaths, numSteps])
        W = np.zeros([numPaths, numSteps])
        for i in range(1, numSteps):
            if numPaths > 1:  # making sure that samples drawn from random.normal have mean 0 and variance 1
                N[:, i - 1] = (N[:, i - 1] - np.mean(N[:, i - 1])) / np.std(N[:, i - 1])
            W[:, i] = W[:, i - 1] + np.power(t[i]-t[i-1], 0.5) * N[:, i - 1]
        self.paths['t'] = t
        self.paths['W'] = W


class StockPricesModel(object):
    def __init__(self, noise):
        self.S = np.zeros([noise.getW().shape[0], noise.getW().shape[1]])
        self.W = noise.getW()
        self.t = noise.getTime()

    def GetS(self):
        return self.S

    def GetW(self):
        return self.W

    def GetNumPaths(self):
        return self.W.shape[0]

    def GetNumSteps(self):
        return self.W.shape[1]

    def GetTime(self):
        return self.t


class ArithmeticBrownianMotion(StockPricesModel):
    def __init__(self, noise):
        StockPricesModel.__init__(self, noise)

    def generateStockPath(self, r, sigma, S0):
        self.S[:, 0] = S0
        for i in range(1, self.GetNumSteps()):
            self.S[:, i] = self.S[:, i-1] + r * (self.GetTime()[i]-self.GetTime()[i-1])\
                           + sigma * (self.GetW()[:, i] - self.GetW()[:, i-1])


class GeometricBrownianMotion(StockPricesModel):
    def __init__(self, noise):
        StockPricesModel.__init__(self, noise)

    def generateStockPath(self, r, sigma, S0):
        self.S[:, 0] = S0
        for i in range(1, self.GetNumSteps()):
            self.S[:, i] = self.S[:, i - 1] * np.exp((r - 0.5 * sigma**2)*(self.GetTime()[i]-self.GetTime()[i-1]) +
                                                     sigma * (self.GetW()[:, i] - self.GetW()[:, i-1]))


class GeneralizedBrownianMotion(StockPricesModel):
    def __init__(self, noise):
        StockPricesModel.__init__(self, noise)
        self.c = 0
        self.B = np.zeros([self.GetNumSteps()])
        self.Z = np.zeros([self.GetNumSteps()])
        self.Pq = np.zeros([self.GetNumPaths(), self.GetNumSteps()])
        self.Omg = np.zeros([self.GetNumPaths(), self.GetNumSteps()])

    def Getc(self):
        return self.c

    def GetB(self):
        return self.B

    def GetZ(self):
        return self.Z

    def GetPq(self):
        return self.Pq

    def GetOmg(self):
        return self.Omg

    def generateStockPath(self, r, sigma, S0, q):
        self.c = (np.pi * gamma(1 / (q - 1) - 0.5) ** 2) / ((q - 1) * gamma(1 / (q - 1)) ** 2)
        self.B = self.c ** ((1 - q) / (3 - q)) * ((2 - q) * (3 - q) * self.GetTime()) ** (-2 / (3 - q))
        self.Z = ((2 - q) * (3 - q) * self.c * self.GetTime()) ** (1 / (3 - q))
        self.Pq[:, 0] = ((1 - self.GetB()[0] * (1 - q) * self.GetOmg()[:, 0] ** 2) ** (1 / (1 - q))) / self.GetZ()[0]
        self.S[:, 0] = S0

        for i in range(1, self.GetNumSteps()):
            self.Omg[:, i] = self.Omg[:, i - 1] + ((1 - self.GetB()[i - 1] * (1 - q) * self.Omg[:, i - 1] ** 2) ** 0.5
                                                   * (self.GetW()[:, i] - self.GetW()[:, i - 1]))\
                                                    / (self.GetZ()[i - 1] ** ((1 - q) / 2))

            self.Pq[:, i] = ((1 - self.GetB()[i] * (1 - q) * self.GetOmg()[:, i] ** 2) ** (1 / (1 - q))) / self.GetZ()[i]

            self.S[:, i] = self.S[:, i - 1] + r * self.S[:, i - 1] * (self.GetTime()[i]-self.GetTime()[i-1])\
                           + sigma * self.S[:, i - 1] * (self.GetOmg()[:, i] - self.GetOmg()[:, i - 1])


# numPaths = 100000
# dt = 0.001
# t0 = 1e-20
# T = 1
# numSteps = int(T / dt)
# r = 0.05
# sigma = 0.2
# S0 = 50
# q = 1.3
#
#
# w1 = WienerProcess()
# w1.generateWiener(numPaths, numSteps, t0, T)
#
# p1 = GeometricBrownianMotion(w1)
# p1.generateStockPath(r, sigma, S0)
#
# p2 = GeneralizedBrownianMotion(w1)
# p2.generateStockPath(r, sigma, S0, q)

def logReturn(func1, func2):
    df = pd.DataFrame({'time': func1.GetTime(),
                       'stock price': func1.GetS()[MaxDifference(func1, func2), :]})
    df['daily log return'] = np.log(df['stock price'] / df['stock price'].shift(1))
    df['daily log return'] = (df['daily log return'] - df['daily log return'].mean()) / df['daily log return'].std()  # standardization
    return df['daily log return']


def distPlot(func1, func2, logScale=False):
    plt.figure(figsize=(8, 5), dpi=500)
    sns.histplot(func1, binwidth=0.2, color='r', binrange=[-10, 10], label='Tsallis', stat='density', log_scale=(False, logScale))
    sns.histplot(func2, binwidth=0.2, binrange=[-10, 10], label='Gaussian', stat='density', log_scale=(False, logScale))
    plt.xlim(-10, 10)
    plt.legend()
    plt.title('Tsallis Distribution')
    plt.show()


def pathPlot(x, y1, numPaths=20):
    plt.figure(figsize=(8, 5), dpi=500)
    for i in range(numPaths):
        plt.plot(x, y1[i, :])
    plt.title('Stock price path')
    plt.ylabel('Price')
    plt.xlabel('Time')
    plt.show()


def pathPlot2(x, y1, y2):
    plt.figure(figsize=(8, 5), dpi=500)
    plt.plot(x, y1[0, :], label='GBM')
    plt.plot(x, y2[0, :], label='Generalized GBM q = {}'.format(q))
    plt.xlim([0.0, x[-1]])
    plt.title('Stock price path')
    plt.ylabel('Price')
    plt.xlabel('Time')
    plt.legend()
    plt.show()


def MaxDifference(func1, func2):
    diff = np.zeros([func1.GetS().shape[0]])
    finalS1 = func1.GetS()[:, -1]
    finalS2 = func2.GetS()[:, -1]
    for i in range(func1.GetS().shape[0]):
        diff[i] = abs(finalS1[i]-finalS2[i])
    return np.argmax(diff)
