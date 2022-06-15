"""
This source code is modified from the lecture on Computational Finance taught by Dr.Lech A. Grzelak
"""


import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
# plt.rc('text', usetex=True)

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
        # t[0] = 1e-20  # set initial t close to 0 to avoid ZeroDivisionError
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
        self.Y = np.zeros([noise.getW().shape[0], noise.getW().shape[1]])  # return
        self.W = noise.getW()
        self.t = noise.getTime()

    def GetS(self):
        return self.S

    def GetY(self):
        return self.Y

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
            self.Y[:, i] = self.Y[:, i - 1] + r * (self.GetTime()[i] - self.GetTime()[i - 1]) + sigma * \
                           (self.GetW()[:, i] - self.GetW()[:, i-1])


class GeneralizedBrownianMotion(StockPricesModel):
    def __init__(self, noise):
        StockPricesModel.__init__(self, noise)
        self.c = 0
        self.q = 0
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

    def GetEntropyIndex(self):
        return self.q

    def generateStockPath(self, r, sigma, S0, q):
        self.q = q
        self.c = (np.pi * gamma(1 / (q - 1) - 0.5) ** 2) / ((q - 1) * gamma(1 / (q - 1)) ** 2)
        self.B = self.c ** ((1 - q) / (3 - q)) * ((2 - q) * (3 - q) * self.GetTime()) ** (-2 / (3 - q))
        self.Z = ((2 - q) * (3 - q) * self.c * self.GetTime()) ** (1 / (3 - q))
        self.Pq[:, 0] = ((1 - self.GetB()[0] * (1 - q) * self.GetOmg()[:, 0] ** 2) ** (1 / (1 - q))) / self.GetZ()[0]
        self.S[:, 0] = S0

        for i in range(1, self.GetNumSteps()):

            self.Omg[:, i] = self.Omg[:, i - 1] + self.GetPq()[:, i-1]**((1-q)/2) * (self.GetW()[:, i] - self.GetW()[:, i - 1])
            self.Pq[:, i] = ((1 - self.GetB()[i] * (1 - q) * self.GetOmg()[:, i] ** 2) ** (1 / (1 - q))) / self.GetZ()[i]
            self.S[:, i] = self.S[:, i - 1] + (r + (sigma**2 / 2) * self.GetPq()[:, i]**(1-self.GetEntropyIndex())) * \
                           self.S[:, i - 1] * (self.GetTime()[i]-self.GetTime()[i-1])\
                           + (sigma/1.4678) * self.S[:, i - 1] * (self.GetOmg()[:, i] - self.GetOmg()[:, i - 1])
            self.Y[:, i] = self.Y[:, i - 1] + r * (self.GetTime()[i]-self.GetTime()[i-1]) + (sigma/1.4678) * \
                           (self.GetOmg()[:, i] - self.GetOmg()[:, i - 1])



#1.4678
# numPaths = 1000
# dt = 0.005
# t0 = 1e-20
# T = 10
# numSteps = int(T / dt)
# r1 = 0.005
# sigma1 = 0.01
# r2 = 0.01
# sigma2 = 0.1
# S0 = 10
# q = 1.56
# #
# #
# w1 = WienerProcess()
# w1.generateWiener(numPaths, numSteps, t0, T)
# # #
# p1 = GeometricBrownianMotion(w1)
# p1.generateStockPath(r1, sigma1, S0)
# #
# p2 = GeneralizedBrownianMotion(w1)
# p2.generateStockPath(r2, sigma2, S0, q)
#
# p3 = GeneralizedBrownianMotion(w1)
# p3.generateStockPath(r, sigma, S0, 1.2)
#
# p4 = GeneralizedBrownianMotion(w1)
# p4.generateStockPath(r, sigma, S0, 1.4)

# p5 = GeneralizedBrownianMotion(w1)
# p5.generateStockPath(r2, sigma2, S0, q)


def LogReturn(func1):
    df = pd.DataFrame({'time': func1.GetTime(),
                       'stock price': func1.GetS()[0, :]})
    df['daily log return'] = np.log(df['stock price'] / df['stock price'].shift(1))
    #df['daily log return'] = (df['daily log return'] - df['daily log return'].mean()) / df['daily log return'].std()  # standardization
    return df['daily log return']

def DistPlot(func1, logScale=False):
    plt.figure(figsize=(8, 5), dpi=500)
    sns.histplot(func1, label='Gaussian', log_scale=(False, logScale))
    plt.legend()
    plt.title('Terminal Time Stock Price Distribution')
    plt.show()



def CompareDistPlot(func1, func2, logScale=False):
    plt.figure(figsize=(8, 5), dpi=500)
    sns.histplot(func1.GetS()[:, -1], binwidth=0.01, binrange=[min(func2.GetS()[:, -1]), max(func2.GetS()[:, -1])],
                 color='r', stat='density', label='Tsallis q = {}'.format(func1.GetEntropyIndex()), log_scale=(False, logScale))
    sns.histplot(func2.GetS()[:, -1], binwidth=0.01, binrange=[min(func2.GetS()[:, -1]), max(func2.GetS()[:, -1])],
                 stat='density', label='Gaussian', log_scale=(False, logScale))
    plt.legend()
    plt.title('Terminal Time Stock Price Distribution')
    plt.show()


def ReturnDistributionPlot(func1, logScale=False):
    df = pd.DataFrame({'time': func1.GetTime(),
                       'stock price': func1.GetS()[0, :]})
    df['daily log return'] = np.log(df['stock price'] / df['stock price'].shift(1))
    plt.figure(figsize=(8, 5), dpi=500)
    sns.histplot(df['daily log return'], binwidth=0.0001, stat='density', color='b', log_scale=(False, logScale))
    plt.legend()
    plt.title('Log return distribution')
    plt.show()


def PathPlot(x, y1, numPaths=20):
    '''
    plot stock path
    '''
    plt.figure(figsize=(8, 5), dpi=500)
    for i in range(numPaths):
        plt.plot(x, y1[i, :])
    plt.title('Stock price paths')
    plt.ylabel('Price')
    plt.xlabel('Time')
    plt.show()


def TimeSeriesPlot(x, y1, y2, pathNum, stop, legend=True, ylabel=None, label1=None, label2=None):
    plt.figure(figsize=(8, 5), dpi=500)
    plt.plot(x[x <= stop], y1[pathNum, x <= stop], label=label1)
    plt.plot(x[x <= stop], y2[pathNum, x <= stop], label=label2)
    plt.xlim([0, stop])
    # plt.ylim([1.06, 1.11])
    plt.ylabel(ylabel)
    plt.xlabel('Time')
    if legend:
        plt.legend()
    plt.show()
