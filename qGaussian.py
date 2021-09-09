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
    def __init__(self, name):
        self.name = name
        self.paths = {}

    def getProcessName(self):
        return self.name

    def getnumPaths(self):
        return self.paths['W'].shape[0]

    def getnumSteps(self):
        return self.paths['W'].shape[1]

    def getTime(self):
        return self.paths['t']

    def getW(self):
        return self.paths['W']

    def generateWiener(self, numPaths, numSteps, T):
        # np.random.seed(9)
        t = np.linspace(0, T, numSteps)
        t[0] = 1e-20  # set initial t close to 0 to avoid ZeroDivisionError
        N = np.random.normal(0.0, 1.0, [numPaths, numSteps])
        W = np.zeros([numPaths, numSteps])
        for i in range(1, numSteps):
            if numPaths > 1:  # making sure that samples drawn from random.normal have mean 0 and variance 1
                N[:, i - 1] = (N[:, i - 1] - np.mean(N[:, i - 1])) / np.std(N[:, i - 1])
            W[:, i] = W[:, i - 1] + np.power(t[i]-t[i-1], 0.5) * N[:, i - 1]
        self.paths['t'] = t
        self.paths['W'] = W


class QGaussianProcess(WienerProcess):
    def __init__(self, name):
        WienerProcess.__init__(self, name)

    def getc(self):
        return self.paths['c']

    def getB(self):
        return self.paths['B']

    def getZ(self):
        return self.paths['Z']

    def getOmg(self):
        return self.paths['Omg']

    def generateOmega(self, q):
        c = (np.pi * gamma(1 / (q - 1) - 0.5) ** 2) / ((q - 1) * gamma(1 / (q - 1)) ** 2)
        B = c ** ((1 - q) / (3 - q)) * ((2 - q) * (3 - q) * self.getTime()) ** (-2 / (3 - q))
        Z = ((2 - q) * (3 - q) * c * self.getTime()) ** (1 / (3 - q))
        Omg = np.zeros([self.getnumPaths(), self.getnumSteps()])
        for i in range(1, self.getnumSteps()):
            Omg[:, i] = Omg[:, i - 1] + ((1 - B[i - 1] * (1 - q) * Omg[:, i - 1] ** 2) ** 0.5 * (self.getW()[:, i] -
                                                        self.getW()[:, i - 1])) / (Z[i - 1] ** ((1 - q) / 2))
        self.paths['c'] = c
        self.paths['B'] = B
        self.paths['Z'] = Z
        self.paths['Omg'] = Omg


class ArithmeticBrownianMotion(WienerProcess):
    def __init__(self, name):
        WienerProcess.__init__(self, name)
        self.name = name

    def getS(self):
        return self.paths['S']

    def generateStockPath(self, r, sigma, S0):
        S = np.zeros([self.getnumPaths(), self.getnumSteps()])
        S[:, 0] = S0
        for i in range(1, self.getnumSteps()):
            S[:, i] = S[:, i-1] + r * (self.getTime()[i]-self.getTime()[i-1]) + sigma * (self.getW()[:, i] - self.getW()[:, i-1])
        self.paths['S'] = S


class GeometricBrownianMotion(WienerProcess):
    def __init__(self, name):
        WienerProcess.__init__(self, name)
        self.name = name

    def getS(self):
        return self.paths['S']

    def generateStockPath(self, r, sigma, S0):
        S = np.zeros([self.getnumPaths(), self.getnumSteps()])
        S[:, 0] = S0
        for i in range(1, self.getnumSteps()):
            S[:, i] = S[:, i - 1] + r * S[:, i - 1] * (self.getTime()[i]-self.getTime()[i-1]) + sigma * S[:, i - 1] * \
                      (self.getW()[:, i] - self.getW()[:, i - 1])
        self.paths['S'] = S


class NonGaussianBrownianMotion(QGaussianProcess):
    def __init__(self, name):
        QGaussianProcess.__init__(self, name)
        self.name = name

    def getS(self):
        return self.paths['S']

    def generateStockPath(self, r, sigma, S0, q):
        S = np.zeros([self.getnumPaths(), self.getnumSteps()])
        S[:, 0] = S0
        Pq = ((1 - self.getB() * (1 - q) * self.getOmg() ** 2) ** (1 / (1 - q))) / self.getZ()
        for i in range(1, self.getnumSteps()):
            S[:, i] = S[:, i - 1] + S[:, i - 1] * (self.getTime()[i]-self.getTime()[i-1]) * \
                      (r + 0.5 * sigma ** 2 * Pq[:, i - 1] ** (1 - q)) + \
                      sigma * S[:, i - 1] * (self.getOmg()[:, i] - self.getOmg()[:, i - 1])
            # S[:, i] = S[:, i - 1] + S[:, i - 1] * r * (self.getTime()[i] - self.getTime()[i - 1]) + sigma * \
            #           S[:, i - 1] * (self.getOmg()[:, i] - self.getOmg()[:, i - 1])
        self.paths['S'] = S


numPaths = 10000
dt = 0.0001
T = 1
numSteps = int(T / dt)
r = 0.05
sigma = 0.2
S0 = 50
q = 1.5

# p1 = GeometricBrownianMotion('Geometric Brownian motion')
p2 = NonGaussianBrownianMotion('qGaussian Process')
#
# p1.generateWiener(numPaths, numSteps, T)
p2.generateWiener(numPaths, numSteps, T)
p2.generateOmega(q)
#
# p1.generateStockPath(r, sigma, S0)
p2.generateStockPath(r, sigma, S0, q)



def logReturn(func):
    df = pd.DataFrame({'time': func.getTime(),
                       'stock price': func.getS()[0, :]})
    df['daily log return'] = np.log(df['stock price'] / df['stock price'].shift(1))
    df['daily log return'] = (df['daily log return'] - df['daily log return'].mean()) / df['daily log return'].std()  # standardization
    return df['daily log return']


def distPlot(func1, func2):
    plt.figure(figsize=(8, 5), dpi=500)
    sns.histplot(func1, binwidth=0.1, color='r', binrange=[-10, 10], label='Tsallis $q = {}$'.format(q), stat='density', log_scale=(False, False))
    sns.histplot(func2, binwidth=0.1, binrange=[-10, 10],label='Gaussian', stat='density', log_scale=(False, False))
    plt.xlim(-10, 10)
    plt.legend()
    plt.title('Tsallis Distribution')
    plt.show()


#distPlot(logReturn(p2), logReturn(p1))
#distPlot(p2.getOmg()[:,-1], p2.getW()[:,-1])

def pathPlot(x, y, numPaths=20):
    plt.figure(figsize=(8, 5), dpi=500)
    for i in range(numPaths):
        plt.plot(x, y[i, :])
    plt.title('Stock price path')
    plt.ylabel('Price')
    plt.xlabel('Time')
    plt.show()


# pathPlot(p1.getTime(), p1.getS(), numPaths=1)
# pathPlot(p2.getTime(), p2.getS(), numPaths=1)
# distPlot(logReturn(p2), logReturn(p1))
# distPlot(p2.getOmg()[:,-1], p2.getW()[:,-1])



def varPlot(func, q):
    exact = TsallisVar(q, func.getTime())
    emp = varOmg(func)
    err = (exact - emp) * 100 / exact
    plt.figure(figsize=(8, 5), dpi=500)
    plt.plot(func.getTime(), err)
    plt.title('Variance percentage error')
    plt.xlabel('Time')
    plt.ylabel('Error')
    plt.show()


def TsallisVar(q, t):
    c = (np.pi * gamma(1 / (q - 1) - 0.5) ** 2) / ((q - 1) * gamma(1 / (q - 1)) ** 2)
    B = c ** ((1 - q) / (3 - q)) * ((2 - q) * (3 - q) * t) ** (-2 / (3 - q))
    return 1 / ((5 - 3 * q) * B)


def varOmg(func):

    var_t = np.zeros([func.getnumSteps()])
    for i in range(1, func.getnumSteps()):
        var_t[i] = func.getOmg()[:, i].var()
    return var_t