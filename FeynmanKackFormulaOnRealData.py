import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import StockModels
from scipy.stats import skew
import pandas as pd


logReturn = np.loadtxt('sp500 minute logR for MM.txt')


def GeneratePrice(R):
    S = np.zeros([R.shape[0], R.shape[1]])
    S[:, 0] = 1
    for i in range(1, R.shape[1]):
        S[:, i] = S[:, i - 1] + R[:, i]
    return S


class FeynmanKacFormula(object):
    def __init__(self, S, numPaths):
        self.numMainPaths = S[:100, :].shape[0]
        self.numPaths = numPaths  # number of path generated for calculating conditional expectation of price given time at each time steps
        self.numSteps = S.shape[1]
        self.t = np.zeros([self.numSteps])
        self.conditionalExpectationS = np.zeros([self.numMainPaths, self.numSteps])
        self.conditionalExpectationS2 = np.zeros([self.numMainPaths, self.numSteps])
        self.conditionalExpectationSTest = np.zeros([self.numMainPaths, self.numSteps])
        self.conditionalExpectationS2Test = np.zeros([self.numMainPaths, self.numSteps])
        self.conditionalVarianceS = np.zeros([self.numMainPaths, self.numSteps])
        self.mainPath = S[:100, :]

    def getConditionalExpectationS(self):
        return self.conditionalExpectationS

    def getConditionalExpectationS2(self):
        return self.conditionalExpectationS2

    def getConditionalExpectationSTest(self):
        return self.conditionalExpectationSTest

    def getConditionalExpectationS2Test(self):
        return self.conditionalExpectationS2Test

    def getVarianceEachTimeStep(self):
        return self.conditionalVarianceS

    def getTime(self):
        return self.t

    def getMainPath(self):
        return self.mainPath

    def generatePath(self, r, sigma, S0, q):
        self.t = t
        for j in range(0, self.numMainPaths):
            for i in range(0, self.numSteps):
                W = StockModels.WienerProcess()
                W.generateWiener(self.numPaths, self.numSteps-i, self.getTime()[i], self.getTime()[-1])
                s1 = StockModels.GeneralizedBrownianMotion(W)
                s1.generateStockPath(r, sigma, self.getMainPath()[j, i], q)
                self.conditionalExpectationS[j, i] = s1.GetS()[:, -1].mean()
                self.conditionalVarianceS[j, i] = s1.GetS()[:, -1].var()
                self.conditionalExpectationS2[j, i] = self.conditionalVarianceS[j, i]+self.conditionalExpectationS[j, i]**2


# numPaths = 100
# t0 = 1e-20
# T = 1
# r = 0.01
# sigma = 0.2
# S0 = 1
# q = 1.3
#
# S = GeneratePrice(logReturn)
# t = np.linspace(t0, T, logReturn.shape[1])
#
# f1 = FeynmanKacFormula(S, numPaths)
# f1.generatePath(r, sigma, S0, q)

def VarPlot(x, y):
    plt.figure(figsize=(8, 5), dpi=500)
    plt.plot(x, y.getVarianceEachTimeStep()[0,:])
    plt.xlim([0.0, x[-1]])
    plt.title('Conditional Variance Given Time.')
    plt.ylabel('Variance')
    plt.xlabel('Time')
    plt.legend()
    plt.show()