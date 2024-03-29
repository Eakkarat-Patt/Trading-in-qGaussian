import StockModels
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime


class FeynmanKacFormula(object):
    def __init__(self, noise, numPaths):
        self.noise = noise
        self.numMainPaths = noise.getW().shape[0]
        self.numPaths = numPaths  # number of path generated for calculating conditional expectation of price given time at each time steps
        self.numSteps = noise.getTime().shape[0]
        self.t = np.zeros([self.numSteps])
        self.conditionalExpectationS = np.zeros([self.numMainPaths, self.numSteps])
        self.conditionalExpectationS2 = np.zeros([self.numMainPaths, self.numSteps])
        self.conditionalExpectationSTest = np.zeros([self.numMainPaths, self.numSteps])
        self.conditionalExpectationS2Test = np.zeros([self.numMainPaths, self.numSteps])
        self.conditionalVarianceS = np.zeros([self.numMainPaths, self.numSteps])
        self.mainPath = np.zeros([self.numMainPaths, self.numSteps])

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
        mainPath = StockModels.GeneralizedBrownianMotion(self.noise)
        mainPath.generateStockPath(r, sigma, S0, q)
        self.mainPath = mainPath.GetS()
        self.t = mainPath.GetTime()
        for j in range(0, self.numMainPaths):
            for i in range(0, self.numSteps):
                W = StockModels.WienerProcess()
                W.generateWiener(self.numPaths, self.numSteps-i, self.getTime()[i], self.getTime()[-1])
                s1 = StockModels.GeneralizedBrownianMotion(W)
                s1.generateStockPath(r, sigma, self.getMainPath()[j, i], q)
                self.conditionalExpectationS[j, i] = s1.GetS()[:, -1].mean()
                self.conditionalVarianceS[j, i] = s1.GetS()[:, -1].var()
                self.conditionalExpectationS2[j, i] = self.conditionalVarianceS[j, i]+self.conditionalExpectationS[j, i]**2

    def generatePathTest(self, r, sigma, S0):
        mainPath = StockModels.GeometricBrownianMotion(self.noise)
        mainPath.generateStockPath(r, sigma, S0)
        self.mainPath = mainPath.GetS()
        self.t = mainPath.GetTime()
        self.conditionalExpectationSTest = self.getMainPath() * np.exp(r * (self.getTime()[-1]-self.getTime()))
        self.conditionalExpectationS2Test = self.getMainPath()**2 * (np.exp((2 * r + sigma**2) * (self.getTime()[-1] - self.getTime())))
        for j in range(0, self.numMainPaths):
            for i in range(0, self.numSteps):
                W = StockModels.WienerProcess()
                W.generateWiener(self.numPaths, self.numSteps-i, self.getTime()[i], self.getTime()[-1])
                s1 = StockModels.GeometricBrownianMotion(W)
                s1.generateStockPath(r, sigma, self.getMainPath()[j, i])
                self.conditionalExpectationS[j, i] = s1.GetS()[:, -1].mean()
                self.conditionalVarianceS[j, i] = s1.GetS()[:, -1].var()
                self.conditionalExpectationS2[j, i] = self.conditionalVarianceS[j, i]+self.conditionalExpectationS[j, i]**2





# numPaths = 1000
# t0 = 1e-20
# dt = 0.01
# T = 1
# numSteps = int(T / dt)
# r = 0.01
# sigma = 0.2
# S0 = 10
# q = 1.5
# mainW = StockModels.WienerProcess()
# mainW.generateWiener(500, numSteps, t0, T)
# f1 = FeynmanKacFormula(mainW, numPaths)
# f1.generatePathTest(r, sigma, S0)
# p2 = StockModels.GeneralizedBrownianMotion(mainW)
# p2.generateStockPath(r, sigma, S0, q)
#

def ExpectationPlot(func):
    plt.figure(figsize=(8, 5), dpi=500)
    plt.plot(func.getTime(), func.getConditionalExpectationS()[0, :])
    plt.xlim([0.0, func.getTime()[-1]])
    plt.title('Conditional Expectation of S given t')
    plt.ylabel('Conditional Expectation')
    plt.xlabel('Time')
    plt.show()

def ExpectationPlot2(func):
    plt.figure(figsize=(8, 5), dpi=500)
    plt.plot(func.getTime(), func.getConditionalExpectationS2()[0, :])
    plt.xlim([0.0, func.getTime()[-1]])
    plt.title('Conditional Expectation of S given t')
    plt.ylabel('Conditional Expectation')
    plt.xlabel('Time')
    plt.show()

def pathPlot(x, y1, y2):
    plt.figure(figsize=(8, 5), dpi=500)
    plt.plot(x, y1, label='Exact')
    plt.plot(x, y2, label='Approx')
    plt.xlim([0.0, x[-1]])
    plt.title('Stock price path')
    plt.ylabel('Price')
    plt.xlabel('Time')
    plt.legend()
    plt.show()

def errorPlot(x, y1, y2):
    exact = y1
    approx = y2
    error = (exact-approx)/exact
    plt.figure(figsize=(8, 5), dpi=500)
    plt.plot(x, error)
    plt.xlim([0.0, x[-1]])
    plt.ylabel('error')
    plt.xlabel('Time')
    plt.legend()
    plt.show()

def VarPlot(func):
    plt.figure(figsize=(8, 5), dpi=500)
    plt.plot(func.getTime(), func.getVarianceEachTimeStep()[0,:])
    plt.xlim([0.0, func.getTime()[-1]])
    plt.title('Conditional Variance Given Time.')
    plt.ylabel('Variance')
    plt.xlabel('Time')
    plt.legend()
    plt.show()
