import StockModels
import numpy as np
import matplotlib.pyplot as plt

class FeynmanKacFormula(object):
    def __init__(self, numPaths, numSteps):
        self.numPaths = numPaths
        self.numSteps = numSteps
        self.t = np.zeros([self.numSteps])
        self.meanPaths = np.zeros([self.numPaths, self.numSteps])
        self.variancePaths = np.zeros([self.numPaths, self.numSteps])

    def getMeanPaths(self):
        return self.meanPaths

    def getVariancePaths(self):
        return self.variancePaths

    def getTime(self):
        return self.t

    def generatePath(self, numMainPaths, r, sigma, S0, q, t0, T):
        mainPath = StockModels.NonGaussianBrownianMotion()
        mainPath.generateWiener(numMainPaths, self.numSteps, t0, T)
        mainPath.generateOmega(q)
        mainPath.generateStockPath(r, sigma, S0, q)
        self.t = mainPath.getTime()
        for j in range(0, numMainPaths):
            for i in range(0, self.numSteps):
                p1 = StockModels.NonGaussianBrownianMotion()
                p1.generateWiener(self.numPaths, self.numSteps-i, mainPath.getTime()[i], T)
                p1.generateOmega(q)
                p1.generateStockPath(r, sigma, mainPath.getS()[j, i], q)
                self.meanPaths[j, i] = p1.getS()[:, -1].mean()
                self.variancePaths[j, i] = p1.getS()[:, -1].std()**2

numPaths = 10000
numMainPaths = 1
t0 = 0
dt = 0.005
T = 1
numSteps = int(T / dt)
r = 0.05
sigma = 0.2
S0 = 1
q = 1.1
f1 = FeynmanKacFormula(numPaths, numSteps)
f1.generatePath(numMainPaths, r, sigma, S0, q, t0, T)


def pathPlot(x, y1):
    plt.figure(figsize=(8, 5), dpi=500)
    plt.plot(x, y1, label='Stock price')
    plt.xlim([0.0, x[-1]])
    plt.title('Stock price path')
    plt.ylabel('Price')
    plt.xlabel('Time')
    plt.legend()
    plt.show()