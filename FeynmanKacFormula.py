import StockModels
import numpy as np
import matplotlib.pyplot as plt

class FeynmanKacFormula(object):
    def __init__(self, numPaths, numSteps):
        self.numPaths = numPaths
        self.numSteps = numSteps
        self.t = np.zeros([self.numSteps])
        self.conditionalExpectationS = np.zeros([self.numSteps])
        self.conditionalExpectationS2 = np.zeros([self.numSteps])
        self.conditionalVarianceS = np.zeros([self.numSteps])
        self.mainPath = np.zeros([self.numSteps])
        self.Stest = np.zeros([self.numSteps])

    def getConditionalExpectationS(self):
        return self.conditionalExpectationS

    def getConditionalExpectationS2(self):
        return self.conditionalExpectationS2

    def getVariancePaths(self):
        return self.conditionalVarianceS

    def getTime(self):
        return self.t

    def getMainPath(self):
        return self.mainPath



    def generatePath(self, r, sigma, S0, q, t0, T):
        mainPath = StockModels.NonGaussianBrownianMotion()
        mainPath.generateWiener(1, self.numSteps, t0, T)
        mainPath.generateOmega(q)
        mainPath.generateStockPath(r, sigma, S0, q)
        self.mainPath = mainPath.getS()
        self.t = mainPath.getTime()
        for i in range(0, self.numSteps):
            p1 = StockModels.NonGaussianBrownianMotion()
            p1.generateWiener(self.numPaths, self.numSteps-i, self.getTime()[i], T)
            p1.generateOmega(q)
            p1.generateStockPath(r, sigma, self.getMainPath()[0, i], q)
            self.conditionalExpectationS[i] = p1.getS()[:, -1].mean()
            self.conditionalExpectationS2[i] = (p1.getS()[:, -1]**2).mean()
            self.conditionalVarianceS[i] = p1.getS()[:, -1].std()**2

# numPaths = 1000
# numMainPaths = 1
# t0 = 1e-20
# dt = 0.005
# T = 1
# numSteps = int(T / dt)
# r = 0.05
# sigma = 0.2
# S0 = 1
# q = 1.1
# f1 = FeynmanKacFormula(numPaths, numSteps)
# f1.generatePath(r, sigma, S0, q, t0, T)



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
