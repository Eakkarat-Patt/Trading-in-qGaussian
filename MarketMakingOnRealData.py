import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import StockModels
from scipy.stats import skew
import pandas as pd


def GeneratePrice(op, high, low, close):
    o1 = np.zeros([op.shape[0], op.shape[1]])
    h1 = np.zeros([high.shape[0], high.shape[1]])
    l1 = np.zeros([low.shape[0], low.shape[1]])
    cl1 = np.zeros([close.shape[0], close.shape[1]])
    o1[:, 0] = 1
    h1[:, 0] = o1[:, 0] + (high[:, 0] - op[:, 0]) / op[:, 0]
    l1[:, 0] = o1[:, 0] + (low[:, 0] - op[:, 0]) / op[:, 0]
    cl1[:, 0] = o1[:, 0] + (close[:, 0] - op[:, 0]) / op[:, 0]
    for i in range(1, op.shape[1]):
        o1[:, i] = o1[:, i - 1] * (op[:, i] / op[:, i - 1])
        h1[:, i] = h1[:, i - 1] * (high[:, i] / high[:, i - 1])
        l1[:, i] = l1[:, i - 1] * (low[:, i] / low[:, i - 1])
        cl1[:, i] = cl1[:, i - 1] * (close[:, i] / close[:, i - 1])
    return o1, h1, l1, cl1


class FeynmanKacFormula(object):
    def __init__(self, S, numPaths, numSims):
        self.numMainPaths = S[:numSims, :].shape[0]
        self.numPaths = numPaths  # number of path generated for calculating conditional expectation of price given time at each time steps
        self.numSteps = S.shape[1]
        self.t = np.zeros([self.numSteps])
        self.conditionalExpectationS = np.zeros([self.numMainPaths, self.numSteps])
        self.conditionalExpectationS2 = np.zeros([self.numMainPaths, self.numSteps])
        self.conditionalExpectationSTest = np.zeros([self.numMainPaths, self.numSteps])
        self.conditionalExpectationS2Test = np.zeros([self.numMainPaths, self.numSteps])
        self.conditionalVarianceS = np.zeros([self.numMainPaths, self.numSteps])
        self.mainPath = S[:numSims, :]

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

    def generatePath(self, r, sigma, q):
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


class MarketMakingStrategy(object):
    def __init__(self, S, H, L,  numSims):
        self.S = S[:numSims, :]
        self.low = L[:numSims, :]
        self.high = H[:numSims, :]
        self.numSims = numSims
        self.numSteps = self.S.shape[1]
        self.t = np.zeros([self.numSteps])
        self.spread = np.zeros([self.numSims, self.numSteps])
        self.bid = np.zeros([self.numSims, self.numSteps])
        self.ask = np.zeros([self.numSims, self.numSteps])
        self.n = np.zeros([self.numSims, self.numSteps])
        self.x = np.zeros([self.numSims, self.numSteps])
        self.w = np.zeros([self.numSims, self.numSteps])
        self.rvPrice = np.zeros([self.numSims, self.numSteps])
        self.ConditionalExpectationSGivenTime = np.zeros([self.numSims, self.numSteps])
        self.ConditionalExpectationS2GivenTime = np.zeros([self.numSims, self.numSteps])
        self.OrderConsumption = np.zeros([self.numSims, self.numSteps])
        self.p = np.zeros([self.numSims, self.numSteps])

    def getProfit(self):
        return self.w

    def getSpread(self):
        return self.spread

    def getInventory(self):
        return self.n

    def getBid(self):
        return self.bid

    def getAsk(self):
        return self.ask

    def getTime(self):
        return self.t

    def getS(self):
        return self.S

    def getnumSims(self):
        return self.numSims

    def getnumSteps(self):
        return self.numSteps

    def getrvPrice(self):
        return self.rvPrice

    def GetConditionalExpectationSGivenTime(self):
        return self.ConditionalExpectationSGivenTime

    def GetConditionalExpectationS2GivenTime(self):
        return self.ConditionalExpectationS2GivenTime

    def GetOrderConsumption(self):
        return self.OrderConsumption

    def GetCash(self):
        return self.x

    def GetPosition(self):
        return self.p

    def GetHigh(self):
        return self.high

    def GetLow(self):
        return self.low


class GBMInventoryStrategy(MarketMakingStrategy):
    def __init__(self, S, H, L, numSims):
        MarketMakingStrategy.__init__(self, S, H, L, numSims)

    def initializeSimulation(self, r0, sigma0, alpha, k):
        spread = 2 / k + alpha
        self.t = t
        self.ConditionalExpectationSGivenTime = self.getS() * np.exp(r0 * (T-self.getTime()))
        self.ConditionalExpectationS2GivenTime = self.getS()**2 * (np.exp((2 * r0 + sigma0**2) * (T - self.getTime())))
        self.rvPrice[:, 0] = self.GetConditionalExpectationSGivenTime()[:, 0]
        self.bid[:, 0] = self.getrvPrice()[:, 0] - spread / 2
        self.ask[:, 0] = self.getrvPrice()[:, 0] + spread / 2


        for j in range(0, self.numSims):
            for i in range(1, self.numSteps):
                self.rvPrice[j, i] = self.GetConditionalExpectationSGivenTime()[j, i] - self.getInventory()[j, i-1] * alpha * \
                             self.GetConditionalExpectationS2GivenTime()[j, i]

                self.bid[j, i] = self.getrvPrice()[j, i] - spread / 2.
                self.ask[j, i] = self.getrvPrice()[j, i] + spread / 2.

                if self.GetLow()[j, i] <= self.getBid()[j, i] and self.GetHigh()[j, i] < self.getAsk()[j, i]:
                    self.n[j, i] = self.n[j, i - 1] + 1
                    self.x[j, i] = self.x[j, i - 1] - self.getBid()[j, i]
                    self.OrderConsumption[j, i] = 1
                if self.GetLow()[j, i] > self.getBid()[j, i] and self.GetHigh()[j, i] >= self.getAsk()[j, i]:
                    self.n[j, i] = self.n[j, i - 1] - 1
                    self.x[j, i] = self.x[j, i - 1] + self.getAsk()[j, i]
                    self.OrderConsumption[j, i] = 2
                if self.GetLow()[j, i] > self.getBid()[j, i] and self.GetHigh()[j, i] < self.getAsk()[j, i]:
                    self.n[j, i] = self.n[j, i - 1]
                    self.x[j, i] = self.x[j, i - 1]
                    self.OrderConsumption[j, i] = 3
                if self.GetLow()[j, i] <= self.getBid()[j, i] and self.GetHigh()[j, i] >= self.getAsk()[j, i]:
                    self.n[j, i] = self.n[j, i - 1]
                    self.x[j, i] = self.x[j, i - 1] - self.getBid()[j, i] + self.getAsk()[j, i]
                    self.OrderConsumption[j, i] = 4
                self.p[j, i] = self.getInventory()[j, i] * self.getS()[j, i]
                self.w[j, i] = self.GetCash()[j, i] + self.GetPosition()[j, i]


class QGaussianInventoryStrategy(MarketMakingStrategy):
    def __init__(self, S, H, L, numSims):
        MarketMakingStrategy.__init__(self, S, H, L, numSims)
        self.q = 0

    def GetEntropyIndex(self):
        return self.q

    def initializeSimulation(self, r, sigma, q, alpha, k, fkNumPaths):
        f1 = FeynmanKacFormula(self.getS(), fkNumPaths, self.getnumSims())
        f1.generatePath(r, sigma, q)
        self.ConditionalExpectationSGivenTime = f1.getConditionalExpectationS()
        self.ConditionalExpectationS2GivenTime = f1.getConditionalExpectationS2()
        self.S = f1.getMainPath()
        self.t = f1.getTime()
        self.q = q
        spread = 2 / k + alpha
        self.rvPrice[:, 0] = self.GetConditionalExpectationSGivenTime()[:, 0]
        self.bid[:, 0] = self.getrvPrice()[:, 0] - spread / 2
        self.ask[:, 0] = self.getrvPrice()[:, 0] + spread / 2
        for j in range(0, self.numSims):
            for i in range(1, self.numSteps):
                self.rvPrice[j, i] = self.GetConditionalExpectationSGivenTime()[j, i] - self.getInventory()[j, i - 1] * \
                                     alpha * self.GetConditionalExpectationS2GivenTime()[j, i]

                self.bid[j, i] = self.getrvPrice()[j, i] - spread / 2.
                self.ask[j, i] = self.getrvPrice()[j, i] + spread / 2.

                if self.GetLow()[j, i] <= self.getBid()[j, i] and self.GetHigh()[j, i] < self.getAsk()[j, i]:
                    self.n[j, i] = self.n[j, i - 1] + 1
                    self.x[j, i] = self.x[j, i - 1] - self.getBid()[j, i]
                    self.OrderConsumption[j, i] = 1
                if self.GetLow()[j, i] > self.getBid()[j, i] and self.GetHigh()[j, i] >= self.getAsk()[j, i]:
                    self.n[j, i] = self.n[j, i - 1] - 1
                    self.x[j, i] = self.x[j, i - 1] + self.getAsk()[j, i]
                    self.OrderConsumption[j, i] = 2
                if self.GetLow()[j, i] > self.getBid()[j, i] and self.GetHigh()[j, i] < self.getAsk()[j, i]:
                    self.n[j, i] = self.n[j, i - 1]
                    self.x[j, i] = self.x[j, i - 1]
                    self.OrderConsumption[j, i] = 3
                if self.GetLow()[j, i] <= self.getBid()[j, i] and self.GetHigh()[j, i] >= self.getAsk()[j, i]:
                    self.n[j, i] = self.n[j, i - 1]
                    self.x[j, i] = self.x[j, i - 1] - self.getBid()[j, i] + self.getAsk()[j, i]
                    self.OrderConsumption[j, i] = 4
                self.p[j, i] = self.getInventory()[j, i] * self.getS()[j, i]
                self.w[j, i] = self.GetCash()[j, i] + self.GetPosition()[j, i]


p = np.loadtxt('Dataset/sp500 open time series.txt')
h = np.loadtxt('Dataset/sp500 high time series.txt')
l = np.loadtxt('Dataset/sp500 low time series.txt')
cl = np.loadtxt('Dataset/sp500 close time series.txt')

path = GeneratePrice(p, h, l, cl)

numSims = 200
fkNumPaths = 200
t0 = 1e-20
T = 1
dt = 1/p.shape[1]


t = np.linspace(t0, T, p.shape[1])

# Fit params
r0 = 0.00028
sigma0 = 0.013
r = 0.0002
sigma = 0.015/1.55
q = 1.51

# Test params
# r0 = 0.001
# sigma0 = 0.08
# r = 0.02
# sigma = 0.2
# q = 1.4


alpha = 0.0001
k = 3000

# op = GeneratePrice(p)
# high = GeneratePrice(h)
# low = GeneratePrice(l)
# cl = GeneratePrice(close)


mm2 = GBMInventoryStrategy(path[0], path[1], path[2], numSims)
mm2.initializeSimulation(r0, sigma0, alpha, k)

mm3 = QGaussianInventoryStrategy(path[0], path[1], path[2], numSims)
mm3.initializeSimulation(r, sigma, q, alpha, k, fkNumPaths)
# mm3 = QGaussianInventoryStrategy(S, numSims)
# mm3.initializeSimulation(r, sigma, S0, q, S, alpha, k, A, fkNumPaths, order)

print('mm2 cash mean: ' + str(mm2.GetCash()[:, -1].mean()))
print('mm3 cash mean: ' + str(mm3.GetCash()[:, -1].mean()))
print('mm2 position value mean: ' + str(mm2.GetPosition()[:, -1].mean()))
print('mm3 position value mean: ' + str(mm3.GetPosition()[:, -1].mean()))
print('mm2 profit mean: ' + str(mm2.getProfit()[:, -1].mean()))
print('mm3 profit mean: ' + str(mm3.getProfit()[:, -1].mean()))
print('mm2 profit std: ' + str(mm2.getProfit()[:, -1].std()))
print('mm3 profit std: ' + str(mm3.getProfit()[:, -1].std()))
print('mm2 inventory mean: ' + str(mm2.getInventory()[:, -1].mean()))
print('mm3 inventory mean: ' + str(mm3.getInventory()[:, -1].mean()))
print('mm2 inventory std: ' + str(mm2.getInventory()[:, -1].std()))
print('mm3 inventory std: ' + str(mm3.getInventory()[:, -1].std()))


def OrderConsumptionAvg():
    x1 = np.zeros(numSims)
    x2 = np.zeros(numSims)
    for i in range(numSims):
        x1[i] = np.count_nonzero(mm2.GetOrderConsumption()[i, :] == 1) + \
             np.count_nonzero(mm2.GetOrderConsumption()[i, :] == 2) + \
             np.count_nonzero(mm2.GetOrderConsumption()[i, :] == 4)
        x2[i] = np.count_nonzero(mm3.GetOrderConsumption()[i, :] == 1) + \
             np.count_nonzero(mm3.GetOrderConsumption()[i, :] == 2) + \
             np.count_nonzero(mm3.GetOrderConsumption()[i, :] == 4)
    return x1, x2
def Savetxt():
    np.savetxt('mm2 Reservation price sp500.txt', mm2.getrvPrice()[:, :], fmt='%1.4f')
    np.savetxt('mm2 Cash sp500.txt', mm2.GetCash()[:, :], fmt='%1.4f')
    np.savetxt('mm2 Position sp500.txt', mm2.GetPosition()[:, :], fmt='%1.4f')
    np.savetxt('mm2 Profit sp500.txt', mm2.getProfit()[:, :], fmt='%1.4f')
    np.savetxt('mm2 Inventory sp500.txt', mm2.getInventory()[:, :], fmt='%s')
    np.savetxt('mm2 Bid sp500.txt', mm2.getBid()[:, :], fmt='%1.4f')
    np.savetxt('mm2 Ask sp500.txt', mm2.getAsk()[:, :], fmt='%1.4f')
    np.savetxt('mm2 Time sp500.txt', mm2.getTime()[:], fmt='%1.4f')
    np.savetxt('mm2 Price sp500.txt', mm2.getS()[:, :], fmt='%1.4f')
    np.savetxt('mm2 Order sp500.txt', mm2.GetOrderConsumption()[:, :], fmt='%s')
    np.savetxt('mm2 DeltaA sp500.txt', mm2.GetDeltaA()[:, :], fmt='%1.4f')
    np.savetxt('mm2 DeltaB sp500.txt', mm2.GetDeltaB()[:, :], fmt='%1.4f')
    np.savetxt('mm2 ProbA sp500.txt', mm2.GetProbA()[:, :], fmt='%1.4f')
    np.savetxt('mm2 ProbB sp500.txt', mm2.GetProbB()[:, :], fmt='%1.4f')
    np.savetxt('mm3 Reservation price sp500.txt', mm3.getrvPrice()[:, :], fmt='%1.4f')
    np.savetxt('mm3 Cash sp500.txt', mm3.GetCash()[:, :], fmt='%1.4f')
    np.savetxt('mm3 Position sp500.txt', mm3.GetPosition()[:, :], fmt='%1.4f')
    np.savetxt('mm3 Profit sp500.txt', mm3.getProfit()[:, :], fmt='%1.4f')
    np.savetxt('mm3 Inventory sp500.txt', mm3.getInventory()[:, :], fmt='%s')
    np.savetxt('mm3 Bid sp500.txt', mm3.getBid()[:, :], fmt='%1.4f')
    np.savetxt('mm3 Ask sp500.txt', mm3.getAsk()[:, :], fmt='%1.4f')
    np.savetxt('mm3 Time sp500.txt', mm3.getTime()[:], fmt='%1.4f')
    np.savetxt('mm3 Price sp500.txt', mm3.getS()[:, :], fmt='%1.4f')
    np.savetxt('mm3 Order sp500.txt', mm3.GetOrderConsumption()[:, :], fmt='%s')
    np.savetxt('mm3 DeltaA sp500.txt', mm3.GetDeltaA()[:, :], fmt='%1.4f')
    np.savetxt('mm3 DeltaB sp500.txt', mm3.GetDeltaB()[:, :], fmt='%1.4f')
    np.savetxt('mm3 ProbA sp500.txt', mm3.GetProbA()[:, :], fmt='%1.4f')
    np.savetxt('mm3 ProbB sp500.txt', mm3.GetProbB()[:, :], fmt='%1.4f')



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

def DistributionPlot(func1, func2, title, binwidth):
    plt.figure(figsize=(8, 5), dpi=500)
    sns.histplot(func1[:, -1], binwidth=binwidth, binrange=[func2[:, -1].min(), func2[:, -1].max()], color='r')
    sns.histplot(func2[:, -1], binwidth=binwidth, binrange=[func2[:, -1].min(), func2[:, -1].max()], color='b')
    # plt.xlim([-50, 150])
    plt.xlim([min(func1[:, -1]), max(func1[:, -1])])
    plt.title(title)
    plt.show()

def TimeSeriesPlot(x, y1, y2, pathNum, stop, label1=None, label2=None, legend=True, ylabel=None):
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


def TimeSeriesPlot2(x, y1, y2, y3, y4, y5, pathNum, stop, label1=None, label2=None, label3=None, label4=None, label5=None, legend=True, ylabel=None):
    plt.figure(figsize=(8, 5), dpi=500)
    plt.plot(x[x <= stop], y1[pathNum, x <= stop], label=label1)
    plt.plot(x[x <= stop], y2[pathNum, x <= stop], label=label2)
    plt.plot(x[x <= stop], y3[pathNum, x <= stop], label=label3)
    plt.plot(x[x <= stop], y4[pathNum, x <= stop], label=label4)
    plt.plot(x[x <= stop], y5[pathNum, x <= stop], label=label5)
    plt.xlim([0, stop])
    # plt.ylim([1.06, 1.11])
    plt.ylabel(ylabel)
    plt.xlabel('Time')
    plt.title('k = {}'.format(k))
    if legend:
        plt.legend()
    plt.show()

def DistPlot(func1, title, label=None, logScale=False):
    plt.figure(figsize=(8, 5), dpi=500)
    sns.histplot(func1, label=label, log_scale=(False, logScale))
    plt.legend()
    plt.title(title)
    plt.show()