import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import StockModels
import FeynmanKacFormula as fk
from scipy.stats import skew
import pandas as pd


def OrderArrival(numPaths, numSteps):
    fa = np.random.random([numPaths, numSteps])
    fb = np.random.random([numPaths, numSteps])
    return np.array([fa, fb])


class MarketMakingStrategy(object):
    def __init__(self, noise, numSims):
        self.noise = noise
        self.numSims = numSims
        self.numSteps = self.noise.getW().shape[1]
        self.t = np.zeros([self.numSteps])
        self.spread = np.zeros([self.numSims, self.numSteps])
        self.bid = np.zeros([self.numSims, self.numSteps])
        self.ask = np.zeros([self.numSims, self.numSteps])
        self.n = np.zeros([self.numSims, self.numSteps])
        self.x = np.zeros([self.numSims, self.numSteps])
        self.w = np.zeros([self.numSims, self.numSteps])
        self.rvPrice = np.zeros([self.numSims, self.numSteps])
        self.S = np.zeros([self.numSims, self.numSteps])
        self.ConditionalExpectationSGivenTime = np.zeros([self.numSims, self.numSteps])
        self.ConditionalExpectationS2GivenTime = np.zeros([self.numSims, self.numSteps])
        self.OrderConsumption = np.zeros([self.numSims, self.numSteps])
        self.ProbA = np.zeros([self.numSims, self.numSteps])
        self.ProbB = np.zeros([self.numSims, self.numSteps])
        self.lambdaA = np.zeros([self.numSims, self.numSteps])
        self.lambdaB = np.zeros([self.numSims, self.numSteps])
        self.deltaA = np.zeros([self.numSims, self.numSteps])
        self.deltaB = np.zeros([self.numSims, self.numSteps])
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

    def GetProbA(self):
        return self.ProbA

    def GetProbB(self):
        return self.ProbB

    def GetLambdaA(self):
        return self.lambdaA

    def GetLambdaB(self):
        return self.lambdaB

    def GetCash(self):
        return self.x

    def GetDeltaA(self):
        return self.deltaA

    def GetDeltaB(self):
        return self.deltaB

    def GetPosition(self):
        return self.p


class BasicInventoryStrategy(MarketMakingStrategy):
    """
    This market making strategy assume that the stock price processes evolve according to arithmetic
    Brownian motion (Wiener process)
    """
    def __init__(self, noise, numSims):
        MarketMakingStrategy.__init__(self, noise, numSims)

    def initializeSimulation(self, r, sigma, S0, alpha, k, A):
        p1 = StockModels.ArithmeticBrownianMotion(self.noise)
        p1.generateStockPath(r, sigma, S0)
        for j in range(0, self.numSims):
            t = p1.GetTime()
            S = p1.GetS()
            spread = alpha * (sigma ** 2) * (T - t) + (2 / alpha) * np.log(1 + (alpha / k))
            bid = np.zeros([self.numSteps])
            ask = np.zeros([self.numSteps])
            n = np.zeros([self.numSteps])
            x = np.zeros([self.numSteps])
            w = np.zeros([self.numSteps])
            rvPrice = np.zeros([self.numSteps])

            deltaB = np.zeros([self.numSteps])
            deltaA = np.zeros([self.numSteps])

            rvPrice[0] = S[j, 0]
            bid[0] = rvPrice[0] - spread[0] / 2.
            ask[0] = rvPrice[0] + spread[0] / 2.

            for i in range(1, self.numSteps):
                rvPrice[i] = S[j, i] - n[i - 1] * alpha * (sigma ** 2) * (T - t[i])

                bid[i] = rvPrice[i] - spread[i] / 2.
                ask[i] = rvPrice[i] + spread[i] / 2.

                deltaB[i] = S[j, i] - bid[i]
                deltaA[i] = ask[i] - S[j, i]

                lambdaA = A * np.exp(-k * deltaA[i])
                ProbA = lambdaA * dt
                fa = np.random.random()

                lambdaB = A * np.exp(-k * deltaB[i])
                ProbB = lambdaB * dt
                fb = np.random.random()

                if ProbB > fb and ProbA < fa:
                    n[i] = n[i - 1] + 1
                    x[i] = x[i - 1] - bid[i]

                if ProbB < fb and ProbA > fa:
                    n[i] = n[i - 1] - 1
                    x[i] = x[i - 1] + ask[i]
                if ProbB < fb and ProbA < fa:
                    n[i] = n[i - 1]
                    x[i] = x[i - 1]
                if ProbB > fb and ProbA > fa:
                    n[i] = n[i - 1]
                    x[i] = x[i - 1] - bid[i] + ask[i]

                w[i] = x[i] + n[i] * S[j, i]
            self.t = t
            self.bid[j, :] = bid
            self.ask[j, :] = ask
            self.n[j, :] = n
            self.spread[j, :] = spread
            self.w[j, :] = w
            self.S[j, :] = S[j, :]
            self.rvPrice[j, :] = rvPrice

class GBMInventoryStrategy(MarketMakingStrategy):
    def __init__(self, noise, numSims):
        MarketMakingStrategy.__init__(self, noise, numSims)

    def initializeSimulation(self, r, r0, sigma, sigma0, S0, alpha, k, A, order):
        p1 = StockModels.GeneralizedBrownianMotion(self.noise)
        p1.generateStockPath(r, sigma, S0, q)
        spread = 2 / k + alpha
        self.t = p1.GetTime()
        self.S = p1.GetS()
        self.ConditionalExpectationSGivenTime = self.getS() * np.exp(r0 * (T-self.getTime()))
        self.ConditionalExpectationS2GivenTime = self.getS()**2 * (np.exp((2 * r0 + sigma0**2) * (T - self.getTime())))
        self.rvPrice[:, 0] = self.GetConditionalExpectationSGivenTime()[:, 0]
        self.bid[:, 0] = self.getrvPrice()[:, 0] - spread / 2
        self.ask[:, 0] = self.getrvPrice()[:, 0] + spread / 2
        self.deltaA[:, 0] = self.getAsk()[:, 0] - self.getS()[:, 0]
        self.deltaB[:, 0] = self.getS()[:, 0] - self.getBid()[:, 0]
        self.lambdaA[:, 0] = A * np.exp(-k * self.GetDeltaA()[:, 0])
        self.lambdaB[:, 0] = A * np.exp(-k * self.GetDeltaB()[:, 0])
        self.ProbA[:, 0] = self.GetLambdaA()[:, 0] * dt
        self.ProbB[:, 0] = self.GetLambdaB()[:, 0] * dt
        for j in range(0, self.numSims):
            for i in range(1, self.numSteps):
                self.rvPrice[j, i] = self.GetConditionalExpectationSGivenTime()[j, i] - self.getInventory()[j, i-1] * alpha * \
                             self.GetConditionalExpectationS2GivenTime()[j, i]

                self.bid[j, i] = self.getrvPrice()[j, i] - spread / 2.
                self.ask[j, i] = self.getrvPrice()[j, i] + spread / 2.

                self.deltaB[j, i] = self.getS()[j, i] - self.getBid()[j, i]
                self.deltaA[j, i] = self.getAsk()[j, i] - self.getS()[j, i]

                self.lambdaA[j, i] = A * np.exp(-k * self.GetDeltaA()[j, i])
                self.ProbA[j, i] = self.GetLambdaA()[j, i] * dt
                AskArrival = order[0][j, i]

                self.lambdaB[j, i] = A * np.exp(-k * self.GetDeltaB()[j, i])
                self.ProbB[j, i] = self.GetLambdaB()[j, i] * dt
                BidArrival = order[1][j, i]
                if self.ProbB[j, i] > BidArrival and self.ProbA[j, i] < AskArrival:
                    self.n[j, i] = self.n[j, i - 1] + 1
                    self.x[j, i] = self.x[j, i - 1] - self.getBid()[j, i]
                    self.OrderConsumption[j, i] = 1

                if self.ProbB[j, i] < BidArrival and self.ProbA[j, i] > AskArrival:
                    self.n[j, i] = self.n[j, i - 1] - 1
                    self.x[j, i] = self.x[j, i - 1] + self.getAsk()[j, i]
                    self.OrderConsumption[j, i] = 2
                if self.ProbB[j, i] < BidArrival and self.ProbA[j, i] < AskArrival:
                    self.n[j, i] = self.n[j, i - 1]
                    self.x[j, i] = self.x[j, i - 1]
                    self.OrderConsumption[j, i] = 3
                if self.ProbB[j, i] > BidArrival and self.ProbA[j, i] > AskArrival:
                    self.n[j, i] = self.n[j, i - 1]
                    self.x[j, i] = self.x[j, i - 1] - self.getBid()[j, i] + self.getAsk()[j, i]
                    self.OrderConsumption[j, i] = 4
                self.p[j, i] = self.getInventory()[j, i] * self.getS()[j, i]
                self.w[j, i] = self.GetCash()[j, i] + self.GetPosition()[j, i]


class QGaussianInventoryStrategy(MarketMakingStrategy):
    def __init__(self, noise, numSims):
        MarketMakingStrategy.__init__(self, noise, numSims)
        self.q = 0

    def GetEntropyIndex(self):
        return self.q

    def initializeSimulation(self, r, sigma, S0, q, alpha, k, A, fkNumPaths, order):
        f1 = fk.FeynmanKacFormula(self.noise, fkNumPaths)
        f1.generatePath(r, sigma, S0, q)
        self.ConditionalExpectationSGivenTime = f1.getConditionalExpectationS()
        self.ConditionalExpectationS2GivenTime = f1.getConditionalExpectationS2()
        self.S = f1.getMainPath()
        self.t = f1.getTime()
        self.q = q
        spread = 2 / k + alpha
        self.rvPrice[:, 0] = self.GetConditionalExpectationSGivenTime()[:, 0]
        self.bid[:, 0] = self.getrvPrice()[:, 0] - spread / 2
        self.ask[:, 0] = self.getrvPrice()[:, 0] + spread / 2
        self.deltaA[:, 0] = self.getAsk()[:, 0] - self.getS()[:, 0]
        self.deltaB[:, 0] = self.getS()[:, 0] - self.getBid()[:, 0]
        self.lambdaA[:, 0] = A * np.exp(-k * self.GetDeltaA()[:, 0])
        self.lambdaB[:, 0] = A * np.exp(-k * self.GetDeltaB()[:, 0])
        self.ProbA[:, 0] = self.GetLambdaA()[:, 0] * dt
        self.ProbB[:, 0] = self.GetLambdaB()[:, 0] * dt
        for j in range(0, self.numSims):
            for i in range(1, self.numSteps):
                self.rvPrice[j, i] = self.GetConditionalExpectationSGivenTime()[j, i] - self.getInventory()[j, i - 1] * \
                                     alpha * self.GetConditionalExpectationS2GivenTime()[j, i]

                self.bid[j, i] = self.getrvPrice()[j, i] - spread / 2.
                self.ask[j, i] = self.getrvPrice()[j, i] + spread / 2.

                self.deltaB[j, i] = self.getS()[j, i] - self.getBid()[j, i]
                self.deltaA[j, i] = self.getAsk()[j, i] - self.getS()[j, i]

                self.lambdaA[j, i] = A * np.exp(-k * self.GetDeltaA()[j, i])
                self.ProbA[j, i] = self.GetLambdaA()[j, i] * dt
                AskArrival = order[0][j, i]

                self.lambdaB[j, i] = A * np.exp(-k * self.GetDeltaB()[j, i])
                self.ProbB[j, i] = self.GetLambdaB()[j, i] * dt
                BidArrival = order[1][j, i]
                if self.ProbB[j, i] > BidArrival and self.ProbA[j, i] < AskArrival:
                    self.n[j, i] = self.n[j, i - 1] + 1
                    self.x[j, i] = self.x[j, i - 1] - self.getBid()[j, i]
                    self.OrderConsumption[j, i] = 1

                if self.ProbB[j, i] < BidArrival and self.ProbA[j, i] > AskArrival:
                    self.n[j, i] = self.n[j, i - 1] - 1
                    self.x[j, i] = self.x[j, i - 1] + self.getAsk()[j, i]
                    self.OrderConsumption[j, i] = 2
                if self.ProbB[j, i] < BidArrival and self.ProbA[j, i] < AskArrival:
                    self.n[j, i] = self.n[j, i - 1]
                    self.x[j, i] = self.x[j, i - 1]
                    self.OrderConsumption[j, i] = 3
                if self.ProbB[j, i] > BidArrival and self.ProbA[j, i] > AskArrival:
                    self.n[j, i] = self.n[j, i - 1]
                    self.x[j, i] = self.x[j, i - 1] - self.getBid()[j, i] + self.getAsk()[j, i]
                    self.OrderConsumption[j, i] = 4
                self.p[j, i] = self.getInventory()[j, i] * self.getS()[j, i]
                self.w[j, i] = self.GetCash()[j, i] + self.GetPosition()[j, i]

class QGaussianInventoryStrategyOnRealData(MarketMakingStrategy):
    def __init__(self, noise, numSims):
        MarketMakingStrategy.__init__(self, noise, numSims)


numPaths = 500
numSims = 500
fkNumPaths = 1000
t0 = 1e-20
T = 10
dt = 0.1
numSteps = int(T / dt)
S0 = 10

# TSLA params
# r0 = 0.004
# sigma0 = 0.068
# r = 0.003
# sigma = 0.044
# q = 1.5

#Test params
r0 = 0.005
sigma0 = 0.05
r = 0.005
sigma = 0.05
q = 1.3

alpha = 0.01
k = 15
A = 10

mainW = StockModels.WienerProcess()
mainW.generateWiener(numPaths, numSteps, t0, T)

order = OrderArrival(numPaths, numSteps)


mm2 = GBMInventoryStrategy(mainW, numSims)
mm2.initializeSimulation(r, r0, sigma, sigma0, S0, alpha, k, A, order)

mm3 = QGaussianInventoryStrategy(mainW, numSims)
mm3.initializeSimulation(r, sigma, S0, q, alpha, k, A, fkNumPaths, order)

# df2 = pd.DataFrame({'Time': mm2.getTime()})
# for i in range(numSims):
#     df2['rvPrice'+ str(i)] = mm2.getrvPrice()[i, :]
#     df2['Bid' + str(i)] = mm2.getBid()[i, :]
#     df2['Ask' + str(i)] = mm2.getAsk()[i, :]
#     df2['Inventory'+str(i)] = mm2.getInventory()[i, :]
#     df2['Profit' + str(i)] = mm2.getProfit()[i, :]
# df2.to_csv('Data/6.12.21/GBM alpha={} k={} A={} r={} sigma={}.csv'.format(alpha, k , A, r, sigma),
#            index=False)
# #
# df3 = pd.DataFrame({'Time': mm3.getTime()})
# for i in range(numSims):
#     df3['rvPrice' + str(i)] = mm3.getrvPrice()[i, :]
#     df3['Bid' + str(i)] = mm3.getBid()[i, :]
#     df3['Ask' + str(i)] = mm3.getAsk()[i, :]
#     df3['Inventory'+str(i)] = mm3.getInventory()[i, :]
#     df3['Profit' + str(i)] = mm3.getProfit()[i, :]
# df2.to_csv('Data/6.12.21/qG alpha={} k={} A={} r={} sigma={} q={}.csv'.format(alpha, k , A, r, sigma, mm3.GetEntropyIndex()),
#            index=False)
# gbm = pd.read_csv('Data/6.12.21/GBM alpha=0.0001 k=1.5 A=100 r=0.002 sigma=0.05.csv')
# qg = pd.read_csv('Data/6.12.21/qG alpha=0.0001 k=1.5 A=100 r=0.002 sigma=0.05 q=1.3.csv')

print('mm2 profit mean: ' + str(mm2.getProfit()[:, -1].mean()))
print('mm3 profit mean: ' + str(mm3.getProfit()[:, -1].mean()))
print('mm2 profit std: ' + str(mm2.getProfit()[:, -1].std()))
print('mm3 profit std: ' + str(mm3.getProfit()[:, -1].std()))
print('mm2 inventory mean: ' + str(mm2.getInventory()[:, -1].mean()))
print('mm3 inventory mean: ' + str(mm3.getInventory()[:, -1].mean()))
print('mm2 inventory std: ' + str(mm2.getInventory()[:, -1].std()))
print('mm3 inventory std: ' + str(mm3.getInventory()[:, -1].std()))


def Savetxt():
    np.savetxt('mm2 Reservation price.txt', mm2.getrvPrice()[:, :], fmt='%1.4f')
    np.savetxt('mm2 Cash.txt', mm2.GetCash()[:, :], fmt='%1.4f')
    np.savetxt('mm2 Position.txt', mm2.GetPosition()[:, :], fmt='%1.4f')
    np.savetxt('mm2 Profit.txt', mm2.getProfit()[:, :], fmt='%1.4f')
    np.savetxt('mm2 Inventory.txt', mm2.getInventory()[:, :], fmt='%s')
    np.savetxt('mm2 Bid.txt', mm2.getBid()[:, :], fmt='%1.4f')
    np.savetxt('mm2 Ask.txt', mm2.getAsk()[:, :], fmt='%1.4f')
    np.savetxt('mm2 Time.txt', mm2.getTime()[:], fmt='%1.4f')
    np.savetxt('mm2 Price.txt', mm2.getS()[:, :], fmt='%1.4f')
    np.savetxt('mm3 Reservation price.txt', mm3.getrvPrice()[:, :], fmt='%1.4f')
    np.savetxt('mm3 Cash.txt', mm3.GetCash()[:, :], fmt='%1.4f')
    np.savetxt('mm3 Position.txt', mm3.GetPosition()[:, :], fmt='%1.4f')
    np.savetxt('mm3 Profit.txt', mm3.getProfit()[:, :], fmt='%1.4f')
    np.savetxt('mm3 Inventory.txt', mm3.getInventory()[:, :], fmt='%s')
    np.savetxt('mm3 Bid.txt', mm3.getBid()[:, :], fmt='%1.4f')
    np.savetxt('mm3 Ask.txt', mm3.getAsk()[:, :], fmt='%1.4f')
    np.savetxt('mm3 Time.txt', mm3.getTime()[:], fmt='%1.4f')
    np.savetxt('mm3 Price.txt', mm3.getS()[:, :], fmt='%1.4f')


def ProbPlot(func1, func2, pathNum, numStep):
    plt.figure(figsize=(8, 5), dpi=500)
    for i in range(numStep):
        plt.scatter(func1.getTime()[i], func1.GetProbA()[pathNum, i], color='r', s=7)
        plt.scatter(func2.getTime()[i], func2.GetProbA()[pathNum, i], color='g', s=7)
        plt.scatter(func2.getTime()[i], order[0][pathNum, i], color='b', s=5)
    plt.xlim([0.0, func1.getTime()[-1]])
    plt.title('Bid price path')
    plt.ylabel('Price')
    plt.xlabel('Time')
    plt.legend()
    plt.show()


def ProbPlot2(x, y1, y2, pathNum, stop):
    plt.figure(figsize=(8, 5), dpi=500)
    plt.scatter(x[x <= stop], y1[pathNum, x <= stop], s=7, label='GBM')
    plt.scatter(x[x <= stop], y2[pathNum, x <= stop], s=7, label='qGaussian q={}'.format(mm3.GetEntropyIndex()))
    plt.scatter(x[x <= stop], order[0][pathNum, x <= stop], s=5, label='iid random number')
    plt.xlim([0, stop])
    # plt.ylim([1.06, 1.11])
    plt.ylabel('Probability')
    plt.xlabel('Time')
    plt.legend()
    plt.show()


def DistributionPlot(func1, func2, title, binwidth):
    plt.figure(figsize=(8, 5), dpi=500)
    sns.histplot(func1[:, -1], binwidth=binwidth, binrange=[func2[:, -1].min(), func2[:, -1].max()], color='r')
    sns.histplot(func2[:, -1], binwidth=binwidth, binrange=[func2[:, -1].min(), func2[:, -1].max()], color='b')
    # plt.xlim([-50, 150])
    plt.xlim([min(func1[:, -1]), max(func1[:, -1])])
    plt.title(title)
    plt.show()


def DistPlot(func1, title, label=None, logScale=False):
    plt.figure(figsize=(8, 5), dpi=500)
    sns.histplot(func1, label=label, log_scale=(False, logScale))
    plt.legend()
    plt.title(title)
    plt.show()


def DistPlotMultiple(func1, title, logScale=False):
    plt.figure(figsize=(8, 5), dpi=500)
    sns.histplot(func1[0, :], color='r', log_scale=(False, logScale))
    sns.histplot(func1[4, :], color='g', log_scale=(False, logScale))
    sns.histplot(func1[8, :], color='b', log_scale=(False, logScale))
    sns.histplot(func1[16, :], color='y', log_scale=(False, logScale))
    plt.legend()
    plt.title(title)
    plt.show()


def BidPlot(func1, func2, pathNum):
    plt.figure(figsize=(8, 5), dpi=500)
    plt.plot(func1.getTime(), func1.getBid()[pathNum, :], label='GBM')
    plt.plot(func2.getTime(), func2.getBid()[pathNum, :], label='qGaussian')
    for i in range(func1.getnumSteps()):
        if func1.GetOrderConsumption()[pathNum, i] == 1 or func1.GetOrderConsumption()[pathNum, i] == 4:
            plt.scatter(func1.getTime()[i], func1.getBid()[pathNum, i], color='r', s=9)
        if func2.GetOrderConsumption()[pathNum, i] == 1 or func2.GetOrderConsumption()[pathNum, i] == 4:
            plt.scatter(func2.getTime()[i], func2.getBid()[pathNum, i], color='g', s=9)
    plt.xlim([0.0, func1.getTime()[-1]])
    plt.title('Bid price path')
    plt.ylabel('Price')
    plt.xlabel('Time')
    plt.legend()
    plt.show()


def AskPlot(func1, func2, pathNum):
    plt.figure(figsize=(8, 5), dpi=500)
    plt.plot(func1.getTime(), func1.getAsk()[pathNum, :], label='GBM')
    plt.plot(func2.getTime(), func2.getAsk()[pathNum, :], label='qGaussian')
    for i in range(func1.getnumSteps()):
        if func1.GetOrderConsumption()[pathNum, i] == 2 or func1.GetOrderConsumption()[pathNum, i] == 4:
            plt.scatter(func1.getTime()[i], func1.getAsk()[pathNum, i], color='r', s=9)
        if func2.GetOrderConsumption()[pathNum, i] == 2 or func2.GetOrderConsumption()[pathNum, i] == 4:
            plt.scatter(func2.getTime()[i], func2.getAsk()[pathNum, i], color='g', s=9)
    plt.xlim([0.0, func1.getTime()[-1]])
    plt.title('Ask price path')
    plt.ylabel('Price')
    plt.xlabel('Time')
    plt.legend()
    plt.show()


def SpreadPlot(func, pathNum, title=None):
    plt.figure(figsize=(8, 5), dpi=500)
    plt.plot(func.getTime(), func.getrvPrice()[pathNum, :], label='Reservation price')
    plt.plot(func.getTime(), func.getS()[pathNum, :], label='Price')
    plt.plot(func.getTime(), func.getBid()[pathNum, :], label='Bid')
    plt.plot(func.getTime(), func.getAsk()[pathNum, :], label='Ask')
    plt.xlim([0.0, func.getTime()[-1]])
    plt.title(title)
    plt.ylabel('Price')
    plt.xlabel('Time')
    plt.legend()
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


def BidAskPlot(x, bid1, ask1, bid2, ask2, stop):
    plt.figure(figsize=(8, 5), dpi=500)
    plt.plot(x[x <= stop], bid1[0, x <= stop], label='GBM', color='r')
    plt.plot(x[x <= stop], ask1[0, x <= stop], color='r')
    plt.plot(x[x <= stop], bid2[0, x <= stop], label='Generalized GBM q = {}'.format(q), color='b')
    plt.plot(x[x <= stop], ask2[0, x <= stop], color='b')
    plt.xlim([0.0, stop])
    plt.title('Bid Ask Spread vs Time')
    plt.ylabel('$S$')
    plt.xlabel('$t$')
    plt.legend()
    plt.show()

