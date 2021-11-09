import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import StockModels
import FeynmanKacFormula as fk
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

    def initializeSimulation(self, r, sigma, S0, alpha, k, A, order):
        p1 = StockModels.GeometricBrownianMotion(self.noise)
        p1.generateStockPath(r, sigma, S0)
        t = p1.GetTime()
        S = p1.GetS()
        self.ConditionalExpectationSGivenTime = S * np.exp(r * (T-t))
        self.ConditionalExpectationS2GivenTime = S**2 * (np.exp((2 * r + sigma**2) * (T - t)))
        for j in range(0, self.numSims):
            spread = 2 / k + alpha
            bid = np.zeros([self.numSteps])
            ask = np.zeros([self.numSteps])
            n = np.zeros([self.numSteps])
            x = np.zeros([self.numSteps])
            w = np.zeros([self.numSteps])
            rvPrice = np.zeros([self.numSteps])

            deltaB = np.zeros([self.numSteps])
            deltaA = np.zeros([self.numSteps])

            rvPrice[0] = S[j, 0]*np.exp(r*(T-t[0]))
            bid[0] = rvPrice[0] - spread / 2.
            ask[0] = rvPrice[0] + spread / 2.
            for i in range(1, self.numSteps):
                rvPrice[i] = self.GetConditionalExpectationSGivenTime()[j, i] - n[i - 1] * alpha * \
                             self.GetConditionalExpectationS2GivenTime()[j, i]

                bid[i] = rvPrice[i] - spread / 2.
                ask[i] = rvPrice[i] + spread / 2.

                deltaB[i] = S[j, i] - bid[i]
                deltaA[i] = ask[i] - S[j, i]

                lambdaA = A * np.exp(-k * deltaA[i])
                ProbA = lambdaA * dt
                AskArrival = order[0][j, i]

                lambdaB = A * np.exp(-k * deltaB[i])
                ProbB = lambdaB * dt
                BidArrival = order[1][j, i]
                if ProbB > BidArrival and ProbA < AskArrival:
                    n[i] = n[i - 1] + 1
                    x[i] = x[i - 1] - bid[i]

                if ProbB < BidArrival and ProbA > AskArrival:
                    n[i] = n[i - 1] - 1
                    x[i] = x[i - 1] + ask[i]
                if ProbB < BidArrival and ProbA < AskArrival:
                    n[i] = n[i - 1]
                    x[i] = x[i - 1]
                if ProbB > BidArrival and ProbA > AskArrival:
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
        for j in range(0, self.numSims):

            bid = np.zeros([self.numSteps])
            ask = np.zeros([self.numSteps])
            n = np.zeros([self.numSteps])
            x = np.zeros([self.numSteps])
            w = np.zeros([self.numSteps])
            rvPrice = np.zeros([self.numSteps])

            deltaB = np.zeros([self.numSteps])
            deltaA = np.zeros([self.numSteps])

            rvPrice[0] = self.GetConditionalExpectationSGivenTime()[j, 0]
            bid[0] = rvPrice[0] - spread / 2.
            ask[0] = rvPrice[0] + spread / 2.
            for i in range(1, self.numSteps):
                rvPrice[i] = self.GetConditionalExpectationSGivenTime()[j, i] - n[i - 1] * alpha * \
                             self.GetConditionalExpectationS2GivenTime()[j, i]

                bid[i] = rvPrice[i] - spread / 2.
                ask[i] = rvPrice[i] + spread / 2.

                deltaB[i] = self.S[j, i] - bid[i]
                deltaA[i] = ask[i] - self.S[j, i]
                lambdaA = A * np.exp(-k * deltaA[i])
                ProbA = lambdaA * dt
                AskArrival = order[0][j, i]

                lambdaB = A * np.exp(-k * deltaB[i])
                ProbB = lambdaB * dt
                BidArrival = order[1][j, i]

                if ProbB > BidArrival and ProbA < AskArrival:
                    n[i] = n[i - 1] + 1
                    x[i] = x[i - 1] - bid[i]

                if ProbB < BidArrival and ProbA > AskArrival:
                    n[i] = n[i - 1] - 1
                    x[i] = x[i - 1] + ask[i]
                if ProbB < BidArrival and ProbA < AskArrival:
                    n[i] = n[i - 1]
                    x[i] = x[i - 1]
                if ProbB > BidArrival and ProbA > AskArrival:
                    n[i] = n[i - 1]
                    x[i] = x[i - 1] - bid[i] + ask[i]

                w[i] = x[i] + n[i] * self.S[j, i]
            self.bid[j, :] = bid
            self.ask[j, :] = ask
            self.n[j, :] = n
            self.spread[j, :] = spread
            self.w[j, :] = w
            self.rvPrice[j, :] = rvPrice


numPaths = 1
numSims = 1
fkNumPaths = 1000
t0 = 1e-20
T = 1
dt = 0.005
numSteps = int(T / dt)
r = 0.002
sigma = 0.09
S0 = 50

alpha = 0.0001
k = 100
A = 1500
q = 1.38

mainW = StockModels.WienerProcess()
mainW.generateWiener(numPaths, numSteps, t0, T)

order = OrderArrival(numPaths, numSteps)

# mm1 = BasicInventoryStrategy(w, numSims)
# mm1.initializeSimulation(r, sigma, S0, alpha, k, A)

# f1 = fk.FeynmanKacFormula(mainW, numPaths)
# f1.generatePath(r, sigma, S0, q)

mm2 = GBMInventoryStrategy(mainW, numSims)
mm2.initializeSimulation(r, sigma, S0, alpha, k, A, order)


mm3 = QGaussianInventoryStrategy(mainW, numSims)
mm3.initializeSimulation(r, sigma, S0, q, alpha, k, A, fkNumPaths, order)

# df = pd.DataFrame({'Profit': mm3.getProfit()[:, -1], 'Inventory': mm3.getInventory()[:, -1]})
# df.to_csv('profit inventory Generalized GBM .csv', index=False)
# profit = pd.read_csv('Profit Generalized GBM on Normal GBM Strategy.csv')




def ProfitDistributionPlot(func1):
    plt.figure(figsize=(8, 5), dpi=500)
    sns.histplot(func1, binwidth=2, color='r')
    # plt.xlim([-50, 150])
    plt.title('Profit Distribution')
    plt.show()


def SpreadPlot(x, y1, y2, y3):
    plt.figure(figsize=(8, 5), dpi=500)
    plt.plot(x, y1, label='Stock price')
    plt.plot(x, y2, label='Ask price')
    plt.plot(x, y3, label='Bid price')
    plt.xlim([0.0, x[-1]])
    plt.title('Stock price path')
    plt.ylabel('Price')
    plt.xlabel('Time')
    plt.legend()
    plt.show()


def rvPricePlot(x, y1, y2):
    plt.figure(figsize=(8, 5), dpi=500)
    plt.plot(x, y1[0, :], label='Stock price')
    plt.plot(x, y2[0, :], label='Reservation price')
    plt.xlim([0.0, x[-1]])
    plt.title('Stock price path')
    plt.ylabel('Price')
    plt.xlabel('Time')
    plt.legend()
    plt.show()


def InventoryPlot(x, y):
    plt.figure(figsize=(8, 5), dpi=500)
    plt.plot(x, y[0, :])
    plt.xlim([0.0, x[-1]])
    plt.title('Share quantity holded')
    plt.ylabel('Number of share')
    plt.xlabel('Time')
    plt.legend()
    plt.show()

def WealthPlot(x, y1, y2):
    plt.figure(figsize=(8, 5), dpi=500)
    plt.plot(x, y1[0, :], label='Gaussian')
    plt.plot(x, y2[0, :], label='Tsallis q = {}'.format(mm3.GetEntropyIndex()))
    plt.xlim([0.0, x[-1]])
    plt.title('Wealth between two strategies')
    plt.ylabel('Wealth')
    plt.xlabel('Time')
    plt.legend()
    plt.show()



def ExpectationPlot(x, y1, y2):
    plt.figure(figsize=(8, 5), dpi=500)
    plt.plot(x, y1[0, :], label='GBM')
    plt.plot(x, y2[0, :], label='Generalized GBM q = {}'.format(q))
    plt.xlim([0.0, x[-1]])
    plt.title('Conditional Expectation of S Given t')
    plt.ylabel('$E[S_T]$')
    plt.xlabel('Time')
    plt.legend()
    plt.show()


def BidAskPlot(x, bid1, ask1, bid2, ask2):
    plt.figure(figsize=(8, 5), dpi=500)
    plt.plot(x, bid1[0, :], label='GBM', color='r')
    plt.plot(x, ask1[0, :], color='r')
    plt.plot(x, bid2[0, :], label='Generalized GBM q = {}'.format(q), color='b')
    plt.plot(x, ask2[0, :], color='b')
    plt.xlim([0.0, x[-1]])
    plt.title('Bid Ask Spread vs Time')
    plt.ylabel('$S$')
    plt.xlabel('$t$')
    plt.legend()
    plt.show()
