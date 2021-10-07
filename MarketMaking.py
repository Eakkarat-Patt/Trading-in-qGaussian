import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import StockModels
import FeynmanKacFormula as fk




class MarketMakingStrategy(object):
    def __init__(self, numSims, numSteps):
        self.numSims = numSims
        self.numSteps = numSteps
        self.t = np.zeros([self.numSteps])
        self.spread = np.zeros([self.numSims, self.numSteps])
        self.bid = np.zeros([self.numSims, self.numSteps])
        self.ask = np.zeros([self.numSims, self.numSteps])
        self.n = np.zeros([self.numSims, self.numSteps])
        self.x = np.zeros([self.numSims, self.numSteps])
        self.w = np.zeros([self.numSims, self.numSteps])
        self.rvPrice = np.zeros([self.numSims, self.numSteps])
        self.S = np.zeros([self.numSims, self.numSteps])

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

    def getMuS(self):
        return self.muS

    def getS(self):
        return self.S

    def getnumSims(self):
        return self.numSims

    def getnumSteps(self):
        return self.numSteps

    def getrvPrice(self):
        return self.rvPrice

class BasicInventoryStrategy(MarketMakingStrategy):
    """
    This market making strategy assume that the stock price processes evolve according to arithmetic
    Brownian motion (Wiener process)
    """
    def __init__(self, numSims, numSteps):
        MarketMakingStrategy.__init__(self, numSims, numSteps)
        self.numSims = numSims
        self.numSteps = numSteps

    def initializeSimulation(self, gamma, k, A):
        p1 = StockModels.ArithmeticBrownianMotion()
        p1.generateWiener(numPaths, self.numSteps, t0, T)
        p1.generateStockPath(r, sigma, S0)
        for j in range(0, self.numSims):
            t = p1.getTime()
            S = p1.getS()
            spread = gamma * (sigma ** 2) * (T - t) + (2 / gamma) * np.log(1 + (gamma / k))
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
                rvPrice[i] = S[j, i] - n[i - 1] * gamma * (sigma ** 2) * (T - t[i])

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
    def __init__(self, numSims, numSteps):
        MarketMakingStrategy.__init__(self, numSims, numSteps)
        self.numSims = numSims
        self.numSteps = numSteps

    def initializeSimulation(self, gamma, k, A):
        p1 = StockModels.GeometricBrownianMotion()
        p1.generateWiener(numPaths, self.numSteps, T)
        p1.generateStockPath(r, sigma, S0)
        for j in range(0, self.numSims):

            t = p1.getTime()
            S = p1.getS()
            spread = 2/k + gamma
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
                rvPrice[i] = S[j, i]*np.exp(r*(T-t[i])) - n[i - 1] * gamma * S[j, i]**2 * (np.exp((2*r + sigma**2) * (T - t[i])))

                bid[i] = rvPrice[i] - spread / 2.
                ask[i] = rvPrice[i] + spread / 2.

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


class QGaussianInventoryStrategy(MarketMakingStrategy):
    def __init__(self, numSims, numSteps):
        MarketMakingStrategy.__init__(self, numSims, numSteps)
        self.numSims = numSims
        self.numSteps = numSteps
        self.S2 = np.zeros([numSims, numSteps])

    def initializeSimulation(self, gamma, k, A, r, sigma, S0, q, t0, T):


        for j in range(0, self.numSims):
            f1 = fk.FeynmanKacFormula(numPaths, numSteps)
            f1.generatePath(r, sigma, S0, q, t0, T)
            avgS = f1.getConditionalExpectationS()
            avgS2 = f1.getConditionalExpectationS2()
            S = f1.getMainPath()
            t = f1.getTime()
            spread = 2/k + gamma
            bid = np.zeros([self.numSteps])
            ask = np.zeros([self.numSteps])
            n = np.zeros([self.numSteps])
            x = np.zeros([self.numSteps])
            w = np.zeros([self.numSteps])
            rvPrice = np.zeros([self.numSteps])

            deltaB = np.zeros([self.numSteps])
            deltaA = np.zeros([self.numSteps])

            rvPrice[0] = avgS[0]
            bid[0] = rvPrice[0] - spread / 2.
            ask[0] = rvPrice[0] + spread / 2.



            for i in range(1, self.numSteps):
                rvPrice[i] = avgS[i] - n[i - 1] * gamma * avgS2[i]

                bid[i] = rvPrice[i] - spread / 2.
                ask[i] = rvPrice[i] + spread / 2.

                deltaB[i] = S[0, i] - bid[i]
                deltaA[i] = ask[i] - S[0, i]

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

                w[i] = x[i] + n[i] * S[0, i]
            self.t = t
            self.bid[j, :] = bid
            self.ask[j, :] = ask
            self.n[j, :] = n
            self.spread[j, :] = spread
            self.w[j, :] = w
            self.S[j, :] = S[:]
            self.rvPrice[j, :] = rvPrice


numPaths = 1000
numSims = 1
t0 = 1e-20
T = 1
dt = 0.005
numSteps = int(T / dt)
r = 0.02
sigma = 0.05
S0 = 1

gamma = 0.001
k = 100
A = 1500
q = 1.4

# mm1 = BasicInventoryStrategy('mm on ABM', numSims, numSteps)
# mm1.initializeSimulation(gamma, k, A)

# mm2 = GBMInventoryStrategy('mm with qGaussian process', numSims, numSteps)
# mm2.initializeSimulation(gamma, k, A)
#
mm3 = QGaussianInventoryStrategy(numSims, numSteps)
mm3.initializeSimulation(gamma, k, A, r, sigma, S0, q, t0, T)


def distPlot(func1):
    plt.figure(figsize=(8, 5), dpi=500)
    sns.histplot(func1, binwidth=2, color='r')
    plt.xlim([-50, 150])
    plt.title('Profit Distribution')
    plt.show()


def pathPlot(x, y1, y2, y3):
    plt.figure(figsize=(8, 5), dpi=500)
    plt.plot(x, y1[0, :], label='Stock price')
    plt.plot(x, y2[0, :], label='Ask price')
    plt.plot(x, y3[0, :], label='Bid price')
    plt.xlim([0.0, x[-1]])
    plt.title('Stock price path')
    plt.ylabel('Price')
    plt.xlabel('Time')
    plt.legend()
    plt.show()


def pathPlot2(x, y1, y2):
    plt.figure(figsize=(8, 5), dpi=500)
    plt.plot(x, y1[0, :], label='Stock price')
    plt.plot(x, y2[0, :], label='Reservation price')
    plt.xlim([0.0, x[-1]])
    plt.title('Stock price path')
    plt.ylabel('Price')
    plt.xlabel('Time')
    plt.legend()
    plt.show()