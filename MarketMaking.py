import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import qGaussian




class MarketMakingStrategy(object):
    def __init__(self, name, numSims, numSteps):
        self.name = name
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
        self.muS = np.zeros([self.numSims, self.numSteps])

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
    This market making strategy assume that the stock price processes involve according to arithmetic
    Brownian motion (Wiener process)
    """
    def __init__(self, name, numSims, numSteps):
        MarketMakingStrategy.__init__(self, name, numSims, numSteps)
        self.name = name
        self.numSims = numSims
        self.numSteps = numSteps

    def initializeSimulation(self, gamma, k, A):
        p1 = qGaussian.ArithmeticBrownianMotion('Arithmetic Brownian Motion')
        p1.generateWiener(numPaths, self.numSteps, T)
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
    def __init__(self, name, numSims, numSteps):
        MarketMakingStrategy.__init__(self, name, numSims, numSteps)
        self.name = name
        self.numSims = numSims
        self.numSteps = numSteps

    def initializeSimulation(self, gamma, k, A):
        p1 = qGaussian.GeometricBrownianMotion('Geometric Brownian motion')
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
    def __init__(self, name, numSims, numSteps):
        MarketMakingStrategy.__init__(self, name, numSims, numSteps)
        self.name = name
        self.numSims = numSims
        self.numSteps = numSteps

    def initializeSimulation(self, gamma, k, A, q):
        for j in range(0, self.numSims):
            p1 = qGaussian.NonGaussianBrownianMotion('Non Gaussian Geometric Brownian motion')
            p1.generateWiener(numPaths, self.numSteps, T)
            p1.generateOmega(q)
            p1.generateStockPath(r, sigma, S0, q)
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

            rvPrice[0] = S[j, 0]
            bid[0] = rvPrice[0] - spread / 2.
            ask[0] = rvPrice[0] + spread / 2.

            for i in range(1, self.numSteps):
                self.muS[j, i] = S[:, i].mean()
                rvPrice[i] = self.muS[j, i] - n[i - 1] * gamma * (S[:, i]**2).mean()

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


numPaths = 10000
numSims = 1000
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

mm2 = GBMInventoryStrategy('mm with qGaussian process', numSims, numSteps)
mm2.initializeSimulation(gamma, k, A)
#
# mm3 = QGaussianInventoryStrategy('mm with qGaussian process', numSims, numSteps)
# mm3.initializeSimulation(gamma, k, A, q)


def distPlot(func1):
    plt.figure(figsize=(8, 5), dpi=500)
    sns.histplot(func1, binwidth=2, color='r')
    plt.xlim([-50, 150])
    plt.title('Profit Distribution')
    plt.show()


def pathPlot(x, y1, y2, y3):
    plt.figure(figsize=(8, 5), dpi=500)
    plt.plot(x, y1[19, :], label='Stock price')
    plt.plot(x, y2[19, :], label='Bid price')
    plt.plot(x, y3[19, :], label='Ask price')
    plt.xlim([0.0, x[-1]])
    plt.title('Stock price path')
    plt.ylabel('Price')
    plt.xlabel('Time')
    plt.legend()
    plt.show()


def pathPlot2(x, y1, y2):
    plt.figure(figsize=(8, 5), dpi=500)
    plt.plot(x, y1[16, :], label='Stock price')
    plt.plot(x, y2[16, :], label='Reservation price')
    plt.xlim([0.0, x[-1]])
    plt.title('Stock price path')
    plt.ylabel('Price')
    plt.xlabel('Time')
    plt.legend()
    plt.show()