import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import qGaussian


def generatePoisson(numPaths, numSteps, I, T):
    t = p1.getTime()
    X = np.zeros([numPaths, numSteps])
    N = np.random.poisson(I * dt, [numPaths, numSteps])
    for i in range(numSteps):
        X[:, i] = X[:, i - 1] + N[:, i - 1]
    paths = {'t': t, 'X': X}
    return paths


numPaths = 1
T = 1
dt = 0.005
numSteps = int(T / dt)
r = 0.05
sigma = 0.5
S0 = 50
rCoeff = 1






# qGaussian.distPlot(generatePoisson(10000, 1000, 1, 1)['X'][:, -1], None)

p1 = qGaussian.ArithmeticBrownianMotion('Arithmetic Brownian Motion')
p1.generateWiener(numPaths, numSteps, T)
p1.generateStockPath(r, sigma, S0)

def Inventory(numPaths, numSteps, I, T):
    Nb = generatePoisson(numPaths, numSteps, I, T)
    Na = generatePoisson(numPaths, numSteps, I, T)
    t = Nb['t']
    n = Nb['X'] - Na['X']
    paths = {'t': t, 'n': n}
    return paths



def indifPrice(path, rCoeff, sigma):
    In = Inventory(numPaths, numSteps, I, T)
    rv = path - In['n'] * rCoeff * sigma**2 * (T-In['t'])
    paths = {'t': In['t'], 'n': In['n'], 'rv': rv}
    return paths


def InventoryPlot(x, y):
    plt.figure(figsize=(8, 5), dpi=500)
    plt.plot(x, y[0, :], label='Inventory')
    plt.title('Inventory vs Time')
    plt.ylabel('Quantity')
    plt.xlabel('Time')
    plt.legend()
    plt.show()

def pathPlot(x, y1, y2):
    plt.figure(figsize=(8, 5), dpi=500)
    plt.plot(x, y1[0, :], label='Stock price')
    plt.plot(x, y2[0, :], label='Reservation price')
    plt.title('Stock price path')
    plt.ylabel('Price')
    plt.xlabel('Time')
    plt.legend()
    plt.show()


class InventoryStrategy(object):
    def __init__(self, name):
        self.name = name
        self.spreadAvg = np.array([])
        self.PnLStd = np.array([])
        self.PnL = np.array([])

    def getProfit(self):
        return self.PnL

    def initializeSimulation(self, numSims, numSteps, gamma, k, A):

        for i in range(1, numSims):
            p1 = qGaussian.ArithmeticBrownianMotion('Arithmetic Brownian Motion')
            p1.generateWiener(numPaths, numSteps, T)
            p1.generateStockPath(r, sigma, S0)
            t = p1.getTime()
            S = p1.getS()
            bid = np.zeros([numSteps])
            ask = np.zeros([numSteps])
            rvPrice = np.zeros([numSteps])
            spread = np.zeros([numSteps])
            deltaB = np.zeros([numSteps])
            deltaA = np.zeros([numSteps])
            n = np.zeros([numSteps])
            x = np.zeros([numSteps])
            w = np.zeros([numSteps])
            rvPrice[0] = S[0][0]
            bid[0] = S[0][0]
            ask[0] = S[0][0]
            spread[0] = 0
            deltaB[0] = 0
            deltaA[0] = 0
            n[0] = 0  # position
            x[0] = 0  # wealth
            w[0] = 0

            for i in range(1, numSteps):
                rvPrice[i] = S[0][i] - n[i - 1] * gamma * (sigma ** 2) * (T - t[i])
                spread[i] = gamma * (sigma ** 2) * (T - t[i]) + (2 / gamma) * np.log(1 + (gamma / k))
                bid[i] = rvPrice[i] - spread[i] / 2.
                ask[i] = rvPrice[i] + spread[i] / 2.

                deltaB[i] = S[0][i] - bid[i]
                deltaA[i] = ask[i] - S[0][i]

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
                    x[i] = x[i - 1] - bid[i]
                    x[i] = x[i] + ask[i]

                w[i] = x[i] + n[i] * S[0][i]

            self.spreadAvg = np.append(self.spreadAvg, spread.mean())
            self.PnL = np.append(self.PnL, w[-1])
            self.PnLStd = np.append(self.PnLStd, w[-1])



def distPlot(func1):
    plt.figure(figsize=(8, 5), dpi=500)
    sns.histplot(func1, binwidth=2, color='r')
    plt.title('Profit Distribution')
    plt.show()