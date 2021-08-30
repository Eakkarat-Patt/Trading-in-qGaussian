import numpy as np
from scipy.special import gamma
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
I = 10
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

rv1 = indifPrice(p1.getS(), rCoeff, sigma)

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
