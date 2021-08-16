import numpy as np
from scipy.special import gamma, factorial
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
plt.rc('text', usetex=True)


def generateOmega(numPaths, numSteps, T, q):
    dt = T / float(numSteps)
    t = np.linspace(0, T, numSteps)
    t[0] = 1e-10  # set initial t close to 0 to avoid ZeroDivisionError
    N = np.random.normal(0.0, 1.0, [numPaths, numSteps])
    c = (np.pi * gamma(1 / (q - 1) - 0.5) ** 2) / ((q - 1) * gamma(1 / (q - 1)) ** 2)
    B = c ** ((1 - q) / (3 - q)) * ((2 - q) * (3 - q) * t) ** (-2 / (3 - q))
    Z = ((2 - q) * (3 - q) * c * t) ** (1 / (3 - q))
    W = np.zeros([numPaths, numSteps])
    Omg = np.zeros([numPaths, numSteps])
    for i in range(1, numSteps):
        # making sure that samples from normal have mean 0 and variance 1
        # if numPaths > 1:
        #     N[:, i - 1] = (N[:, i - 1] - np.mean(N[:, i - 1])) / np.std(N[:, i - 1])
        W[:, i] = W[:, i - 1] + np.power(dt, 0.5) * N[:, i - 1]
        Omg[:, i] = Omg[:, i - 1] + ((1 - B[i - 1] * (1 - q) * Omg[:, i - 1] ** 2) ** 0.5 * (W[:, i] - W[:, i - 1])) / (
                Z[i - 1] ** ((1 - q) / 2))

    paths = {'time': t, 'c': c, 'B': B, 'Z': Z, 'W': W, 'Omg': Omg}
    return paths

# var = np.var(Omg_T[:, -1])

def generateStockPath(sigma, r, T, q, S_0, numPaths, numSteps):
    paths = generateOmega(numPaths, numSteps, T, q)
    dt = T / float(numSteps)
    S = np.zeros([numPaths, numSteps])
    S[:, 0] = S_0
    alpha = 0.5 * (3-q) * ((2-q) * (3-q) * paths['c'])**((q-1)/(3-q))
    for i in range(1, numSteps):
        S[:, i] = S[:, i-1] * np.exp(sigma * (paths['Omg'][:, i]-paths['Omg'][:, i-1]) +
                                     r * dt - 0.5 * sigma**2 * (1 - (1-q) * paths['B'][i] *
                                                                (paths['Omg'][:, i]-paths['Omg'][:, i-1])**2) *
                                     alpha * dt**(2/(3-q)))
    stockPaths = {'time': paths['time'],'Omg': paths['Omg'], 'S': S}
    return stockPaths


stock1 = generateStockPath(sigma=0.3, r=0.006, T=1, q=1.1, S_0=50, numPaths=100000, numSteps=1000)
stock2 = generateStockPath(sigma=0.3, r=0.006, T=1, q=1.4, S_0=50, numPaths=100000, numSteps=1000)


def logReturn(func):
    df = pd.DataFrame({'time': func['time'],
                       'stock price': func['S'][0, :]})
    df['daily log return'] = (np.log(df['stock price']/df['stock price'].shift(1)))
    df['daily log return'] = (df['daily log return']-df['daily log return'].mean())/df['daily log return'].std() # standardization
    return df['daily log return']

def TsallisVar(q, t):
    c = (np.pi * gamma(1 / (q - 1) - 0.5) ** 2) / ((q - 1) * gamma(1 / (q - 1)) ** 2)
    B = c ** ((1 - q) / (3 - q)) * ((2 - q) * (3 - q) * t) ** (-2 / (3 - q))
    return 1 / ((5 - 3 * q) * B)

def varOmg(numPaths, numSteps, T, q):
    paths = generateOmega(numPaths, numSteps, T, q)
    time = paths['time']
    Omg_t = paths['Omg']
    varOmg_t = np.zeros([numSteps])
    for i in range(1, numSteps):
        varOmg_t[i] = Omg_t[:, i].var()
    paths2 = {'time' : paths['time'],'varOmg_t': varOmg_t}
    return paths2

def distPlot(func1, func2):
    plt.figure(figsize=(8, 5), dpi=500)
    sns.histplot(func1, binwidth=0.1, binrange=[-10, 10])
    sns.histplot(func2, color='r', binwidth=0.1, binrange=[-10, 10])
    plt.xlim(-6, 6)
    plt.title('Tsallis Distribution')
    plt.show()

def distPlot2(func1):
    plt.figure(figsize=(8, 5), dpi=500)
    plt.hist(func1, bins=150)
    plt.xlim(-5, 5)
    plt.title('Tsallis Distribution')
    plt.show()


def pathPlot(numPaths=20):
    plt.figure(figsize=(8, 5), dpi=500)
    for i in range(numPaths):
        plt.plot(stock2['time'], stock2['S'][i, :])
    plt.title('Stock price path according to Tsallis statistics')
    plt.show()


def varPlot(numPaths, numSteps, T, q):
    exact = TsallisVar(q, varOmg(numPaths, numSteps, T, q)['time'][1:])
    emp = varOmg(numPaths, numSteps, T, q)['varOmg_t'][1:]
    err = (exact-emp)/exact
    plt.figure(figsize=(8,5), dpi=500)
    plt.plot(varOmg(1000, 1000, T, 1.3)['time'][1:], err)
    plt.title('Variance percentage error')
    plt.xlabel('Time')
    plt.ylabel('Error')
    plt.show()