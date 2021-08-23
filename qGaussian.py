import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pandas_datareader.data as web
import datetime

plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
plt.rc('text', usetex=True)

class qGaussian(object):
    def __init__(self):
        self.paths = {}

    def generateOmega(self, numPaths, numSteps, T, q):
        """
        Generate random numbers distributed according to Tsallis statistics
        :param numPaths: number of path
        :param numSteps: number of time step from 0 to terminal time T
        :param T: terminal time
        :param q: entropic index
        :return: dictionary
        """
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
            if numPaths > 1:
                N[:, i - 1] = (N[:, i - 1] - np.mean(N[:, i - 1])) / np.std(N[:, i - 1])
            W[:, i] = W[:, i - 1] + np.power(dt, 0.5) * N[:, i - 1]
            Omg[:, i] = Omg[:, i - 1] + ((1 - B[i - 1] * (1 - q) * Omg[:, i - 1] ** 2) ** 0.5 * (W[:, i] - W[:, i - 1])) / (
                    Z[i - 1] ** ((1 - q) / 2))

        paths = {'time': t, 'c': c, 'B': B, 'Z': Z, 'W': W, 'Omg': Omg}
        return paths



    def generateStockPath(self, sigma, r, T, q, S_0, numPaths, numSteps):
        """
        Generate stock paths and map all paths to self.paths dictionary
        :param sigma: volatility
        :param r: risk-free rate of return
        :param T: terminal time
        :param q: entropic index
        :param S_0: initial stock price
        :param numPaths:
        :param numSteps:
        :return:
        """
        params = self.generateOmega(numPaths, numSteps, T, q)
        dt = T / float(numSteps)
        S1 = np.zeros([numPaths, numSteps])
        S2 = np.zeros([numPaths, numSteps])
        S1[:, 0] = S_0
        S2[:, 0] = S_0
        Pq = ((1-params['B'] * (1-q) * params['Omg']**2)**(1/(1-q))) / params['Z']
        for i in range(1, numSteps):
            S1[:, i] = S1[:, i - 1] + S1[:, i - 1] * dt * (r + 0.5 * sigma**2 * Pq[:, i-1]**(1-q)) + sigma * S1[:, i-1] * \
                      (params['Omg'][:, i] - params['Omg'][:, i - 1])
            S2[:, i] = S2[:, i-1] + r * S2[:, i-1] * dt + sigma * S2[:, i-1] * (params['W'][:, i] - params['W'][:, i-1])

        self.paths['time'] = params['time']
        self.paths['c'] = params['c']
        self.paths['B'] = params['B']
        self.paths['Z'] = params['Z']
        self.paths['W'] = params['W']
        self.paths['Omg'] = params['Omg']
        self.paths['S1'] = S1
        self.paths['S2'] = S2

    def getTime(self):
        return self.paths['time']

    def getW(self):
        return self.paths['W']

    def getOmg(self):
        return self.paths['Omg']

    def getS1(self):
        return self.paths['S1']

    def getS2(self):
        return self.paths['S2']

stock1 = qGaussian()
stock2 = qGaussian()

#stock1.generateStockPath(sigma=0.3, r=0.006, T=10, q=1.1, S_0=50, numPaths=10, numSteps=100000)
stock2.generateStockPath(sigma=0.3, r=0.06, T=0.05, q=1.4, S_0=50, numPaths=10, numSteps=100000)


def logReturn(func):
    df = pd.DataFrame({'time': stock2.getTime(),
                       'stock price': func[0, :]})
    df['daily log return'] = np.log(df['stock price']/df['stock price'].shift(1))
    df['daily log return'] = (df['daily log return']-df['daily log return'].mean())/df['daily log return'].std() # standardization
    return df['daily log return']


def distPlot(func1, func2):
    plt.figure(figsize=(8, 5), dpi=500)
    sns.histplot(func1, color='r', binwidth=0.1, binrange=[-10, 10])
    sns.histplot(func2, binwidth=0.1, binrange=[-10, 10])
    plt.xlim(-6, 6)
    plt.title('Tsallis Distribution')
    plt.show()


def pathPlot(numPaths=20):
    plt.figure(figsize=(8, 5), dpi=500)
    for i in range(numPaths):
        plt.plot(stock2.getTime(), stock2.getS()[i, :])
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

