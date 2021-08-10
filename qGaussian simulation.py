import numpy as np
from scipy.special import gamma, factorial
import matplotlib.pyplot as plt


def generateOmega(numPaths, numSteps, T, q):
    dt = T / float(numSteps)
    t = np.linspace(0, T, numSteps)
    t[0] = 1e-10
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

    paths = {'time': t, 'W': W, 'Omg': Omg}
    return paths


# S[:, i + 1] = S[:, i] * np.exp(sigma * (Omg[:, i+1]-Omg[:, i])+r*dt - (np.power(sigma, 2)/2)*(1-(1-q)*(B[:, i]) * (Omg[:, i+1]-Omg[:, i])**2)
#                               * alpha * np.power(dt, 2/(3-q)))
# alpha = 0.5 * (3 - q) * ((2 - q) * (3 - q) * c)**((q-1)/(3-q))
output1 = generateOmega(numPaths=100000, numSteps=1000, T=10, q=1.1)
output2 = generateOmega(numPaths=100000, numSteps=1000, T=10, q=1.2)
output3 = generateOmega(numPaths=100000, numSteps=1000, T=10, q=1.3)

#timeGrid = output["time"]
Omg_T1 = output1['Omg']
Omg_T2 = output2['Omg']
Omg_T3 = output3['Omg']
#var = np.var(Omg_T[:, -1])


def TsallisVar(q, t):
    c = (np.pi * gamma(1 / (q - 1) - 0.5) ** 2) / ((q - 1) * gamma(1 / (q - 1)) ** 2)
    B = c ** ((1 - q) / (3 - q)) * ((2 - q) * (3 - q) * t) ** (-2 / (3 - q))
    return 1 / ((5 - 3 * q) * B)


# plt.figure(1)
# plt.grid()
# plt.hist(W_T[:, -1], 50)
# plt.xlabel("time")
# plt.ylabel("value")
# plt.title("Wiener Distribution")
# plt.show()

plt.figure(1)
plt.grid()
plt.hist([Omg_T1[:, -1], Omg_T3[:,-1]], 50)
plt.xlabel("time")
plt.ylabel("value")
plt.xlim(-25,25)
plt.title("Tsallis Distribution")
plt.show()

# plt.figure(1)
# plt.grid()
# plt.plot(timeGrid[:], W_T[1, :])
# plt.xlabel("time")
# plt.ylabel("value")
# plt.title("Wiener path")
# plt.show()
