import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd



def TsallisDist(numSamples):
    Z = np.random.normal(0.0, 1.0, [numSamples])


def spReturn(start, end=None):
    df = pd.read_csv('spx_d.csv')[['Date', 'Close']]
    df = df.set_index('Date')
    if end == None:
        df = df.loc[start:]
    else:
        df = df.loc[start:end]
    print(df.shape)
    df['Log return'] = np.log(df['Close']/df['Close'].shift(1))
    df['Log return'] = (df['Log return']-df['Log return'].mean())/df['Log return'].std() #Standardization
    plt.figure(figsize=(8,5), dpi=400)
    sns.histplot(df['Log return'], binwidth=0.1, kde=True, color='r', binrange=[-8, 8])
    sns.histplot(np.random.normal(0, df['Log return'].var(), [df.shape[0]]), kde=True, binwidth=0.1, binrange=[-8, 8])
    plt.xlim(-6, 6)
    plt.show()

spReturn('2000-01-01', end=None)