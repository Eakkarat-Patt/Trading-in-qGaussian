import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pandas_datareader.data as web
import datetime



def TsallisDist(numSamples):
    Z = np.random.normal(0.0, 1.0, [numSamples])


def spReturn():
    start = datetime.datetime(1980, 1, 1)
    end = datetime.datetime(2021, 12, 31)

    price = web.DataReader(['sp500'], 'fred', start, end)
    price['daily_return'] = (price['sp500'] / price['sp500'].shift(1)) - 1
    price['daily_log_return'] = (np.log(price['sp500'] / price['sp500'].shift(1)))
    mean = price['daily_log_return'].mean()
    std = price['daily_log_return'].std()
    price['normalized log return'] = (price["daily_log_return"] - mean) / std
    plt.figure(figsize=(8, 5))