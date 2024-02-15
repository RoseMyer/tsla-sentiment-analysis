import pandas as pd
import numpy as np
import sklearn
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
from dateutil.easter import easter
import pickle
from datetime import datetime
from alpha_vantage.alphavantage import AlphaVantage
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.cryptocurrencies import CryptoCurrencies
from matplotlib.pyplot import figure
import config
plt.style.use('seaborn')
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 1000)



if __name__ == '__main__':
    # API key
    key = config.ALPHAVANTAGE_API_KEY

    # create timeseries object
    ts = TimeSeries(key, output_format='pandas')

    ticker = 'TSLA'
    data, meta_data = ts.get_intraday(symbol=ticker, outputsize='full', interval='1min')
    data = data[::-1]
    figure(num=None, figsize=(15, 6), dpi=80, facecolor='w', edgecolor='k')
    data['4. close'].plot()

    with open('pickle-folder/tsla_week3_price.pkl', 'wb') as f:
        pickle.dump(data, f)

    with open('pickle-folder/tsla_week2_price.pkl', 'rb') as f:
        tempe = pickle.load(f)
    tempe.iloc[769]

    ticker = 'AMZN'
    data, meta_data = ts.get_intraday(symbol=ticker, outputsize='full', interval='1min')
    data = data[::-1]
    figure(num=None, figsize=(15, 6), dpi=80, facecolor='w', edgecolor='k')
    data['4. close'].plot()


    with open('amzn_week2_price.pkl', 'wb') as f:
        pickle.dump(data, f)

    ticker = 'AAPL'
    data, meta_data = ts.get_intraday(symbol=ticker, outputsize='full', interval='1min')
    data = data[::-1]
    figure(num=None, figsize=(15, 6), dpi=80, facecolor='w', edgecolor='k')
    data['4. close'].plot()


    with open('aapl_week2_price.pkl', 'wb') as f:
        pickle.dump(data, f)

    ticker = 'FB'
    data, meta_data = ts.get_intraday(symbol=ticker, outputsize='full', interval='1min')
    data = data[::-1]
    figure(num=None, figsize=(15, 6), dpi=80, facecolor='w', edgecolor='k')
    data['4. close'].plot()

    with open('fb_week2_price.pkl', 'wb') as f:
        pickle.dump(data, f)

    ticker = 'NFLX'
    data, meta_data = ts.get_intraday(symbol=ticker, outputsize='full', interval='1min')
    data = data[::-1]
    figure(num=None, figsize=(15, 6), dpi=80, facecolor='w', edgecolor='k')
    data['4. close'].plot()

    with open('nflx_week2_price.pkl', 'wb') as f:
        pickle.dump(data, f)