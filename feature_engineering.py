# import all necessary libraries

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow
import keras
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
import time
import json
import re
import math
import pickle
from datetime import datetime
from datetime import timedelta
import requests
from requests.auth import AuthBase
import config
import tools
import mysql.connector
import urllib.parse
from mysql.connector import errorcode
import sqlite3
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 1000)

if __name__ == "main":
  # reload our tweets from pickle checkpoint
    with open("pickle-folder/tsla_week2_tweets.pkl", 'rb') as f:
        df1 = pickle.load(f)
    df1

    # read in pickle file from alphavantage api call
    with open('pickle-folder/tsla_week3_price.pkl', 'rb') as f:
        tsla_data = pickle.load(f)
    # create a new column containing average price, calculated as arithmetic average of open and close
    tsla_data['avg_price'] = ((tsla_data['1. open'] + tsla_data['4. close'])/2.0)
    tsla_data = tsla_data[:'2023-05-22 16:00:00']
    tsla_data

    # capture tweet volume information
    df1['tweet_count'] = [1 for i in range(len(df1.index))]
    count_only = df1['tweet_count'].groupby('datetime').sum()
    count_only

    # discard all but the sentiment and volume information from our tweet dataframe
    sent_only = df1.drop(columns=['tweet','id','price']).groupby('datetime').mean()
    sent_only['tweet_count'] = count_only
    sent_only

    # quick visual of tweet volume distribution
    sns.distplot(sent_only['tweet_count'])

    # convert from UTC to EST, as twitter API defaults to UTC, but AlphaVantage defaults to EST
    sent_only['eastern_time'] = sent_only.index - timedelta(hours = 4)
    sent_only.set_index('eastern_time', drop=True, inplace=True)
    sent_only

    # chunk up our price dataframe into trading days
    wed = tsla_data.loc['2023-05-20 09:31':'2023-05-20 16:00:00']
    thur = tsla_data.loc['2023-05-21 09:31':'2023-05-21 16:00:00']
    fri = tsla_data.loc['2023-05-22 09:31':'2023-05-22 16:00:00']

    # add padding to the beginning of our price series so it aligns with our tweets
    a = wed.index[0]
    mins = int(60*4.5) + 1
    timeList = []
    for x in range (0, mins):
        timeList.append(a - timedelta(minutes = x))
    timeList = timeList[1:]
    time_adj_df1 = pd.DataFrame(timeList)
    time_adj_df1['1. open'] = np.NaN
    time_adj_df1['2. high'] = np.NaN
    time_adj_df1['3. low'] = np.NaN
    time_adj_df1['4. close'] = np.NaN
    time_adj_df1['5. volume'] = np.NaN
    time_adj_df1['avg_price'] = np.NaN
    time_adj_df1 = time_adj_df1[::-1]
    time_adj_df1['date'] = time_adj_df1[0]
    time_adj_df1.set_index('date', drop=True, inplace=True)
    time_adj_df1.drop(columns=[0], inplace=True)


    # create datapoints for wednesday night so our data remains continuous
    a = thur.index[0]
    mins = int(60*17.5) + 1
    timeList = []
    for x in range (0, mins):
        timeList.append(a - timedelta(minutes = x))
    timeList = timeList[1:]
    wed_night = pd.DataFrame(timeList)
    wed_night['1. open'] = np.NaN
    wed_night['2. high'] = np.NaN
    wed_night['3. low'] = np.NaN
    wed_night['4. close'] = np.NaN
    wed_night['5. volume'] = np.NaN
    wed_night['avg_price'] = np.NaN
    wed_night = wed_night[::-1]
    wed_night['date'] = wed_night[0]
    wed_night.set_index('date', drop=True, inplace=True)
    wed_night.drop(columns=[0], inplace=True)

    # create datapoints for thursday night so our data remains continuous
    a = fri.index[0]
    mins = int(60*17.5) + 1
    timeList = []
    for x in range (0, mins):
        timeList.append(a - timedelta(minutes = x))
    timeList = timeList[1:]
    thur_night = pd.DataFrame(timeList)
    thur_night['1. open'] = np.NaN
    thur_night['2. high'] = np.NaN
    thur_night['3. low'] = np.NaN
    thur_night['4. close'] = np.NaN
    thur_night['5. volume'] = np.NaN
    thur_night['avg_price'] = np.NaN
    thur_night = thur_night[::-1]
    thur_night['date'] = thur_night[0]
    thur_night.set_index('date', drop=True, inplace=True)
    thur_night.drop(columns=[0], inplace=True)

    # concatenate our trading days, nights, and adjustment dataframes
    dfs = [time_adj_df1, wed, wed_night, thur, thur_night, fri]
    tsla_round_clock = pd.concat(dfs)

    # ensure that all of our datetime stamps are actually datetimes, as some were merely strings
    tsla_round_clock[0] = [pd.to_datetime(i) for i in tsla_round_clock.index]
    tsla_round_clock.set_index(0, inplace=True)
    tsla_round_clock.sort_values(0, inplace=True)
    tsla_round_clock

    # integrate our tweet sentiment and volume information into our price series and volume dataframe
    for row in tsla_round_clock.index:
        if row in sent_only.index:
            tsla_round_clock.loc[row, 'compound_sentiment'] = sent_only.loc[row, 'compound_sentiment']
            tsla_round_clock.loc[row, 'positive_sentiment'] = sent_only.loc[row, 'positive_sentiment']
            tsla_round_clock.loc[row, 'negative_sentiment'] = sent_only.loc[row, 'negative_sentiment']
            tsla_round_clock.loc[row, 'neutral_sentiment'] = sent_only.loc[row, 'neutral_sentiment']
            tsla_round_clock.loc[row, 'tweet_count'] = sent_only.loc[row, 'tweet_count']
        else:
            tsla_round_clock.loc[row, 'compound_sentiment'] = np.NaN
            tsla_round_clock.loc[row, 'positive_sentiment'] = np.NaN
            tsla_round_clock.loc[row, 'negative_sentiment'] = np.NaN
            tsla_round_clock.loc[row, 'neutral_sentiment'] = np.NaN
            tsla_round_clock.loc[row, 'tweet_count'] = 0
    tsla_round_clock


    # another checkpoint
    with open('pickle-folder/tsla_week2_combined.pkl', 'wb') as f:
        pickle.dump(tsla_round_clock, f)

    # reload from checkpoint
    with open('pickle-folder/tsla_week2_combined.pkl', 'rb') as f:
        tsla_round_clock = pickle.load(f)

    # visual of our current price series, clearly discontinuous overnight and mid-day on wednesday
    fig, ax = plt.subplots(1, figsize=(15,10))
    plt.plot(tsla_round_clock.index, tsla_round_clock.avg_price)

    # visual of compound sentiment time series


    with plt.style.context('seaborn'):
        fig, ax = plt.subplots(1, figsize=(15,10))
        plt.plot(tsla_round_clock.index, tsla_round_clock.compound_sentiment, '--bo')
        plt.title('Compound Sentiment over Time')
        plt.xlabel('Date Time')
        plt.ylabel('Compound Sentiment Intensity')
        plt.legend()
    plt.show()

    # another pickle checkpoint
    with open('pickle-folder/tsla_week2_combined.pkl', 'wb') as f:
        pickle.dump(tsla_round_clock, f)

    # reload from pickle checkpoint
    with open('pickle-folder/tsla_week2_combined.pkl', 'rb') as f:
        tsla_round_clock = pickle.load(f)
    tsla_round_clock

    # fill NaN values with last valid value
    tsla_round_clock1 = tsla_round_clock.fillna(method='ffill')
    tsla_round_clock1['trade_volume'] = tsla_round_clock['5. volume'].fillna(0)
    tsla_round_clock1['avg_price'].fillna(808.01, inplace=True)
    # drop superfluous columns
    tsla_round_clock1.drop(columns=['1. open','2. high','3. low','4. close', '5. volume'], inplace=True)
    tsla_round_clock = tsla_round_clock1
    tsla_round_clock

    # visualization of our completed price series
    fig, ax = plt.subplots(1, figsize=(15,10))
    plt.plot(tsla_round_clock.index, tsla_round_clock.avg_price)

    # frequency distribution of various types of sentiments captured
    with plt.style.context('seaborn'):
        fig, ax = plt.subplots(1, figsize=(15,10), sharex=True)
        plt.title('Compound Sentiment over Time')
        plt.plot(tsla_round_clock.index, tsla_round_clock.compound_sentiment)
        ax.set_xlabel('Date Time')
        ax.set_ylabel('Compound Sentiment Intensity')
        plt.legend()
    plt.show()

    # frequency distribution of various types of sentiments captured
    with plt.style.context('seaborn'):
        fig, ax = plt.subplots(1, figsize=(15,10), sharex=True)
        plt.title('Twitter Volume over Time')
        plt.plot(tsla_round_clock.index, tsla_round_clock['tweet_count'],)
        ax.set_xlabel('Date Time')
        ax.set_ylabel('Number of Tweets')
        plt.legend()
    plt.show()


    # frequency distribution of various types of sentiments captured
    with plt.style.context('seaborn'):
        fig, ax = plt.subplots(1, figsize=(15,10), sharex=True)
        plt.title('Sentiment Distribution')
        sns.distplot(tsla_round_clock.compound_sentiment, color='blue', label='compound', ax=ax)
        sns.distplot(tsla_round_clock.positive_sentiment, color='green', label='positive', ax=ax)
        sns.distplot(tsla_round_clock.negative_sentiment, color='red', label='negative', ax=ax)
        sns.distplot(tsla_round_clock.neutral_sentiment, color = 'orange', label='neutral', ax=ax)
        ax.set_xlabel('Sentiment Intensity')
        ax.set_ylabel('Sentiment Frequency')
        plt.legend()
    plt.show()


    # more detailed visualization of compound sentiment, which we will most likely use in our final analysis
    fig, (ax, ax2) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)}, figsize = (15,10))

    sns.distplot(tsla_round_clock.compound_sentiment, 
                hist=True, hist_kws={
                                    "linewidth": 2,
                                    "edgecolor" :'red',
                                    "alpha": 0.4, 
                                    "color":  "w",
                                    "label": "Histogram",
                                    },
                kde=True, kde_kws = {'linewidth': 3,
                                    'color': "blue",
                                    "alpha": 0.7,
                                    'label':'Kernel Density Estimation Plot'
                                    },
                fit= sp.stats.norm, fit_kws = {'color' : 'green',
                                            'label' : 'parametric fit',
                                            "alpha": 0.7,
                                            'linewidth':3},
                ax=ax2)
    ax2.set_title('Density Estimations')
    ax2.set_ylabel('frequency')
    sns.boxplot(x=tsla_round_clock.compound_sentiment, ax = ax,color = 'red')
    ax.set_title('Box and Whiskers Plot')
    plt.legend();


    # another pickle checkpoint
    with open('pickle-folder/tsla_week2_combined.pkl', 'wb') as f:
        pickle.dump(tsla_round_clock, f)



    # reload from pickle checkpoint
    with open('pickle-folder/tsla_week2_combined.pkl', 'rb') as f:
        tsla_round_clock = pickle.load(f)
    tsla_round_clock

    data = tsla_round_clock.drop(columns=['positive_sentiment', 'negative_sentiment', 'neutral_sentiment'])

    # another pickle checkpoint
    with open('pickle-folder/tsla_week2_data.pkl', 'wb') as f:
        pickle.dump(data, f)

    # reload from pickle checkpoint
    with open('pickle-folder/tsla_week2_data.pkl', 'rb') as f:
        data = pickle.load(f)