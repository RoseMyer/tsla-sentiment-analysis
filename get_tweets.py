# import all necessary libraries
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
import sklearn
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


# class definition for BearerTokenAuth, inherits from AuthBase
class BearerTokenAuth(AuthBase):
  """class which handles bearer token requests and authentification"""
  def __init__(self, consumer_key, consumer_secret):
    """initializes a BearerTokenAuth object"""
    self.bearer_token_url = "https://api.twitter.com/oauth2/token"
    self.consumer_key = consumer_key
    self.consumer_secret = consumer_secret
    self.bearer_token = self.get_bearer_token()

  def get_bearer_token(self):
    """requests bearer token"""
    response = requests.post(
      self.bearer_token_url,
      auth=(self.consumer_key, self.consumer_secret),
      data={'grant_type': 'client_credentials'},
      headers={'User-Agent': 'LabsRecentSearchQuickStartPython'}
    )

    if response.status_code is not 200:
        raise Exception("Cannot get a Bearer token (HTTP %d): %s" %
                        (response.status_code, response.text))

    body = response.json()
    return body['access_token']

  def __call__(self, r):
    """sets headers"""
    r.headers['Authorization'] = f"Bearer %s" % self.bearer_token
    r.headers['User-Agent'] = 'LabsRecentSearchQuickStartPython'
    return r
    

# define a mysql connection to aws db
def aws_connect(db_name=None):
  """function which will connect us to aws database instance of our choosing"""
  connection = mysql.connector.connect(
  host = config.AWS_ENDPOINT,
  user = config.AWS_USER,
  passwd = config.AWS_PASSWORD,
  port = config.AWS_PORT,
  database = db_name)
  return connection



# function to create our AWS database
def create_database(cursor, database, tables):
  """creates an aws instance"""
  try:
    cursor.execute("USE {}".format(db_name))
  except mysql.connector.Error as err:
    print("Database {} does not exists.".format(db_name))
    if err.errno == errorcode.ER_BAD_DB_ERROR:
      try:
        cursor.execute("CREATE DATABASE {} DEFAULT CHARACTER SET 'utf8'".format(database))
        # create tables according to above template
        for table_name in tables:
          table_description = tables[table_name]
          try:
            print("Creating table {}: ".format(table_name), end='')
            cursor.execute(table_description)
          except mysql.connector.Error as err:
            if err.errno == errorcode.ER_TABLE_EXISTS_ERROR:
              print("already exists.")
            else:
              print(err.msg)
          else:
            print("OK")
      except mysql.connector.Error as err:
        print("Failed creating database: {}".format(err))
        exit(1)
      print("Database {} created successfully.".format(db_name))
      cnx.database = db_name
    else:
      print(err)
      exit(1)
 
        
if __name__ == "__main__":

  print("get_tweets.py is being run directly")

  cnx = aws_connect() # create a connection with aws db
  cursor = cnx.cursor() # create a cursor over the mysql connection
  db_name = '$TSLA' # name of our database

  # template for our db tables
  tables = {}
  tables['tweet_time_price2'] = (
      "CREATE TABLE tweet_time_price2 ("
      "  tweet varchar(250) NOT NULL,"
      "  id varchar(50) NOT NULL,"
      "  datetime datetime NOT NULL,"
      "  price varchar(10) NOT NULL"
      ") ENGINE=InnoDB")

  # create our database
  create_database(cursor, db_name, tables)

  cursor.close() # close connections to our cursors
  cnx.close() # close connections to our cursors
  
  cnx = aws_connect('$TSLA')# reopen a mysql connection
  
  cursor = cnx.cursor() # recreate a cursor over the mysql connection
  # insert statement to add a tweet
  add_tweet = ("INSERT INTO tweet_time_price2 (tweet, id, datetime, price) VALUES (%s, %s, %s, %s)")

  # time intervals over which we'll query our tweets
  initial_time = '2023-05-20 09:00:00'
  final_time = '2023-05-22 23:00:00'
  intervals = list(pd.date_range(initial_time, final_time, freq='T'))
  intervals1 = [str(i).split() for i in intervals]
  intervals2 = [i[0]+"T"+i[1]+"Z" for i in intervals1]

  # constant search parameters
  query = urllib.parse.quote(""""$TSLA" lang:en""")
  max_results = "100"
  headers = {"Accept-Encoding": "gzip"}

  for i in range(2694, len(intervals2)-1):
      
    start = intervals2[i]
    end =  intervals2[i+1]
    url = f"https://api.twitter.com/labs/2/tweets/search?query={query}&max_results={max_results}&start_time={start}&end_time={end}"
    
    # first page of calls
    response = requests.get(
      url,
      auth=BearerTokenAuth(
        config.TWITTER_API_KEY,
        config.TWITTER_API_SECRET_KEY
      ),
      headers=headers
    )
    if response.status_code is not 200:
        raise Exception(f"Request returned an error: %s, %s" % (response.status_code, response.text))
    parsed = json.loads(response.text)
    pretty_print = json.dumps(parsed, indent=2, sort_keys=True)
    print(pretty_print)
    # adds our parsed results to our database
    try:
      for j in parsed['data']:
        cursor.execute(add_tweet, (j['text'], j['id'], intervals2[i+1], 'null'))
        cnx.commit()
    except:
      time.sleep(5)
      continue
    time.sleep(5)

  # recall our tweets from our aws database
  cursor.execute("""SELECT tweet, id, datetime, price FROM $TSLA.tweet_time_price2 ORDER BY datetime ASC""")
  # store the tweets in a dataframe
  df1 = pd.DataFrame(cursor.fetchall())
  df1.columns = [x[0] for x in cursor.description]
  df1

  # pickle our dataframe of tweets to save the work we've done
  with open("pickle-folder/tsla_week2_tweets.pkl", 'wb') as f:
      pickle.dump(df1, f)

  # sort tweets
  df1.sort_values("tweet", inplace = True) 
  # drop duplicte values 
  df1.drop_duplicates(subset ="tweet", keep = 'first', inplace = True)
  # resort our de-duped tweets by datetime
  df1.sort_values("datetime", inplace = True)
  df1.set_index('datetime',inplace=True)
  df1

  # instantiate a VADER sentiment analyzer
  analyser = SentimentIntensityAnalyzer()
  # get sentiment scores, store compound sentiment score as our sentiment column
  df1['sentiment'] = [analyser.polarity_scores(i) for i in df1['tweet']]
  df1['compound_sentiment'] = [i['compound'] for i in df1['sentiment']]
  df1['positive_sentiment'] = [i['pos'] for i in df1['sentiment']]
  df1['negative_sentiment'] = [i['neg'] for i in df1['sentiment']]
  df1['neutral_sentiment'] = [i['neu'] for i in df1['sentiment']]
  # drop the column containing sentiment dictionary object
  df1.drop(columns=['sentiment'], inplace=True)
  df1

  # quick visual of cyclical tweet volume
  sns.distplot(df1.index)

  # save our tweets for pickle checkpoint
  with open("pickle-folder/tsla_week2_tweets_df.pkl", 'wb') as f:
      pickle.dump(df1, f)

  # save our tweets for pickle checkpoint
  with open("pickle-folder/tsla_week2_tweets_df.pkl", 'rb') as f:
      temp = pickle.load(f)

  with open("pickle-folder/tsla_week2_tweets.pkl", 'wb') as f:
      pickle.dump(temp, f)