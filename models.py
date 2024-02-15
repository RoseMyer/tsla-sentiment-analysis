# import all necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow
import matplotlib.pyplot as plt
import math
import pickle
import tools
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 1000)

if __name__ == '__main__':

  # load training and validation set
  with open('pickle-folder/tsla_week1_data.pkl', 'rb') as f:
      train_data = pickle.load(f)
  # load test set
  with open('pickle-folder/tsla_week2_data.pkl', 'rb') as f:
      test_data = pickle.load(f)

  values = train_data.values.astype('float32')

  # normalize features
  scaler = MinMaxScaler(feature_range=(0, 1))
  scaled = scaler.fit_transform(values)

  # specify the number of lag periods
  window = 270
  features = 4

  # frame as supervised learning
  reframed = tools.series_to_supervised(scaled, window, 1)

  # drop columns we don't want to predict
  reframed.drop(reframed.columns[[-3,-2,-1]], axis=1, inplace=True)

  # split into train and test sets
  values = reframed.values
  train_mins = len(train_data.loc[:'2023-05-13 16:00:00'])
  train = values[:train_mins, :]
  test = values[train_mins:, :]

  # split into input and outputs
  observations = window * features
  train_X, train_y = train[:, :-1], train[:, -1]
  test_X, test_y = test[:, :-1], test[:, -1]
  print(train_X.shape, len(train_X), train_y.shape)

  # reshape input to be 3D [samples, timesteps, features]
  train_X = train_X.reshape((train_X.shape[0], window, features))
  test_X = test_X.reshape((test_X.shape[0], window, features))
  print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

  # design network
  model = tensorflow.keras.Sequential()
  model.add(tensorflow.keras.layers.LSTM(25, input_shape=(train_X.shape[1], train_X.shape[2])))
  model.add(tensorflow.keras.layers.Dense(1))
  model.compile(loss='mse', optimizer='adam')

  # fit network
  history = model.fit(train_X, train_y, epochs=20, batch_size=64, verbose=0, validation_data=(test_X, test_y), shuffle=False, workers=4)

  # plot history
  with plt.style.context('seaborn'):
      plt.plot(history.history['loss'], label='train')
      plt.plot(history.history['val_loss'], label='val', color='red')
      plt.title('Training and Validation Loss')
      plt.xlabel('Epochs')
      plt.ylabel('Loss')
      plt.legend()
  plt.show()

  # make a prediction
  y_pred = model.predict(test_X)
  test_X = test_X.reshape((test_X.shape[0], observations))

  # invert scaling for forecast
  y_pred = np.concatenate((y_pred, test_X[:, -3:]), axis=1)
  y_pred = scaler.inverse_transform(y_pred)
  y_pred = y_pred[:,0]

  # invert scaling for actual
  test_y = test_y.reshape((len(test_y), 1))
  y_actual = np.concatenate((test_y, test_X[:, -3:]), axis=1)
  y_actual = scaler.inverse_transform(y_actual)
  y_actual = y_actual[:,0]

  # calculate RMSE
  rmse = math.sqrt(mean_squared_error(y_actual, y_pred))
  print('Test RMSE: %.3f' % rmse)
  rmse_std = rmse/y_actual.std()
  print('Test RMSE/std_dev: %.3f' % rmse_std)
  mae = mean_absolute_error(y_actual, y_pred)
  print('Test MAE: %.3f' % mae)

  with plt.style.context('seaborn'):
      fig, ax = plt.subplots(1, figsize=(15,10))
      plt.title('Predicted vs Validation')
      plt.plot(train_data.index[-2610:], y_pred, label='predicted', color='red')
      plt.plot(train_data.index[-2610:], y_actual, label='actual')
      plt.xlabel('Date')
      plt.ylabel('Price')
      plt.legend()
  plt.show()

  # load dataset
  values = test_data.values

  # ensure all data is float
  values = values.astype('float32')

  # normalize features
  scaler = MinMaxScaler(feature_range=(0, 1))
  scaled = scaler.fit_transform(values)

  # specify the number of lag hours
  window = 270
  features = 4

  # frame as supervised learning
  reframed = tools.series_to_supervised(scaled, window, 1)

  # drop columns we don't want to predict
  reframed.drop(reframed.columns[[-3,-2,-1]], axis=1, inplace=True)

  # split into input and outputs
  values = reframed.values
  observations = window * features
  test_X, test_y = values[:, :-1], values[:, -1]
  # reshape input to be 3D [samples, timesteps, features]
  test_X = test_X.reshape((test_X.shape[0], window, features))
  print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

  # make a prediction
  y_pred = model.predict(test_X)
  test_X = test_X.reshape((test_X.shape[0], observations))

  # invert scaling for forecast
  y_pred = np.concatenate((y_pred, test_X[:, -3:]), axis=1)
  y_pred = scaler.inverse_transform(y_pred)
  y_pred = y_pred[:,0]

  # invert scaling for actual
  test_y = test_y.reshape((len(test_y), 1))
  y_actual = np.concatenate((test_y, test_X[:, -3:]), axis=1)
  y_actual = scaler.inverse_transform(y_actual)
  y_actual = y_actual[:,0]

  # calculate RMSE
  rmse = math.sqrt(mean_squared_error(y_actual, y_pred))
  print('Test RMSE: %.3f' % rmse)
  rmse_std = rmse/y_actual.std()
  print('Test RMSE/std_dev: %.3f' % rmse_std)
  mae = mean_absolute_error(y_actual, y_pred)
  print('Test MAE: %.3f' % mae)

  with plt.style.context('seaborn'):
      fig, ax = plt.subplots(1, figsize=(15,10))
      plt.title('Predicted vs Test')
      plt.plot(test_data.index[270:], y_pred, label='predicted', color='red')
      plt.plot(test_data.index[270:], y_actual, label='actual')
      plt.xlabel('Date')
      plt.ylabel('Price')
      plt.legend()
  plt.show()