import pandas as pd
from pandas import read_csv
from datetime import datetime
from matplotlib import pyplot as plt
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import LSTM
from numpy import concatenate
from sklearn.metrics import mean_squared_error
from math import sqrt
from tensorflow.python.keras.regularizers import l2
import sys

class RecurrentNeuralNet:
  def __init__(self, url, learning_rate, epochs, batch, reg, header=True):
    raw_input = pd.read_csv(url, header=0)
    raw_input.columns = ['Temperature', 'Exhaust Vacuum', 'Ambient Pressure', 'Relative Humidity', 'Eneger Output']
    # print out first 5 rows
    print('First Five Rows of Dataset: \n')
    print(raw_input.head(5))
    self.values = self.preprocess(raw_input)
    #split dataset into train_X, train_y, test_X, test_y
    a = self.split_dataset(self.values)
    self.train_X = a[0]
    self.train_y = a[1]
    self.test_X = a[2]
    self.test_y = a[3]
    #shape dataset into dimension[sample length, 1 timestep, number of attributes]
    self.train_X = self.train_X.reshape((self.train_X.shape[0], 1, self.train_X.shape[1]))
    self.test_X = self.test_X.reshape((self.test_X.shape[0], 1, self.test_X.shape[1]))    
    #make a Sequential model in keras
    self.model = self.plotNetwork(self.train_X, self.train_y, self.test_X, self.test_y, learning_rate, epochs, batch, reg)
    self.model.summary()
    #make prediction
    x = self.predict(self.model, self.train_X, self.train_y)
    b = self.predict(self.model, self.test_X, self.test_y)
    #calculate error RMSE
    rmse_train = sqrt(mean_squared_error(x[0], x[1]))
    rmse = sqrt(mean_squared_error(b[0], b[1]))
    print('Train RSME: %.3f' %rmse_train)
    print('Test RMSE: %.3f' % rmse)

  def preprocess(self, X):
    values = X.values.astype('float32')
    # normalize/scale features so they're between 0 and 1
    self.scaler = MinMaxScaler(feature_range=(0, 1))
    self.scaled = self.scaler.fit_transform(values)
    print('-----------------------------------------------------------------------')
    print('First Five Rows of Dataset After Scaling: \n')
    print(self.scaled)
    # call the method to transform series to a supervised learning
    reframed = self.series_to_supervised(self.scaled, 1, 1)
    # drop columns unwanted: we only want the energy output so 9 is not dropped
    reframed.drop(reframed.columns[[5,6,7,8]], axis=1, inplace=True)
    print('-----------------------------------------------------------------------')
    print('Reframed dataset: \n')
    print(reframed.head())

    return reframed.values

  def split_dataset(self, X):
    num_train = int(0.67 * len(X)) # number of train set data
    num_test = len(X) - num_train # the remaining number of data is the test data
    self.train = X[:num_train,:] # all rows till we reach num_train
    self.test = X[num_train:,:] # the rest is the test data
    #split into x and y set
    self.train_X = self.train[:,:-1] # all columns except the last column
    self.train_y = self.train[:,-1] # only last column
    self.test_X = self.test[:,:-1] # all columns except the last column
    self.test_y = self.test[:,-1] # only the last column

    return [self.train_X, self.train_y, self.test_X, self.test_y]
    
  #convert series to supervised learning
  def series_to_supervised(self, data, input_n=1, output_n=1, dropnan=True):
    if type(data) is list:
      n = 1
    else:
      n = data.shape[1]
    df = DataFrame(data)
    columns = list()
    names = list()
    # input (t-n, ... t-1)
    while input_n > 0:
      columns.append(df.shift(input_n))
      j = 0
      while (j < n):
        suffix ='v%d(t-%d)' % (j+1, input_n) #%d is a placeholder for int
        names += [suffix] # append all t-1, t-2, t-3, ... t-n to names list
        j += 1
      input_n -= 1
    # predict sequence (t, t+1, ... t+n)
    count = 0
    while (count < output_n):
      columns.append(df.shift(-count))
      if count == 0:
        j = 0
        while (j < n):
          suffix = 'v%d(t)' % (j+1) #%d is a placeholder for j+1
          names += [suffix]
          j += 1
        count += 1
      else:
        j = 0
        while (j < n):
          suffix = 'v%d(t+%d)' % (j+1, count)
          names += [suffix]
        count += 1

    #combine everything
    comb = concat(columns, axis=1)
    comb.columns = names
    # drop rows with null values
    if dropnan:
      comb.dropna(inplace=True)
    return comb
  
  def plotNetwork(self, train_X, train_Y, test_X, test_y, lr, e, b, r):
    model = Sequential()
    #to prevent overfitting, we use regularizer, dropout
    Regularizer = l2(r)
    #add a LSTM layer with 100 units, expecting the corresponding input and output size
    model.add(LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2]),kernel_regularizer=Regularizer ,
      recurrent_regularizer=Regularizer , bias_regularizer=Regularizer , activity_regularizer=Regularizer, dropout=0.2, recurrent_dropout=0.3))
    model.add(keras.layers.Dense(1,activation=tf.keras.activations.softsign))
    opt = keras.optimizers.Adam(lr)
    model.compile(loss='mae', optimizer=opt, metrics=['accuracy'])
    score, accuracy = model.evaluate(self.test_X, self.test_y, verbose=0)    
    # fit network
    history = model.fit(self.train_X, self.train_y, epochs=e, batch_size=b, validation_data=(self.test_X, self.test_y), verbose=2, shuffle=False)
    # plot history
    plt.title('model loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    textstr = ('epochs=' + str(e) + '\n' + 'batch_size=' + str(b) + '\n' + 'regularizer rate=' + str(r) + '\n' + 'learning rate='
      + str(lr) + '\n' + 'regularizer=' + str(r))
    xmin, xmax, ymin, ymax = plt.axis() #get the max value of y_axis so we can place parameter information at the approapriate position
    plt.text(0.5*e, 0.7*ymax, textstr)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

    return model
  
  def predict(self, model, test_X,  test_y):
    y_pred = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
    # invert scaling for forecast
    inv_y_pred = concatenate((y_pred, test_X[:, :-1]), axis=1)
    inv_y_pred = self.scaler.inverse_transform(inv_y_pred)
    inv_y_pred = inv_y_pred[:,0]
    # invert scaling
    test_y = test_y.reshape((len(test_y), 1))
    original_y = concatenate((test_y, test_X[:, :-1]), axis=1)
    original_y = self.scaler.inverse_transform(original_y)
    original_y = original_y[:,0]

    return [original_y, inv_y_pred]
    

if __name__== "__main__":
    url = "https://personal.utdallas.edu/~nxc180007/Book2.csv"
    learning_rate = float(sys.argv[1])
    epochs = int(sys.argv[2])
    batch_size = int(sys.argv[3])
    regularizer = float(sys.argv[4])
    recurrent_network = RecurrentNeuralNet(url, learning_rate, epochs, batch_size, regularizer)
    
    

