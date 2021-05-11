# -*- coding: utf-8 -*-
"""
Created on Tue May 11 10:55:55 2021

@author: xxpit
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, SimpleRNN, GRU, LSTM, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Create synthetic data for evaluation
series = np.sin((0.1*np.arange(400))**2)
plt.plot(series)
plt.show()

#Build dataset 
T = 10
D = 1
X = []
Y = []

for t in range(len(series) - T):
  x = series[t:t+T]
  X.append(x)
  y = series[t+T]
  Y.append(y)

X = np.array(X).reshape(-1,T)
Y = np.array(Y) 
N = len(X)
print("X.shape", X.shape, "Y.shape", Y.shape)


#Build autoregressive linear model
#input -> dense -> Model -> compile -> fit

i = Input(shape = (T,))
x = Dense(1)(i)
model = Model(i, x)
model.compile(
    loss = 'mse',
    optimizer = Adam(lr=0.01)
)

r = model.fit(
    X[:-N//2], Y[:-N//2],
    epochs = 80, 
    validation_data = (X[-N//2:], Y[-N//2:])
)

#Plot loss per iteration
plt.plot(r.history['loss'], label = 'loss')
plt.plot(r.history['val_loss'], label = 'val_loss')
plt.legend()

#One-step forecast using true targets
outputs = model.predict(X)
print(outputs.shape)
predictions = outputs[:,0]

plt.plot(Y, label = 'targets')
plt.plot(predictions, label = 'predictions')
plt.title('Linear Regression Predictions')
plt.legend()
plt.show()

#RNN/LSTM Model
X = X.reshape(-1, T, 1)

#RNN
#input, rnn, dense, model, compile, fit
i = Input(shape = (T,D))
x = SimpleRNN(10)(i)
x = Dense(1)(x)
model = Model(i,x)
model.compile(
    loss = 'mse',
    optimizer = Adam(lr=0.05)
)

r = model.fit(
    X[:-N//2], Y[:-N//2],
    batch_size = 32,
    epochs = 200,
    validation_data = (X[-N//2:], Y[-N//2:])
)

#Plot loss
plt.plot(r.history['loss'], label = 'loss')
plt.plot(r.history['val_loss'], label = 'val_loss')
plt.legend()
plt.show()

#Predictions
outputs = model.predict(X)
print(outputs.shape)
predictions = outputs[:,0]

plt.plot(Y, label = 'targets')
plt.plot(predictions, label = 'predictions')
plt.title('many-to-one RNN')
plt.legend()
plt.show()

#LSTM Model
#RNN/LSTM Model
X = X.reshape(-1, T, 1)

#RNN
#input, rnn, dense, model, compile, fit
i = Input(shape = (T,D))
x = LSTM(10)(i)
x = Dense(1)(x)
model = Model(i,x)
model.compile(
    loss = 'mse',
    optimizer = Adam(lr=0.05)
)

r = model.fit(
    X[:-N//2], Y[:-N//2],
    batch_size = 32,
    epochs = 200,
    validation_data = (X[-N//2:], Y[-N//2:])
)

#Plot loss
plt.plot(r.history['loss'], label = 'loss')
plt.plot(r.history['val_loss'], label = 'val_loss')
plt.legend()
plt.show()

#Predictions
outputs = model.predict(X)
print(outputs.shape)
predictions = outputs[:,0]

plt.plot(Y, label = 'targets')
plt.plot(predictions, label = 'predictions')
plt.title('LSTM')
plt.legend()
plt.show()