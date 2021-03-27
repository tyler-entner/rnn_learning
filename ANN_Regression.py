# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 12:43:32 2021

@author: Tyler
"""


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Make dataset for testing
N = 1000
x = np.random.random((N,2)) * 6 - 3
y = np.cos(2*x[:,0]) + np.cos(3*x[:,1])

#Plot dataset for visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(x[:,0], x[:,1], y)

#Build model for training
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(128, input_shape = (2,), activation = 'relu'),
  #tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(1)
])

#Compile the model
opt = tf.keras.optimizers.Adam(0.01)
model.compile(optimizer = opt, loss = 'mse')
r = model.fit(x, y, epochs = 100)

#Visualize loss
plt.plot(r.history['loss'], label = 'loss')

#Plot prediction surface
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(x[:,0], x[:,1], y)

#Surface plot
line = np.linspace(-3, 3, 50)
xx, yy = np.meshgrid(line, line)
xgrid = np.vstack((xx.flatten(), yy.flatten())).T
yhat = model.predict(xgrid).flatten()
ax.plot_trisurf(xgrid[:,0], xgrid[:,1], yhat, linewidth = 0.2, antialiased = True)
plt.show()

#Test extrapolation
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(x[:,0], x[:, 1], y)

line = np.linspace(-5, 5, 50)
xx, yy = np.meshgrid(line, line)
xgrid = np.vstack((xx.flatten(), yy.flatten())).T
yhat = model.predict(xgrid).flatten()
ax.plot_trisurf(xgrid[:,0], xgrid[:,1], yhat, linewidth = 0.2, antialiased = True)
plt.show()

#Shows that ANN cannot extrapolate past what it is trained on due to non periodic activation function