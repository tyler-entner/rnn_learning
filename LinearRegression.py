# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 11:31:24 2021

@author: Tyler
"""


#Import tensorflow and ensure correct version
import tensorflow as tf
print(tf.__version__)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Get data
!wget https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/tf2.0/moore.csv

#Load in data
data = pd.read_csv('moore.csv', header = None).values
x = data[:,0].reshape(-1,1) #make 2D array of NxD D==1
y = data[:,1]

#Explore data
plt.scatter(x,y)

#Does log transform make linear?
plt.scatter(x, np.log(y))

#Center the data so values aren't too large
x = x - x.mean()

#Create tensorflow model

model = tf.keras.models.Sequential([
  tf.keras.layers.Input(shape = (1,)),
  tf.keras.layers.Dense(1)
])
model.compile(optimizer = tf.keras.optimizers.SGD(0.001, 0.9), loss = "mse")

#model = tf.keras.models.Sequential()
#model.add(tf.keras.layers.Dense(1, input_shape = (1,)))
#model.compile(optimizer = tf.keras.optimizers.SGD(0.001, 0.9), loss = 'mse')

#create learning rate scheduler

def lr_schedule(epoch, lr):
  if epoch >= 50:
    return 0.0001
  return 0.001

scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

#Train the model
r = model.fit(x, y, epochs = 200, callbacks = [scheduler])



#Plot training data
plt.plot(r.history["loss"], label = "loss")
plt.legend()

#Determine slope of fitted line
print(model.layers[0].get_weights())
a = model.layers[0].get_weights()[0][0,0]



