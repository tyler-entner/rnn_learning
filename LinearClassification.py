# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 10:41:45 2021

@author: Tyler
"""


#Import tensorflow and ensure correct version
import tensorflow as tf
print(tf.__version__)

#Import breast cancer dataset for exploration
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
type(data)

#Examine target distribution
data.target

#Understand target classification
data.target_names

#Determine shape of dataset
data.data.shape

#Ensure targets match the shape of data
data.target.shape

#What features do we have? 
data.feature_names

#Split data into train and testing sets
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size = 0.33)
N, D = x_train.shape

#Standardize all input features
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#Build tensorflow Linear Classification model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(1, activation ='sigmoid', input_shape = (D,)))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

r = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = 100)
print("train score:", model.evaluate(x_train, y_train))
print("test score:", model.evaluate(x_test, y_test))

#Plot loss per iteration to examine fitting process
import matplotlib.pyplot as plt

plt.plot(r.history["loss"], label = "loss")
plt.plot(r.history["val_loss"], label = "val_loss")
plt.legend()

#Plot accuracy per iteration
plt.plot(r.history["accuracy"], label = "acc")
plt.plot(r.history["val_accuracy"], label = "val_acc")
plt.legend()