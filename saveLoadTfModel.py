# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 19:52:41 2021

@author: Tyler
"""


#To save a model
model.save('___.h5')

#Check that the model exists in colab
#!ls -lh

#Loading in a model:
model = tf.keras.models.load_model('linearclassifier.h5')
print(model.layers)