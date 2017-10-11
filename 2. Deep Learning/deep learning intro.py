# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 23:03:45 2017

@author: Stefan Draghici
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets import make_moons

X, y=make_moons(n_samples=1000, noise=0.1, random_state=0)
plt.plot(X[y==0, 0], X[y==0, 1], 'ob', alpha=0.5)
plt.plot(X[y==1, 0], X[y==1, 1], 'xr', alpha=0.5)
plt.legend(['0', '1'])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam

# Shallow model
model=Sequential()
model.add(Dense(units=1, input_shape=(2, ), activation='sigmoid'))
model.compile(Adam(lr=0.05), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=200, verbose=0)
results=model.evaluate(X_test, y_test)

# Deep model
deep_model=Sequential()
deep_model.add(Dense(4, input_shape=(2,), activation='tanh'))
deep_model.add(Dense(2, activation='tanh'))
deep_model.add(Dense(1, activation='sigmoid'))
deep_model.compile(Adam(lr=0.05), loss='binary_crossentropy', metrics=['accuracy'])
deep_model.fit(X_train, y_train, epochs=200, verbose=0)
results=deep_model.evaluate(X_test, y_test)

y_train_pred=deep_model.predict_classes(X_train)
y_test_pred=deep_model.predict_classes(X_test)
