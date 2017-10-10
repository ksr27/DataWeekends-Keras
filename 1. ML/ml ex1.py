#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 14:54:26 2017

@author: user
"""

import pandas as pd

dataset=pd.read_csv('../data/housing-data.csv')

import matplotlib.pyplot as plt

plt.hist(x=dataset['sqft'])
plt.hist(x=dataset['bdrms'])
plt.hist(x=dataset['age'])
plt.hist(x=dataset['price'])

X=dataset[['sqft', 'bdrms', 'age']].values
y=dataset['price'].values

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

model=Sequential()
model.add(Dense(units=1, input_shape=(3,)))
model.compile(Adam(lr=0.5), 'mean_squared_error')

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model.fit(X_train, y_train)

dataset['sqft1000']=dataset['sqft']/1000.0
dataset['age10']=dataset['age']/10.0
dataset['price100k']=dataset['price']/1e5

X=dataset[['sqft1000', 'bdrms', 'age10']].values
y=dataset['price100k'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model=Sequential()
model.add(Dense(units=1, input_shape=(3,)))
model.compile(Adam(lr=0.5), 'mean_squared_error')
model.fit(X_train, y_train, epochs=50)