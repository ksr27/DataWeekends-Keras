# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 09:39:33 2017

@author: Stefan Draghici
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('../data/weight-height.csv')

X=df[['Height']].values
y_true=df['Weight'].values

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, SGD

model=Sequential()
model.add(Dense(units=1, input_shape=(1,)))

model.summary()

model.compile(Adam(lr=0.2), 'mean_squared_error')
model.fit(X, y_true, epochs=50)

y_pred=model.predict(X)

plt.scatter(x=X, y=y_true, data=df)
plt.plot(X, y_pred, color='red')

weights, biases=model.get_weights()