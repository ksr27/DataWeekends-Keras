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

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.2, random_state=0)

weights[0,0]=0.0
biases[0]=0.0
model.set_weights((weights, biases))

model.fit(X_train, y_train, epochs=50, verbose=0)

y_train_pred=model.predict(X_train)
y_test_pred=model.predict(y_test)

from sklearn.metrics import mean_squared_error as mse

mse_train=mse(y_train, y_train_pred)
mse_test=mse(y_test, y_test_pred)