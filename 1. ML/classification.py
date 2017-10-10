#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 12:40:32 2017

@author: user
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('../data/user_visit_duration.csv')
X=df[['Time (min)']].values
y=df['Buy'].values

plt.scatter(x=df['Time (min)'], y=df['Buy'], data=df)

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

model=Sequential()
model.add(Dense(units=1, input_shape=(1,), activation='sigmoid'))
model.compile(SGD(lr=0.1), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=50)

plt.scatter(x=df['Time (min)'], y=df['Buy'], data=df)
temp=np.linspace(0, 4)
plt.plot(temp, model.predict(temp), color='red')

temp_class=model.predict(temp)>0.5

plt.scatter(x=df['Time (min)'], y=df['Buy'], data=df)
temp=np.linspace(0, 4)
plt.plot(temp, temp_class, color='red')

y_pred=model.predict(X)
y_class_pred=y_pred>0.5

from sklearn.metrics import accuracy_score

score=accuracy_score(y, y_class_pred)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
params=model.get_weights()
params=[np.zeros(w.shape) for w in params]
model.set_weights(params)

model.fit(X_train, y_train, epochs=50, verbose=0)
score_train=accuracy_score(y_train, model.predict(X_train)>0.5)
score_test=accuracy_score(y_test, model.predict(X_test)>0.5)