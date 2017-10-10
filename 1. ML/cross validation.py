#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 13:35:40 2017

@author: user
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('../data/user_visit_duration.csv')
X=df[['Time (min)']].values
y=df['Buy'].values

from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

def build_logistic_regression_model():
    model=Sequential()
    model.add(Dense(units=1, input_shape=(1, ), activation='sigmoid'))
    model.compile(SGD(lr=0.05), loss='binary_crossentropy', metrics=['accuracy'])
    return model

model=KerasClassifier(build_fn=build_logistic_regression_model, epochs=50, verbose=0)

from sklearn.model_selection import cross_val_score, KFold

cross_validation=KFold(n_splits=5, shuffle=True)
scores=cross_val_score(estimator=model, X=X, y=y, cv=cross_validation)