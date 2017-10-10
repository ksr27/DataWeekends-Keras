# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 08:50:17 2017

@author: Stefan Draghici
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('../data/weight-height.csv')
df.plot(kind='scatter', x='Height', y='Weight')

def line(x, weight=0.01, bias=00.1):
    return x*weight+bias

def mean_squared_error(y_true, y_pred):
    squared_error=(y_true-y_pred)**2
    return squared_error.mean()

X=df[['Height']].values
y_true=df['Weight'].values
y_pred=line(X)
msq=mean_squared_error(y_true, y_pred)