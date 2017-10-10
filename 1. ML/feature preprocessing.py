#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 14:19:09 2017

@author: user
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('../data/weight-height.csv')

pd.get_dummies(df['Gender'])

# rescale with fixed factor
df['Height (feet)']=df['Height']/12.0
df['Weight (100 lbs)']=df['Weight']/100.0

# min max normalization
from sklearn.preprocessing import MinMaxScaler

mms=MinMaxScaler()
df['Height mms']=mms.fit_transform(df[['Height']])
df['Weight mms']=mms.fit_transform(df[['Weight']])

# standard normalization
from sklearn.preprocessing import StandardScaler

ss=StandardScaler()
df['Height ss']=ss.fit_transform(df[['Height']])
df['Weight ss']=ss.fit_transform(df[['Weight']])