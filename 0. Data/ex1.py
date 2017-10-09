# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 17:14:56 2017

@author: Stefan Draghici
"""

import pandas as pd

df=pd.read_csv('international-airline-passengers.csv')
df['Month']=pd.to_datetime(df['Month'])
df=df.set_index('Month')

df.plot()