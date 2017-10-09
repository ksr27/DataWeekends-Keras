# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 17:14:56 2017

@author: Stefan Draghici
"""

import pandas as pd

df=pd.read_csv('weight-height.csv')

df.plot(kind='scatter', x='Height', y='Weight')