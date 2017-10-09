# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 14:36:24 2017

@author: Stefan Draghici
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('titanic-train.csv')

df.head()
df.info()
df.describe()

# indexing
df.iloc[3]
df.loc[0:4, 'Ticket']
df['Ticket'].head()
df[['Embarked', 'Ticket']].head()

# selections
df[df['Age']>70]
df['Age']>70
df.query('Age > 70')
df[(df['Age']==11) & (df['SibSp']==5)]
df[(df['Age']==11) | (df['SibSp']==5)]
df.query("(Age==11) | (SibSp==5)")
df['Embarked'].unique()

# sorting
df.sort_values('Age', ascending=False)

# aggregations
df['Survived'].value_counts()