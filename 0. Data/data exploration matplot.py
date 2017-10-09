# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 15:01:28 2017

@author: Stefan Draghici
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data1=np.random.normal(0, 0.1, 1000)
data2=np.random.normal(1, 0.4, 1000)+np.linspace(0, 1, 1000)
data3=2+np.random.random(1000)+np.linspace(0, 5, 1000)
data4=np.random.normal(3, 0.2, 1000)+0.3+np.sin(np.linspace(0, 20, 1000))

data=np.stack([data1, data2, data3, data4]).transpose()

df=pd.DataFrame(data, columns=['data1', 'data2', 'data3', 'data4'])
df.head()

# line plot 1
df.plot(title='Line Plot')
# line plot 2
plt.plot(df)
plt.title('Line Plot')
plt.legend(['data1', 'data2', 'data3', 'data4'])

# scatter plot 1
df.plot(style='.')

# histogram
df.plot(kind='hist', bins=50, title='Histo', alpha=0.8)

# cumulative hist
df.plot(kind='hist', bins=100, title='Cumulative Histo', alpha=0.8, normed=True, cumulative=True)

# box plot
df.plot(kind='box')

# subplots
fig, ax=plt.subplots(2, 2)
df.plot(ax=ax[0][0], title='line plot')
df.plot(ax=ax[0][1], style='o', title='scatter plot')
df.plot(ax=ax[1][0], kind='hist', title='histogram')
df.plot(ax=ax[1][1], kind='box', title='box plot')