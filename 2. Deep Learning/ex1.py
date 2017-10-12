# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
df=pd.read_csv('../data/diabetes.csv')

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

X=sc.fit_transform(df.drop('Outcome', axis=1))
y=df['Outcome'].values

from keras.utils.np_utils import to_categorical
y_cat=to_categorical(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)