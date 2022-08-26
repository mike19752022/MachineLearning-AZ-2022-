# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importar el dataset
dataset=pd.read_csv('Data.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,3].values

#Tratamientos de los NAs
from sklearn.impute import SimpleImputer
#imputer= SimpleImputer(missing_values="NaN",strategy="mean",axis=0)
imputer=SimpleImputer(missing_values=np.nan, strategy='mean', fill_value=None, verbose=0, copy=True, add_indicator=False)
imputer=imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])

#Codificar datos categoricos
from sklearn import preprocessing
labelencoder_X=preprocessing.LabelEncoder()
X[:,0]=labelencoder_X.fit_transform(X[:,0])

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct=ColumnTransformer(
    [('one_hot_encoder',OneHotEncoder(categories='auto'),[0])],
    remainder='passthrough')
X=np.array(ct.fit_transform(X), dtype=np.float64)

labelencoder_y=preprocessing.LabelEncoder()
y=labelencoder_y.fit_transform(y)