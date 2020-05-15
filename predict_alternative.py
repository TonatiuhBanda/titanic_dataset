#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 23:14:48 2020

@author: tonatiuh
"""

import pandas as pd

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


#READ DATA
file_train = 'titanic_kaggle/train.csv'
data_org = pd.read_csv(file_train)


#TRANSFORM
data_raw = data_org[['Survived', 'Age', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Fare']]
data_raw = data_raw.dropna()

enc = preprocessing.OrdinalEncoder()
enc.fit(data_raw[['Sex']])
transform_data = enc.transform(data_raw[['Sex']])
data_raw['sex_enc'] = transform_data
data_smart = data_raw[['Survived', 'Age', 'Pclass', 'sex_enc', 'SibSp', 'Parch', 'Fare']]
data_smart = data_smart.reset_index(drop=True)

#SCALER
scaler = MinMaxScaler()
scaler.fit(data_smart)
data_scaled = scaler.transform(data_smart)
df_data = pd.DataFrame(data_scaled, columns=data_smart.columns)


#TRAIN-TEST
X = df_data.iloc[:, 1:]
y = df_data.iloc[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


#CLASIFICATION
classifiers = {
    'dt': {'model':tree.DecisionTreeClassifier()},
    'rf': {'model':RandomForestClassifier(n_estimators=30, random_state=12)},
    'gb': {'model':GradientBoostingClassifier()}
    }

for key in classifiers:
    classifiers[key]['model'].fit(X_train, y_train)
    classifiers[key]['y_pred'] = classifiers[key]['model'].predict(X_test)
    
    #METRICS
    print(key)
    tn, fp, fn, tp = confusion_matrix(y_test, classifiers[key]['y_pred']).ravel()
    print(tn, fp, fn, tp)
    print(accuracy_score(y_test, classifiers[key]['y_pred']))
    print('')

