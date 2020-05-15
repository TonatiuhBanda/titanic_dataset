#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 16:42:55 2020

@author: tonatiuh
"""

import pandas as pd

from sklearn import preprocessing
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#READ DATA
file_train = 'titanic_kaggle/train.csv'
data_org = pd.read_csv(file_train)


#TRANSFORM
data_raw = data_org[['Survived', 'Age', 'Pclass', 'Sex']]
data_raw = data_raw.dropna()

enc = preprocessing.OrdinalEncoder()
enc.fit(data_raw[['Sex']])
transform_data = enc.transform(data_raw[['Sex']])
data_raw['sex_enc'] = transform_data
data_smart = data_raw[['Survived', 'Age', 'Pclass', 'sex_enc']]
data_smart = data_smart.reset_index(drop=True)


#CLASIFICATION
train = data_smart.iloc[:600, :]
x_train = train.iloc[:, 1:]
y_train = train.iloc[:, 0]

test = data_smart.iloc[600:, :]
x_test = test.iloc[:, 1:]
y_test = test.iloc[:, 0]

classifiers = {
    'dt': {'model':tree.DecisionTreeClassifier()},
    'rf': {'model':RandomForestClassifier(n_estimators=30, random_state=12)},
    'gb': {'model':GradientBoostingClassifier()}
    }

for key in classifiers:
    classifiers[key]['model'].fit(x_train, y_train)
    classifiers[key]['y_pred'] = classifiers[key]['model'].predict(x_test)
    
    #METRICS
    print(key)
    tn, fp, fn, tp = confusion_matrix(y_test, classifiers[key]['y_pred']).ravel()
    print(tn, fp, fn, tp)
    print(accuracy_score(y_test, classifiers[key]['y_pred']))
    print('')
