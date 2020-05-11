#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 10 13:05:01 2020

@author: tonatiuh
"""

import pandas as pd
from sklearn import preprocessing

import h2o
from h2o.automl import H2OAutoML
from h2o.estimators import H2OGradientBoostingEstimator
from h2o.estimators import H2OSupportVectorMachineEstimator

h2o.init()
h2o.remove_all

#READ DATA
file_train = 'titanic_kaggle/train.csv'
data_raw = pd.read_csv(file_train)

#TRANSFORM
data_raw = data_raw[['Survived', 'Age', 'Pclass', 'Sex']]
data_raw = data_raw.dropna()
enc = preprocessing.OrdinalEncoder()
enc.fit(data_raw[['Sex']])
transform_data = enc.transform(data_raw[['Sex']])
data_raw['sex_enc'] = transform_data
data_smart = data_raw[['Survived', 'Age', 'Pclass', 'sex_enc']]
data_smart = data_smart.reset_index(drop=True)


#SPLIT DATA
train = data_smart.iloc[:600, :]
train_h2o = h2o.H2OFrame(train)
x = train_h2o.columns
y = 'Survived'
x.remove(y)
train_h2o[y] = train_h2o[y].asfactor()

test = data_smart.iloc[600:, :]
test_h2o = h2o.H2OFrame(test)
test_h2o[y] = test_h2o[y].asfactor()

#TRAINING
aml = H2OAutoML(max_models=20, seed=1)
aml.train(x=x, y=y, training_frame=train_h2o)
lb = aml.leaderboard

#PREDICTION
preds = aml.predict(test_h2o)
accuracy_score(test.iloc[:, 0], preds_tmp.iloc[:, 0])




# split into train and validation sets
train_ab, valid = train_h2o.split_frame(ratios = [.8], seed = 1234)
# Build and train the model:
pros_gbm = H2OGradientBoostingEstimator(nfolds=5,
                                        seed=1,
                                        keep_cross_validation_predictions = True,
                                        stopping_metric='AUCPR',
                                        ntrees=60,
                                        max_depth=2)
pros_gbm.train(x=x, y=y, training_frame=train_ab, validation_frame = valid)
pred = pros_gbm.predict(test_h2o)
pred_tmp = pred.as_data_frame()
accuracy_score(test.iloc[:, 0], pred_tmp.iloc[:, 0])



# Build and train the model:
svm_model = H2OSupportVectorMachineEstimator(gamma=0.1,
                                             rank_ratio = 0.3,
                                             disable_training_metrics = True)
svm_model.train(x=x, y=y, training_frame=train_h2o)
pred = svm_model.predict(test_h2o)
pred_tmp = pred.as_data_frame()
accuracy_score(test.iloc[:, 0], pred_tmp.iloc[:, 0])