import pandas as pd
import matplotlib.pyplot as plt 

from sklearn import preprocessing
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


#READ DATA
file_train = 'titanic_kaggle/train.csv'
data_train = pd.read_csv(file_train)

#EXPLORE
#print(data_train.head())
#print(data_train.count())

'''
plt.figure()
plt.hist(data_train['Age'], bins=range(1, int(max(data_train['Age'])) + 1))

plt.figure()
plt.hist(data_train['Pclass'])

'''
#print(data_train.describe())


#TRANSFORM
data_raw = data_train[['Survived', 'Age', 'Pclass', 'Sex']]
data_raw = data_raw.dropna()
enc = preprocessing.OrdinalEncoder()
enc.fit(data_raw[['Sex']])
transform_data = enc.transform(data_raw[['Sex']])
data_raw['sex_enc'] = transform_data
data_smart = data_raw[['Survived', 'Age', 'Pclass', 'sex_enc']]
data_smart = data_smart.reset_index(drop=True)

#CLASIFICATION
train = data_smart.iloc[:600, :]
test = data_smart.iloc[600:, :]

clf_dt = tree.DecisionTreeClassifier()
clf_dt = clf_dt.fit(train.iloc[:, 1:], train.iloc[:, 0])
y_pred_dt = clf_dt.predict(test.iloc[:, 1:])

clf_rf = RandomForestClassifier(n_estimators=20)
clf_rf = clf_rf.fit(train.iloc[:, 1:], train.iloc[:, 0])
y_pred_rf = clf_rf.predict(test.iloc[:, 1:])

clf_gb = GradientBoostingClassifier()
clf_gb = clf_gb.fit(train.iloc[:, 1:], train.iloc[:, 0])
y_pred_gb = clf_gb.predict(test.iloc[:, 1:])

#METRICS
y_true = test.iloc[:, 0]
print('DT')
tn, fp, fn, tp = confusion_matrix(y_true, y_pred_dt).ravel()
print(tn, fp, fn, tp)
print(accuracy_score(y_true, y_pred_dt))
print('')

print('RF')
tn, fp, fn, tp = confusion_matrix(y_true, y_pred_rf).ravel()
print(tn, fp, fn, tp)
print(accuracy_score(y_true, y_pred_rf))
print('')

print('GB')
tn, fp, fn, tp = confusion_matrix(y_true, y_pred_gb).ravel()
print(tn, fp, fn, tp)
print(accuracy_score(y_true, y_pred_gb))
print('')