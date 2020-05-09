import pandas as pd
import matplotlib.pyplot as plt 


#READ DATA
file_train = 'titanic_kaggle/train.csv'
data_raw = pd.read_csv(file_train)

#EXPLORE
print(data_raw.describe())
print(data_raw.columns)

plt.figure()
plt.hist(data_raw['Age'], bins=range(1, int(max(data_raw['Age'])) + 1))

plt.figure()
plt.hist(data_raw['Pclass'])

plt.figure()
plt.hist(data_raw['Sex'])
