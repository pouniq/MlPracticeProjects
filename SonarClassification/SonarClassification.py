# Setup


## import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

## config.
pd.set_option('display.max_columns', None)


## read dataset
df = pd.read_csv('sonar.csv', header=None)
len(df.columns)
df.columns


## EDA:
### basic dataset overview:

df.describe()
df.info()
df.duplicated().sum()

df.shape

### encoded missing values
for col in df.columns:
    print(df[col].value_counts())
### -- we do not have missing values (in strange ways)

### class distribution:
# -- if we have so different values in our target column then we need to apply
# some methods like undersample, oversample , etc


plt.figure(figsize=(10,7))
plt.scatter(df[2], df[12])

X = df.drop(columns = 60)
y = df[60]

df.corr()
