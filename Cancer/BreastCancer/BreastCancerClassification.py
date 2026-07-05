import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline



RANDOM_STATE = 42

df = pd.read_csv('breast_cancer_dataset.csv')
df.shape
df.describe()
df.info()
df.isnull().sum()

len(df.columns)
df['diagnosis'].value_counts(normalize=True)


fig, axes = plt.subplots(5,7, figsize=(15,18))
axes = axes.flatten()
for i,col in enumerate(df.columns):
    sns.histplot(df[col], ax = axes[i])
plt.show()



fig , axes = plt.subplots(5,7, figsize=(15,18)) 
axes = axes.flatten()

for i, col in enumerate(df.columns) :
    sns.boxplot(df[col], ax = axes[i]) 
    
    

df.duplicated().sum()
df.drop(columns = ['id'], inplace = True)

df['diagnosis'] = df['diagnosis'].map({'M' : 1 , 'B': 0})


# solving target sample problem
sns.heatmap(df.corr())``
# there is a REAL problem with correlation in our dataset
df.groupby('diagnosis').mean()


X = df.drop(columns = 'diagnosis')
y = df['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X , y , random_state=RANDOM_STATE, stratify=y)
X_train
X_test

st_scaler = StandardScaler()
X_train_st = st_scaler.fit_transform(X_train)
X_test_st = st_scaler.transform(X_test)

X_train_st

model1 = SVC()
model1.fit(X_train_st, y_train)

y_train_pred = model1.predict(X_train_st)
y_pred = model1.predict(X_test)



accuracy_score(y_train_pred, y_train )
accuracy_score(y_pred , y_test)

# explain the overfit problem 


model2 = LogisticRegression()
model2.fit(X_train_st , y_train)

y_pred2 = model2.predict(X_test_st)
y_train_pred2 = model2.predict(X_train_st)

accuracy_score(y_test, y_pred2)
accuracy_score(y_train, y_train_pred2)


