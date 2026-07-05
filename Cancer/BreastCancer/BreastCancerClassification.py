import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import recall_score

from scipy.stats import ttest_ind


# config.
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
sns.set_theme(style='darkgrid')
RANDOM_STATE = 42

df = pd.read_csv('breast_cancer_dataset.csv')
df.shape
df.describe()
df.info()
df.isnull().sum()


for col in df.columns:
    print(df[col].value_counts().head())

df.duplicated().sum()

len(df.columns)
df['diagnosis'].value_counts(normalize=True)




fig, axes = plt.subplots(5,7, figsize=(15,18))
axes = axes.flatten()
for i,col in enumerate(df.columns):
    sns.histplot(df[col], ax = axes[i], kde=True)
plt.show()



fig , axes = plt.subplots(5,7, figsize=(15,18)) 
axes = axes.flatten()

for i, col in enumerate(df.columns) :
    sns.boxplot(df[col], ax = axes[i]) 
    
    

df.duplicated().sum()
df.drop(columns = ['id'], inplace = True)

df['diagnosis'] = df['diagnosis'].map({'M' : 1 , 'B': 0})


# solving target sample problem
sns.heatmap(df.corr()) 
# there is a REAL problem with correlation in our dataset
m = df.groupby('diagnosis').mean()
m.iloc[1]
ttest_ind(m.iloc[0], m.iloc[1])


# Data Preprocessing

X = df.drop(columns = 'diagnosis')
y = df['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X , y , random_state=RANDOM_STATE, stratify=y)



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



#####################################
### fitting a Logistic Regression ###
#####################################

model2 = LogisticRegression()
model2.fit(X_train_st , y_train)

y_pred2 = model2.predict(X_test_st)
y_train_pred2 = model2.predict(X_train_st)

accuracy_score(y_test, y_pred2)
accuracy_score(y_train, y_train_pred2)

model2.score(X_train_st, y_train)
model2.score(X_test_st, y_test)


#####################################
#####################################




# model Optimization
# Pipeline Creation

Pip = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("pca", PCA()),
        ("model", SVC(kernel = 'rbf'))
    ]
)

# grid of hyperparameter
param_grid = [
    {
        "pca__n_components": [10, 15, 20, 30], 
        "model__C": [0.1, 1, 10, 100, 1000],
        "model__gamma": [0.001, 0.01, 0.1, 1, "scale"]
    }
]


# Cross Validation setup
# (used inside grid search)
k = 5
cv = StratifiedKFold(n_splits=k , shuffle=True, random_state=RANDOM_STATE)

grid = GridSearchCV(
    estimator=Pip,
    param_grid= param_grid,
    scoring= 'recall',
    cv = cv,
    n_jobs= -1,
    
)

grid.fit(X_train, y_train)

grid.best_params_
grid.best_score_

# train with best params
best_Pip = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components= 15)),
        ("model", SVC(kernel = 'rbf', C= 100, gamma=0.01))
    ]
)

best_Pip.fit(X_train, y_train)
y_train_pred_best_param = best_Pip.predict(X_train)
y_test_pred_best_param = best_Pip.predict(X_test)
recall_score(y_test_pred_best_param, y_test)




cm = confusion_matrix(y_test, y_test_pred_best_param)
sns.heatmap(cm, annot=True, cmap= 'Reds', 
            xticklabels= ['pred b', 'pred m'],
            yticklabels= ['actual b', 'actual m'])
print(classification_report(y_test, y_test_pred_best_param))

cm = confusion_matrix(y_train, y_train_pred_best_param)
sns.heatmap(cm, annot=True, cmap= 'Reds', 
            xticklabels= ['pred b', 'pred m'],
            yticklabels= ['actual b', 'actual m'],
            fmt = 'd')
print(classification_report(y_train, y_pred=y_train_pred_best_param))




# building Predictive System
def PredictCancer(input_features):
    input_df = pd.DataFrame([input_features], columns=X_train.columns)
    predictions = best_Pip.predict(input_df)
    if predictions[0] == 1:
        print('you have cancer')
    else:
        print('not cancer')

y_test
sample_1 = X_test.iloc[83].tolist()
PredictCancer(sample_1)









