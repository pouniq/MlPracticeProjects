import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    average_precision_score
)


# config. 
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: f'{x:.3f}')
sns.set_theme(style='darkgrid')
RANDOM_STATE = 42
######################################

df = pd.read_csv('creditcard.csv')



df.head()
df.isnull().sum()
df.info()
df.describe()
df.groupby('Class').mean()

for col in df.columns:
    print(df[col].value_counts())
    
df.loc[df.duplicated()]
# df = df.drop_duplicates()
df['Class'].value_counts(normalize=True )

df[['Time', 'Amount']].describe()


# there is HUGE imbalanced class
plt.figure(figsize=(4,2))
sns.countplot(df , x = "Class")
plt.show()



plt.figure(figsize=(10,7))
sns.histplot(df["Amount"], bins = 60, kde=True)


plt.figure(figsize=(10,7))
sns.histplot(np.log1p(df["Amount"]), bins = 60, kde=True)


plt.figure(figsize=(10,7))
sns.histplot(df['Time'], kde= True)
plt.show()


sns.boxplot(x = 'Class', y = 'Amount', data = df)

sns.heatmap(df.corr(numeric_only= True), center= 0)
# we do not see any highly correlated columns, because the PCA
# is already applied


## data preprocessing
df['Amount_log'] = np.log1p(df['Amount'])
df.columns

X = df.drop(columns = ['Class', 'Amount']) 
# removing Amount as we already have log_amount
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    stratify=y,
    test_size=0.2,
    random_state=RANDOM_STATE
)


# baseline model
base_pip = Pipeline(
    [
        ('model', DecisionTreeClassifier())
        
    ]
)

base_pip.fit(X_train, y_train)

y_base_train_predicted = base_pip.predict(X_train)
y_base_test_predicted = base_pip.predict(X_test)

print(classification_report(y_test, y_base_test_predicted))
accuracy_score(y_base_test_predicted, y_test)


## NOTE: good learning happen here
y_train_prob = base_pip.predict_proba(X_train)[:, 1]
print(f'training average precision  score {average_precision_score(y_train, y_train_prob)}')

y_test_prob = base_pip.predict_proba(X_test)[:, 1]
print(f'test average precision  score {average_precision_score(y_test, y_test_prob)}')
# the model is as good as predicting RANDOMLY





## MODEL SELECTION & OPTIMAZIATION
k = 5
cv = StratifiedKFold(n_splits=k , shuffle= True, random_state= RANDOM_STATE)
models = {
    'dt_balanced': Pipeline(
        [
            ('model', DecisionTreeClassifier(
                random_state= RANDOM_STATE,
                class_weight= 'balanced'
            ))
        ]
    ),
    
    'hgb_balanced': Pipeline(
        [
            ('model', HistGradientBoostingClassifier(
                random_state= RANDOM_STATE,
                class_weight= 'balanced'
            ))
        ]
    ),
    'rf_balanced': Pipeline(
        [
            ('model', RandomForestClassifier(
                random_state= RANDOM_STATE,
                class_weight= 'balanced'
            ))
        ]
    )
}


for name, m in models.items():
    
   pr_auc = []
   
   for tr_idx, te_idx in cv.split(X_train, y_train):
        X_tr, X_te = X_train.iloc[tr_idx] , X_train.iloc[te_idx]
        y_tr, y_te = y_train.iloc[tr_idx] , y_train.iloc[te_idx]
       
       
        m.fit(X_tr, y_tr)
        pred_prob = m.predict_proba(X_te)[:, 1]
    
        pr_auc.append(average_precision_score(y_te, pred_prob))

print("model name", name)
print(pr_auc)
print(round(float(np.mean(pr_auc)), 4))


## best model ######

### Hyper Parameter Tuning :
rfc_pipeline = Pipeline(
    [
        ('model', RandomForestClassifier(
            random_state= RANDOM_STATE,
            n_jobs= -1,
            class_weight="balanced_subsample"
            ))
    ]
)


param_dist = {
    'model__n_estimators': [200, 300, 400],
    'model__max_depth': [None, 10, 20, 30],
    'model__min_samples_leaf': [1, 5, 10, 20],
    'model__max_features': ['sqrt', 'log2', 0.5]
}
      
      
#‌ random search   
random_search = RandomizedSearchCV(
    estimator= rfc_pipeline,
    param_distributions= param_dist,
    n_iter= 10,
    scoring="average_precision", # PR-AUC
    cv = cv,
    verbose=True
)

random_search.fit(X_train, y_train)


best_rfc = Pipeline(
    [
        ('model', RandomForestClassifier(
            min_samples_leaf=5,
            max_depth=20,
            max_features='log2',
            class_weight='balanced_subsample',
            n_jobs=-1,
            n_estimators=500,
        ))
    ]
)

best_rfc.fit(X_train, y_train)

y_train_pred = best_rfc.predict(X_train)
train_acc = accuracy_score(y_train_pred, y_train)
print(classification_report(y_train, y_train_pred))

train_prob = best_rfc.predict_proba(X_train)[:,1]

y_test_pred = best_rfc.predict(X_test)
test_acc = accuracy_score(y_test_pred, y_test)

confusion_matrix(y_test_pred, y_test)

test_proba = best_rfc.predict_proba(X_test)[:, 1]

## predictive system:
def predictive(input_features):
    df = pd.DataFrame(
        [input_features],
        columns=X_train.columns)
    
    prediction = best_rfc.predict(df)
    print(prediction)
    if prediction[0] == 1:
        print('the transaction was fraud')
    else:
        print('Normal transaction')


X_test.head()
y_test[y_test == 1]

predictive(X_test.loc[143333].tolist())
 
        
    
