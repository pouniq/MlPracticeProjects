import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold, train_test_split, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer # for handling categorical values
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    root_mean_squared_error,
    r2_score
)


# config.
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: f'{x:.3f}')
sns.set_theme(style='darkgrid')

plt.rcParams.update(
    {
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8
    }
)

RANDOM_STATE = 42
CSV_PATH = 'housing.csv' 
TARGET_COL = 'median_house_value'


df = pd.read_csv(CSV_PATH)

df.shape
df.describe()
df.info()
df.value_counts(normalize=True)
df.columns
# target column: median_house_value



df.groupby(TARGET_COL).value_counts()

num_cols = df.select_dtypes(include= [np.number]).columns.tolist()
cat_cols = df.select_dtypes(include = ['object']).columns.tolist()


df.isna().sum()

df.duplicated().sum()


for col in df.columns:
    print(df[col].value_counts().head(20))


df[num_cols].describe().T



## Data visulization ##
for cat in cat_cols:
    plt.figure(figsize= (10,7))
    sns.countplot(x = df[cat])
    plt.title(f'Dist. of {col}')
    plt.show()
    
for cat in cat_cols:
    print(df[col].value_counts())
    

plt.figure(figsize=(10,7))
sns.histplot(df[TARGET_COL], bins = 40, kde=True)
plt.show()

df[TARGET_COL].value_counts()


## histogram plot -- distribution ##
# for numerical columns #
fig, axes = plt.subplots(3, 3, figsize=(10,7))
axes = axes.flatten()
for i, col in enumerate(df[num_cols]):
    sns.histplot(df[col], kde=True, ax = axes[i])
    axes[i].set_title(col, fontsize = 10)
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(3, 3, figsize=(10,7))
axes = axes.flatten()
for i, col in enumerate(df[num_cols]):
    sns.boxplot(x = df[col], ax = axes[i])
    axes[i].set_title(col, fontsize = 10)
plt.tight_layout()
plt.show()
# those numbers are based on real world 

sns.heatmap(df[num_cols].corr(), annot=True, center = 0, )

df[num_cols].corr()[TARGET_COL].sort_values(ascending = False)


## data preprocessing ##
X = df.drop(columns = [TARGET_COL])
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(X, y ,
                                                    test_size=0.2,
                                                    random_state=RANDOM_STATE)

X_train.shape
X_test.shape


## preprocessing pipeline
numerical_features = X.select_dtypes(include = [np.number]).columns.tolist()
categorical_features = X.select_dtypes(include = 'str').columns.tolist()
# categorical_features = X.select_dtypes(exclude = [np.number]').columns.tolist()


# numerical features -- preprocessing
numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
])

# categorical features -- preprocessing
cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocess = ColumnTransformer(

    transformers= [
        ('num', numeric_transformer, numerical_features),
        ('cat', cat_transformer, categorical_features )
    ]
)

## baseline model ##
baseline_pip = Pipeline(
    steps=[
        ('preprocess', preprocess),
        ('model', LinearRegression())
    ]
)

baseline_pip.fit(X_train, y_train)
train_baseline_pred = baseline_pip.predict(X_train)
root_mean_squared_error(y_train, train_baseline_pred)
r2_score(y_train, train_baseline_pred)

test_baseline_pred = baseline_pip.predict(X_test)
root_mean_squared_error(test_baseline_pred, y_test)
r2_score(y_test, test_baseline_pred)



## model Selection & optimaziatiom ##

models = {
    'LinearRegression': LinearRegression(),
    'ridge': Ridge(random_state=RANDOM_STATE),
    'lasso': Lasso(random_state=RANDOM_STATE, max_iter= 10000),
    'RadnomForest': RandomForestRegressor(),
    'HistGBR': HistGradientBoostingRegressor()
}

k = 5
cv = KFold(
    n_splits= k ,
    shuffle= True,
    random_state=RANDOM_STATE
)


scoring = {
    'rmse': 'neg_root_mean_squared_error',
    'mae': 'neg_mean_absolute_error',
    'r2': 'r2'
}

rows = []
for name, model in models.items():
    pipe = Pipeline(
        steps= [
            ('preprocess', preprocess),
            ('model', model)
        ]
    )
    scores = cross_validate(pipe, X_train, y_train, cv = cv , scoring=scoring,
                            n_jobs= -1)
    rows.append({
        'model': name,
        'cv_rmse': -scores['test_rmse'].mean(),
        'cv_mae': -scores['test_mae'].mean(),
        'cv_r2': scores['test_r2'].mean(),
    })

# sort based on lowest rmse value:
cv_results = pd.DataFrame(rows).sort_values('cv_rmse')
print('CV model comparison')
print(cv_results)


best_row = cv_results.iloc[0]

## Hyper Parameter Tuning ##
hgb_pipe = Pipeline(
    [
        ('preprocess', preprocess),
        ('model', HistGradientBoostingRegressor(
            random_state= RANDOM_STATE
        ))
    ]
)

# hyper parameter search
param_grid = {
    
    
    "model__learning_rate": [0.03, 0.05, 0.1],
    "model__max_depth": [None, 3, 6],
    "model__max_leaf_nodes": [15, 31, 63],
    "model__min_samples_leaf": [20, 50, 100],
    "model__l2_regularization": [0.0, 0.1, 1.0]
    
}

grid = GridSearchCV(
    estimator= pipe,
    param_grid= param_grid,
    cv = cv,
    scoring= "neg_root_mean_squared_error",
    n_jobs= -1,
    verbose= 1
)

grid.fit(X_train, y_train)

print(-grid.best_score_)
print(grid.best_params_)
# best possible combinations of hyper parameter


## retrain with best model ##
best_hgb = Pipeline(
    [
        ('preprocess', preprocess),
        ('model', HistGradientBoostingRegressor(
            random_state=RANDOM_STATE,
            max_leaf_nodes=63,
            l2_regularization=0.1,
            learning_rate=0.1,
            max_depth=None,
            min_samples_leaf=20,
        ))
    ]
)

best_hgb.fit(X_train, y_train)

## evaluation ##

y_train_pred = best_hgb.predict(X_train)
y_test_pred = best_hgb.predict(X_test)




r2_score(y_train_pred, y_train)
r2_score(y_test_pred, y_test)

root_mean_squared_error(y_test,y_test_pred)


res = y_test_pred - y_test
plt.scatter(res, y_test, s = 10)
plt.axhline(0, col = 'red')

sns.histplot(res, kde = True)

## predictive system ##
def house_price_prediction(Input_features):
    Input_features = X_train.columns
    
