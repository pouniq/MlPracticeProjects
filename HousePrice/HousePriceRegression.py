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