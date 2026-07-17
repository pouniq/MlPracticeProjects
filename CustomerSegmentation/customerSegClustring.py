import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


pd.set_option('display.max_columns', None)
sns.set_theme(style= 'darkgrid')
pd.set_option('display.float_format', lambda x : f'{x:.3f}')
plt.rcParams.update(
    {
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8
        
    }
)

RANDOM_STATE = 42
DF_PATH = 'customer_segmentation_data.csv'

df = pd.read_csv(DF_PATH)




df.info()
df.describe().T
df.describe()

df.shape
df.isnull().sum()


for col in df.columns:
    print(df[col].value_counts().head(10))


df.duplicated().sum()

df = df.drop(columns= 'id')


## data viz ###
sns.countplot(x = 'gender', data= df)

fig, axes = plt.subplots(4,2, figsize = (10,7))
axes = axes.flatten()

for i, col in enumerate(df.columns):
    sns.histplot(df[col], kde = True, ax = axes[i])

plt.tight_layout()
plt.show()


