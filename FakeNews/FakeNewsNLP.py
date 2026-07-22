import re # regular expression library
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from collections import Counter

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
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
CSV_PATH = 'WELFake_Dataset.csv' 
TARGET_COL = 'label'

df = pd.read_csv(CSV_PATH)
print(df)
# here: 1 --> real news
#       0 --> fake news
df.drop(columns= ['Unnamed: 0'], inplace= True)

# we get sample from this dataset 
# to better handle the model training time 
# and better preprocessing

# target distribution.
df['label'].value_counts()


df_real = df[df['label'] == 1]
df_fake = df[df['label'] == 0]

n_samples = 1000


data_fake = df_fake.sample(n_samples, random_state= RANDOM_STATE)
data_real = df_real.sample(n_samples, random_state= RANDOM_STATE)

df = (
    
    pd.concat([data_real, data_fake]).sample(frac = 1,
                                             random_state=RANDOM_STATE).reset_index(drop=True)
    
)
df

df.columns
df.info()
df.isnull().sum()
df.dropna(inplace=True)


df['label'].value_counts()
df['label'].value_counts(normalize=True)


## text cleaning ##

df['text']


def clean_text(s: str) -> str:
    """
    text cleaning:
    - lowercase
    - remove URLs
    - remove html tags
    - normalize spaces

    Args:
        s (str):
        - this function only gets str objects

    Returns:
        str: 
        - this function only return str objects
    """
    if pd.isna(s):
        return ''
    s = str(s).lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)          # URLs
    s = re.sub(r"<.*?>", " ", s)                    # HTML tags
    s = re.sub(r"[^a-z0-9\s\.\,\!\?\-\']", " ", s)   # keep basic chars
    s = re.sub(r"\s+", " ", s).strip()               # normalize spaces
    return s
