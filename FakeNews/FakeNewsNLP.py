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

