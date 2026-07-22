import re # regular expression library
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from collections import Counter

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
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


df['content'] = (df['title'] + ' ' + df['text']).apply(clean_text)


## adding additional columns for analysis ##
df['title_len_char'] = df['title'].str.len()
df['text_len_char'] = df['text'].str.len()
df['content_len_words'] = df['content'].str.split().apply(len)


num_cols = ['title_len_char', 'text_len_char', 'content_len_words']
df[num_cols].describe().T


## data visualization ##
sns.countplot(x = 'label', data= df)
# - we have almost equal label dist.


# boxplots for fake news and real news
fake_len = df[df["label"] == 0]['content_len_words'].values
real_len = df[df["label"] == 1]['content_len_words'].values


plt.boxplot([fake_len, real_len], showfliers = False)
plt.show()


X = df['content']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=RANDOM_STATE)


model_pip = Pipeline(
    steps= [
        ('tfidf', TfidfVectorizer(
            max_features= 50000,
            ngram_range= (1,2),
            stop_words = 'english',
            min_df = 2
        )),
        ('model', LogisticRegression())
        
    ]
)




# train pipline
model_pip.fit(X_train, y_train)


## model Evaluation ##
y_pred_train = model_pip.predict(X_train)
y_pred = model_pip.predict(X_test)

acc_train = accuracy_score(y_pred_train, y_train)
acc_test = accuracy_score(y_pred , y_test)
cm_train = confusion_matrix(y_pred , y_test)
print(classification_report(y_pred , y_test))




