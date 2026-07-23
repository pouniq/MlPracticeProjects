import re # regular expression library
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from collections import Counter

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_validate, RandomizedSearchCV, KFold, GridSearchCV
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

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



# build predictive system
def pred_news(title, text):
    combined_text = clean_text(f"{title} {text}")
    pred = model_pip.predict([combined_text])
    print(pred)
    if pred == 0:
        print('The news is fake')
    else: 
        print('The news is Real')
    

example_title = "Breaking: Government announces new economic policy"
example_text = "The finance minister introduced a new policy today after discussions in parliament..."
pred_news(example_title, example_title)



# Model selection
models = {
    'LogisticRegression': LogisticRegression(),
    'SVC': LinearSVC(),
    'NaiveBayes': MultinomialNB()
}

k = 5
cv = KFold(
    n_splits= k,
    shuffle= True,
    random_state=RANDOM_STATE
)

scoring = 'accuracy'

rows = []
for name, model in models.items():
    pipe = Pipeline(
        steps=[
            ('tfidftransform', TfidfVectorizer(
                max_features= 50000,
                min_df= 2,
                stop_words= 'english',
                ngram_range= (1,2)
            )) ,
            ('model', model)
        ]
    )
    scores = cross_validate(pipe, X_train, y_train, cv = cv , scoring=scoring,
                            n_jobs= -1)
    rows.append({
        'model': name,
        'cv_acc': scores['test_score'].mean(),
    })

# sort based on lowest rmse value:
cv_results = pd.DataFrame(rows).sort_values('cv_acc', ascending = False)
print('CV model comparison')
print(cv_results)

## best model for my data is SVC ##
svc_pip = Pipeline(
    steps=[
        ('tfidftransform', TfidfVectorizer()),
        ('model', LinearSVC(
            random_state= RANDOM_STATE
        ))
    ]
)


param_grid = {
    'model__penalty': ['l1', 'l2'],
    'model__loss': ['hinge', 'squared_hinge'],
    'model__multi_class': ['ovr', 'crammer_singer'],
    'model__C': [0.01, 0.1, 1, 10, 100],
    'model__class_weight': [None, 'balanced'],
}

# grid_r = RandomizedSearchCV(
#     estimator=svc_pip,
#     param_distributions = param_grid,
#     cv = cv ,
#     scoring= 'accuracy',
#     n_jobs= -1,
#     verbose= 1
# )


grid = GridSearchCV(
    estimator=svc_pip,
    param_grid = param_grid,
    cv = cv ,
    scoring= 'accuracy',
    n_jobs= -1,
    verbose= 1
)

grid.fit(X_train, y_train)
grid.best_params_
grid.best_score_

svc_model = LinearSVC(
    C = 1,
    class_weight= None,
    loss= 'squared_hinge',
    multi_class= 'ovr',
    penalty='l1'
)


