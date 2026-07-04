# Setup


## import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

## config.
pd.set_option('display.max_columns', None)
sns.set_theme(style='darkgrid')
RANDOM_STATE = 42


## read dataset
df = pd.read_csv('sonar.csv', header=None)
len(df.columns)
df.columns


## EDA:
### basic dataset overview:

df.describe()
df.info()

df.shape

### encoded missing values
for col in df.columns:
    print(df[col].value_counts())
### -- we do not have missing values (in strange ways)

### class distribution:
# -- if we have so different values in our target column then we need to apply
# some methods like undersample, oversample , etc
df[60].value_counts(normalize=True)
## -- in this case we dont have different classes


### encoding strings to numbers 
df[60] = df[60].map({'M':1 , 'R':0})
## -- we have 1 when we see a MINE and Zero we get 0


df.duplicated().sum() # we do not have any duplicates!

## we need to identify the constant columns 
## we do not have any constant or quasi-constant cols
df.nunique()

df.groupby(60).mean()




## Data visualization
### distrobution

# instead of using this for loop to print it one by one
# we get ONE picture of all plots.

# for col in df.columns:
#     plt.figure(figsize=(4,3)),
#     sns.histplot(df[col], kde= True),
#     plt.title(col),
#     plt.xlabel(col),
#     plt.ylabel('freq'),
#     plt.show()

    

# creating subplots
fig, axes = plt.subplots(13,5, figsize=(15,18))
axes = axes.flatten()

for i, col in enumerate(df.columns):
    sns.histplot(df[col], kde=True, ax= axes[i])
    axes[i].set_title(col, fontsize = 8)
plt.tight_layout()
plt.show()
    
    
# Outlier handling -- BoxPlots
fig, axes = plt.subplots(13,5, figsize=(15,18))
axes = axes.flatten()

for i, col in enumerate(df.columns):
    sns.boxplot(x = df[col], ax= axes[i])
    axes[i].set_title(col, fontsize = 8)
plt.tight_layout()
plt.show() 


# pairPlot -- it is computanationally expensive
# we can draw the correlation plot (heatmap)
plt.figure(figsize=(14,10))
sns.heatmap(
    df.corr(),
    cmap='coolwarm',
    center=0
)
plt.title('correlation heatmap')
plt.show()


# Data preprocessing
X = df.drop(columns = 60)
y = df[60]


X_train, X_test ,y_train, y_test = train_test_split(X,
                                                    y,
                                                    random_state=RANDOM_STATE,
                                                    stratify=y,
                                                    test_size=0.2)

## feature Scalling

st_scaler = StandardScaler()
X_train_st = st_scaler.fit_transform(X_train)
X_test_st = st_scaler.transform(X_test)



# Model Selection


model = SVC()
model.fit(X_train_st, y_train)
y_train_pred = model.predict(X_train_st)
y_pred = model.predict(X_test_st)



## model evaluation
accuracy_score(y_train_pred, y_train)
accuracy_score(y_pred, y_test)


print(classification_report(y_train, y_train_pred))


# buiding a predictive model

def predict_obj(input_features):
    # scale the features
    scaled_features = st_scaler.transform([input_features])
    # getting the prediction 
    prediction = model.predict(scaled_features)
    print(prediction)
    if prediction[0] == 1:
        print('the object is identified as MINE')
    else: 
        print('the object is identified as ROCK')
        
    


## sample prediction
## it is when a new record comes and you want to get the prediction


y_test.head() # ground truths
test_1 = X_test.loc[103].tolist()
predict_obj(test_1)

    
    
test_2 = X_test.loc[90].tolist()
predict_obj(test_2)










