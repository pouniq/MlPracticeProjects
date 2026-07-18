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



df.columns

sns.scatterplot(
    y= 'spending_score',
    x= 'income',
    data= df,
    s = 55
)



sns.scatterplot(
    y= 'preferred_category',
    x= 'gender',
    data= df,
    s = 55
)

## data preprocessing ##

### feature selection:
columns_to_select = ['income', 'spending_score']
X = df[columns_to_select]

# scale features

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Elbow method - WCSS
# (find best k value)
wcss = []
k_range = range(2, 11)
print(*k_range)

for k in k_range:
    K_means = KMeans(n_clusters= k, random_state= RANDOM_STATE, n_init= 10)
    K_means.fit(X_scaled)
    wcss.append(K_means.inertia_)

plt.figure(figsize=(10,7))
plt.plot(list(k_range), wcss, marker = 'o')
plt.xlabel('Number of clusters')
plt.ylabel('inertia')
plt.title('Elbow Method')
plt.show()

    

# model -- k_means clustring
K_FINAL = 5
K_means_final = KMeans(
    n_clusters= K_FINAL,
    random_state= RANDOM_STATE,
    n_init= 10
)
K_means_final.fit(X_scaled)
clusters = K_means_final.predict(X_scaled)

df_cluster = df.copy(deep = True)

df_cluster['cluster'] = clusters

sns.scatterplot(
    x = 'income',
    y = 'spending_score',
    hue = 'cluster' ,
    data= df_cluster
)


# evaluation 
kmeans_score = silhouette_score(X_scaled, df_cluster['cluster'])
round(kmeans_score, 3)

# we will see how you silhouette_score would differ with different k values

sil_score = []
for k in k_range:
    model = KMeans(n_clusters= k, random_state= RANDOM_STATE, n_init= 10)
    cluster_label = model.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled,cluster_label)
    sil_score.append(sil)
plt.figure(figsize=(10,7))
plt.plot(list(k_range) , sil_score, marker = 'o')
plt.xlabel('k cluster')
plt.ylabel('silhouette score')
plt.title('silhouette score for different k clusters')
plt.show()
 
# visualize clusters

sns.scatterplot(
    x = 'income',
    y = 'spending_score',
    hue = 'cluster' ,
    data= df_cluster,
    palette= 'tab10',
    s = 12
)

## cluster interpretation
### business profiling 
profile_cols =  ["age", "income", "spending_score", "last_purchase_amount"]
cluster_sizes = df_cluster["cluster"].value_counts().sort_index()
cluster_sizes_mean = df_cluster.groupby('cluster')[profile_cols].mean().round(2)
cluster_sizes_median = df_cluster.groupby('cluster')[profile_cols].median().round(2)
cluster_sizes_mean

df.columns

plt.bar(range(0,5),cluster_sizes)


# segment customers 
def assign_customer(income, spneding_score, scaler, model):
    new_row = pd.DataFrame([[income, spneding_score]],
                           columns= ["income", 'spending_score'])
    
    new_row_scaled = scaler.transform(new_row)
    cluster_no = model.predict(new_row_scaled)[0]
    return cluster_no

new_cl = assign_customer(
    income= 40,
    spneding_score= 43,
    scaler= scaler,
    model= K_means_final
)
    