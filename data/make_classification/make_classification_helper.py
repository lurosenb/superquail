# TODO: cleanup into function
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

n_samples = 2000
n_features = 5
n_informative = 5 # relevant features to explain target
n_redundant = 0 # linear combinations of informative
n_repeated = 0 # random copies of informative and redundant
n_useless = n_features - n_informative - n_redundant - n_repeated # noise

n_classes = 2
seed = 1

# define feature names
def feature_names (n_items, prefix = 'feature'):
    names = []
    for i in range(n_items):
        names.append(prefix + '_' + str(i))
    return names

inf_features = feature_names(n_informative, 'inf')
red_features = feature_names(n_redundant, 'red')
rep_features = feature_names(n_repeated, 'rep')
useless_features = feature_names(n_useless, 'noise')

feature_names = inf_features + red_features + rep_features + useless_features

X, y = make_classification(n_samples=n_samples, 
                    n_features=n_features, 
                    n_informative=n_informative, 
                    n_redundant=n_redundant, 
                    n_repeated=n_repeated, 
                    n_classes=n_classes, 
                    n_clusters_per_class=2, 
                    weights=None, 
                    flip_y=0.05, 
                    class_sep=5.0, 
                    hypercube=True, 
                    shift=15.0,
                    scale=0.4,
                    shuffle=True, 
                    random_state=seed)

# Convert to Dataframe
Z=np.zeros((X.shape[0], X.shape[1]+1))
Z[:,:-1]=X
Z[:,-1]=y

columns = feature_names + ['class']
df = pd.DataFrame(Z, columns=columns)
for f in feature_names:
    df[f] = df[f].astype('int32')
df['class'] = df['class'].astype('int32')

df.head()

Xy_train, Xy_test = train_test_split(df, test_size=0.2, stratify = df['class'], random_state = seed)

# from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler()

# Xy_train.loc[:,inf_features] = scaler.fit_transform(Xy_train.loc[:,inf_features])
# Xy_test.loc[:,inf_features] = scaler.transform(Xy_test.loc[:,inf_features])

X_train = Xy_train.drop('class', axis=1)
y_train = Xy_train['class']

X_test = Xy_test.drop('class', axis=1)
y_test = Xy_test['class']