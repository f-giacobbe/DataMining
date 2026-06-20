# Load dataset
```python
from sklearn.datasets import load_wine
import pandas as pd

wine = load_wine()      # Not a DataFrame!!
df = pd.DataFrame(wine.data, columns=wine.feature_names)    # DataFrame SENZA TARGET
df['target'] = wine.target      # Aggiungo la colonna target

# oppure, se mi serve per allenare un modello
X, y = wine.data, wine.target
```

# Holdout
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = test_train_split(X, y, test_size=0.2)
```

# Data cleaning (Transformers)
## Imputer (attributi numerici)
```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')      # Rimpiazza i nulli con la mediana
df_num = df.select_dtypes(include=[np.number])
df_num = DataFrame(imputer.fit_transform(df_num), columns=df_num.columns, index=df.index)
```

## OrdinalEncoder (attributi categorici)
> Non si usa perché i modelli interpretano male l'indice di categoria assumendo che la vicinanza sia significativa.
```python
from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()
df_cat = df.select_dtypes(include=['object', 'str'])
df_cat = DataFrame(ordinal_encoder.fit_transform(df_cat), columns=df_cat.columns, index=df.index)
```

## OneHotEncoder (attributi categorici)
```python
from sklearn.preprocessing import OneHotEncoder

one_hot_encoder = OneHotEncoder()
df_cat = df.select_dtypes(include=['object', 'str'])
                                                        # Serve perché di default restituisce un array sparso (una lista con solo gli 1)
df_cat = DataFrame(one_hot_encoder.fit_transform(df_cat).toarray(), columns=df_cat.columns, index=df.index)
```

## Pipeline
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('std_scaler', StandardScaler()),
])

df_num = DataFrame(num_pipeline.fit_transform(df_num), columns=df_num.columns, index=df.index)
```

## ColumnTransformer
> Serve per applicare diverse pipeline a diversi tipi di attributi
```python
from sklearn.compose import ColumnTransformer

cat_attr = [...]
num_attr = [...]

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attr),
    ("cat", OneHotEncoder(), cat_attr),
])

df_prepared = DataFrame(full_pipeline.fit_transform(df), columns=df.columns, index=df.index)
```

# pd.Series
```python
import numpy as np

series = pd.Series(np.arange(10), index=list('abcdefghil'))
series.index
series.values

series.isnull()     # mask

# Sort values
sorted_series = series.sort_values()

# Count
value_count = series.value_counts() # sort=True
```

# pd.DataFrame
```python
df.info()   # Informazioni del tipo: data types, utilizzo memoria, valori non-NA
df.describe()   # Media, count, ...
df.head(10)   # Prime 10 righe
df.tail(10)     # Ultime 10 righe




data = {
    'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
    'year': [2000, 2001, 2002, 2001, 2002],
    'pop': [1.5, 1.7, 3.6, 2.4, 2.9]
}
df = pd.DataFrame(data) # , columns=['year', 'state', 'pop'], index=np.arange(5))
# Index -> Row index
# Columns -> Column index


# Accessing the column
df['state']

# Accessing the rows
df.iloc[0]  # First element
df.loc[0]   # Element with index 0

df.index = np.arange(10)    # Yes
df.index[0] = 0             # No, the index is immutable
df.index = df.index.append(pd.Index(['ciao']))     # Yes, concatenation

df.columns = new_columns

# Dropping rows (axis = 0) (not in-place)
df = df.drop(['Colorado', 'Utah'])

# Dropping columns
df = df.drop([0, 1], axis=1)
df = df.drop(columns=[0, 1])

'''
Select all the element of
the dataframe with value greater
than 3 in column 'three'
'''
df[df['three'] > 3]

df.dropna() # how='any'/'all'

df.fillna('a')

df.duplicated()

df.drop_duplicates()

# Merge
df.join(df_)

# Sort index
sorted_df = df.sort_index() # axis=1, ascending=True

# Sample of n elements (or of n columns)
df.sample(n)    # axis=1
```

# Grafici

## Istogramma
```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()   # Enable seaborn

plt.figure()
plt.hist(df['eta'], bins=x)
df['eta'].hist(bins=x)  # equivalente
plt.show()

# Oppure, per un istogramma di ogni attributo
df.hist()
```

## Pairplot (importantissimo)
```python
sns.pairplot(iris, hue='species')

# Eventualmente anche per un sottoinsieme degli attributi
sns.pairplot(iris[['a', 'b']], hue='b')

# O per un sottoinsieme degli oggetti
sns.pairplot(iris.samples(200), hue='species')
```

## KDE
```python
sns.kdeplot(df['eta'])
```

## Jointplot
```python
sns.jointplot(data=df, x='x', y='y', kind='kde')
```

## Grafico a dispersione cluster (scatter)
```python
import matplotlib.pyplot as plt

plt.figure()
plt.scatter(df['x'], df['y'], c=df['target'])
plt.show()
```

## Heatmap correlazione (richiede Pandas, Seaborn, e PyPlot di MatPlotLib)
```python
import matplotlib.pyplot as plt
import seaborn as sns

correlation_matrix = df.corr(numeric_only=True)     # Considero solo gli attributi numerici

plt.figure()
sns.heatmap(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
plt.show()

# Può essere utile vedere la correlazione rispetto al target, per fare eventuale pruning
correlation_matrix['target'].sort_values(ascending=False)
```

## Confusion matrix
```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Oppure tramite sns
plt.figure()
sns.heatmap(cm, ...)
plt.show()
```

## Barh (es. plot 5 most important features)
```python
import matplotlib.pyplot as plt
import numpy as np

feature_importances = random_forest_clf.feature_importances_
top_5_indices = np.argsort(feature_importances_)[:5][::-1]

plt.figure()    # y                                             width
plt.barh([wine.feature_names[i] for i in top_5_indices], [feature_importances[i] for i in top_5_indices])
plt.show()
```

## Plot generico
```python
import matplotlib.pyplot as plt

plt.figure()
plt.plot(X, Y)
plt.show()
```

# Classificazione

## Decision Tree con massima profondità
```python
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(max_depth=15)
dt.fit(X_train, y_train)

y_pred = df.predict(y_test)
y_pred_prob = df.predict_proba(y_test)
```

## KNN con k neighbor
```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
```

## Ensemble

### Voting Classifier (hard e soft)
> Nel soft voting, il peso del voto di un classificatore è pari alla probabilità max predetta. (ATTENZIONE! Per il classificatore SVC è necessario istanziarlo con `probability=True`).
```python
from sklearn.ensemble import VotingClassifier

voting = VotingClassifier(estimators = [('dt', dt), ('knn', knn)], voting='hard')   # per il soft voting è sufficiente rimpiazzare 'hard' con 'soft'
voting.fit(X_train, y_train)
```

### Bagging & Pasting
> Bagging: bootstrap aggregating with replacement. Pasting: bootstrap aggregating without replacement.
```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier # base classifier

base = DecisionTreeClassifier()

bagging = BaggingClassifier(base,
    n_estimators=500,   # quante istanze dello stesso modello
    max_samples=100,
    oob_score=True      # Out-of-bag: utilizza gli oggetti fuori dal training set per valutare il modello
    bootstrap=True,)    # True -> bagging; False -> pasting

bagging.fit(X_train, y_train)
y_pred = bagging.predict(X_test)

# Se oob_score=True
bagging.oob_score_      # Simile all'errore sul test set
```

### Random forest
```python
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Feature importances
clf.feature_importances_    # out: Array con importanza di ogni feature
```

### AdaBoost
```python
from sklearn.ensemble import AdaBoostClassifier

clf = AdaBoostClassifier(estimator=base_clf, learning_rate=0.5)
clf.fit(X_train, y_train)
```

### Gradient Boosting Regressor
```python
from sklearn.ensemble import GradientBoostingRegressor

clf = GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=1.0)
clf.fit(X_train, y_train)
```

## Metriche di valutazione

### Cross validation score
```python
from sklearn.model_selection import cross_val_score

cross_val_score(classifier, X_train, y_train, cv=n, scoring='accuracy')
```


### Cross validation predict (dà in output proprio le classi/proba predette invece dello score)
> La differenza tra predict e cross_val_predict è che la predict la si fa dopo la fit (quindi su un modello allenato) ed è opportuno fornire un oggetto non visto dal modello durante il training. La cross_val predict, invcece, viene fatta sugli stessi dati di training.
```python
from sklearn.model_selection import cross_val_predict

cross_val_predict(clf, X_train, y_train, cv=3)#, method='predict_proba')
```

### Precision/recall
```python
from sklearn.metrics import precision_score, recall_score

precision_score(y_train, y_pred)
recall_score(y_train, y_pred)
```

### F1-score
```python
from sklearn.metrics import f1_score

f1_score(y_test, dt.predict(X_test))
```

### ROC AUC
```python
from sklearn.metrics import roc_auc_score

roc_auc_score(y_train, y_pred)
```

### Accuracy score
```python
from sklearn.metrics import accuracy score

accuracy_score(y_test, y_pred)
```

# Clustering

## K-means
```python
from sklearn.cluster import KMeans

# Ho già X = wine.data e y = wine.target

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)       # Solo X! (unsupervised)

# Cluster predetti (da X)
kmeans.labels_

# Aggiungere eventuali nuovi punti (se X_new = X, allora coincide con labels_)
kmeans.predict(X_new)

# Centroidi
kmeans.cluster_centers_
```

## K-means++
```python
from sklearn.cluster import KMeans

kmeans_pp = KMeans(n_clusters=3, init='k-means++')
```

## DBSCAN (non ha la predict(), solo fit_predict() -> non riesce a predire nuovi dati al di fuori da quelli di training)
```python
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.05)
dbscan.fit(X)
```

## Metriche di valutazione

### SSE (inertia)
```python
kmeans.inertia_
kmeans.score(X)     # SSE negativa (score da massimizzare)
```

### Silhouette (not in cheatsheet!)
```python
from sklearn.metrics import silhouette_score

silhouette_score(X, kmeans.labels_)
```

# Regressione
## Linear Regression
```python
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
```

## Decision Tree Regressor
```python
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(X_train, y_train)
```

## Random Forest Regressor
```python
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(n_estimators=10)
forest_reg.fit(X_train, y_train)
```

## Metriche di valutazione
### (Root) Mean Squared Error, Mean Absolute Error
```python
from sklearn.metrics import mean_squared_error
import numpy as np

y_pred = lin_reg.predict(X_train)
lin_mse = mean_squared_error(y_train, y_pred)        # In questo caso vogliamo l'errore sul training set
lin_rmse = np.sqrt(lin_mse)
```

### Cross validation score
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, X_train, y_train, cv=n, scoring='neg_mean_squared_error')     # NEG perché score va massimizzato
rmse_scores = np.sqrt(-scores)
```


# Rete neurale
```python
import torch
from torch import nn
# import torch.nn.functional as F       permette di usare F.relu invece di torch.relu, F.sigmoid invece di torch.sigmoid, ecc.

input = 128
neuroni_layer_1 = 64
neuroni_layer_2 = 16
output = 4

class MyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(input, neuroni_layer_1)
        self.fc2 = nn.Linear(neuroni_layer_1, neuroni_layer_2)
        self.fc3 = nn.Linear(neuroni_layer_2, output)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.softmax(x, dim=1)      # dim=1 fa in modo che la somma delle probabilità di ogni riga sia 1 (quello che vogliamo)
```
