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
```

# Grafici

## Istogramma
```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()   # Enable seaborn

plt.figure()
plt.hist(df['eta'])
plt.show()
```

## Pairplot (importantissimo)
```python
sns.pairplot(iris, hue='species')
```

## KDE
```python
sns.kdeplot(df['eta'])
```

## Jointplot
```python
sns.jointplot(data=df, x='x', y='y', kind='kde')
```

## Grafico a dispersione cluster (?)

## Heatmap correlazione (richiede Pandas, Seaborn, e PyPlot di MatPlotLib)
```python
import matplotlib.pyplot as plt
import seaborn as sns

correlation_matrix = df.corr(numeric_only=True)     # Considero solo gli attributi numerici

plt.figure()
sns.heatmap(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
plt.show()
```

## Confusion matrix
```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()
```

# Classificazione

## Decision Tree con massima profondità
```python
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(max_depth=15)
dt.fit(X_train, y_train)
```

## KNN con k neighbor
```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
```

## Voting Classifier (con hard voting)
```python
from sklearn.ensemble import VotingClassifier

voting = VotingClassifier(estimators = [('dt', dt), ('knn', knn)], voting='hard')
voting.fit(X_train, y_train)
```

## Random forest
```python
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=n)
???????????????????????????????????????????????????????????????????
```

## Cross validation score
```python
from sklearn.model_selection import cross_val_score

cross_val_score(classifier, X_train, y_train, cv=n, scoring='accuracy')
```

## Cross validation predict (dà in output proprio le classi/proba predette invece dello score)
> La differenza tra predict e cross_val_predict è che la predict la si fa dopo la fit (quindi su un modello allenato) ed è opportuno fornire un oggetto non visto dal modello durante il training. La cross_val predict, invcece, viene fatta sugli stessi dati di training.
```python
from sklearn.model_selection import cross_val_predict

cross_val_predict(clf, X_train, y_train, cv=3)#, method='predict_proba')
```

## Precision/recall
```python
from sklearn.metrics import precision_score, recall_score

precision_score(y_train, y_pred)
recall_score(y_train, y_pred)
```

## F1-score
```python
from sklearn.metrics import f1_score

f1_score(y_test, dt.predict(X_test))
```

## ROC AUC
```python
from sklearn.metrics import roc_auc_score

roc_auc_score(y_train, y_pred)
```

# Clustering

## K-means
```python
from sklearn.cluster import KMeans

# Ho già X = wine.data e y = wine.target

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)       # Solo X!

kmeans.inertia_     # (SSE)
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
