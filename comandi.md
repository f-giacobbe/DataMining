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

# pd.DataFrame
```python
df.info()   # Informazioni del tipo: data types, utilizzo memoria, valori non-NA
df.describe()   # Media, count, ...
df.head(10)   # Prime 10 righe
df.tail(10)     # Ultime 10 righe
```

# Heatmap correlazione (richiede Pandas, Seaborn, e PyPlot di MatPlotLib)
```python
import matplotlib.pyplot as plt
import seaborn as sns

correlation_matrix = df.corr(numeric_only=True)     # Considero solo gli attributi numerici

plt.figure()
sns.heatmap(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
plt.show()
```

# Modelli (sklearn)
## K-means
```python
from sklearn.cluster import KMeans

# Ho già X = wine.data e y = wine.target

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)       # Solo X!

kmeans.inertia_     # (SSE)
```

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

### F1-score
```python
from sklearn.metrics import f1_score

f1_score(y_test, dt.predict(X_test))
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
