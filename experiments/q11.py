# ==============================================================================
# Q11. Iris: K-Means with Silhouette Score
# ==============================================================================
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.datasets import load_iris
iris = load_iris(as_frame=True)
df = iris.data
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

km = KMeans(n_clusters=3, n_init=10, random_state=42)
labels = km.fit_predict(df)
print('Cluster sizes:', np.bincount(labels))
print('Silhouette Score:', round(silhouette_score(df, labels), 4))

plt.figure(figsize=(6, 4.5))
plt.scatter(df.iloc[:, 2], df.iloc[:, 3], c=labels, cmap='viridis', alpha=0.8)
plt.scatter(km.cluster_centers_[:, 2], km.cluster_centers_[:, 3], c='red', marker='X', s=200, label='Centroids')
plt.xlabel('petal length (cm)'); plt.ylabel('petal width (cm)'); plt.title('K-Means (k = 3)'); plt.legend(); plt.show()
