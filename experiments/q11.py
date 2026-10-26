
# ==============================================================================
# Working
# Q11. Iris: K-Means with Silhouette Score
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. Load the Iris dataset
iris = load_iris()
X = iris.data
features = iris.feature_names

# 2. Apply K-Means clustering
km = KMeans(
    n_clusters=3,
    n_init=10,
    random_state=42
)

labels = km.fit_predict(X)

# 3. Calculate cluster sizes
cluster_sizes = np.bincount(labels)

print("=" * 45)
print("IRIS K-MEANS CLUSTERING")
print("=" * 45)

for i, size in enumerate(cluster_sizes):
    print(f"Cluster {i}: {size} samples")

# 4. Calculate Silhouette Score
score = silhouette_score(X, labels)
print(f"\nSilhouette Score: {score:.4f}")

# 5. Plot clusters using petal length and petal width
plt.figure(figsize=(7, 5))

plt.scatter(
    X[:, 2],
    X[:, 3],
    c=labels,
    cmap="viridis",
    alpha=0.8,
    edgecolors="black",
    label="Data points"
)

# Plot cluster centroids
plt.scatter(
    km.cluster_centers_[:, 2],
    km.cluster_centers_[:, 3],
    c="red",
    marker="X",
    s=200,
    edgecolors="black",
    label="Centroids"
)

plt.xlabel("Petal length (cm)")
plt.ylabel("Petal width (cm)")
plt.title("K-Means Clustering (k = 3)")
plt.legend()
plt.tight_layout()
plt.show()
