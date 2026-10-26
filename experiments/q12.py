
# ==============================================================================
# Working
# Q12. Iris: PCA to Two Components
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# 2. Standardize the features
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# 3. Apply PCA to reduce 4 features to 2 components
pca = PCA(n_components=2)
Z = pca.fit_transform(X_std)

# 4. Display explained variance
print("=" * 45)
print("PCA: IRIS DATASET")
print("=" * 45)

print("Explained variance ratio:",
      np.round(pca.explained_variance_ratio_, 4))

print("Total variance kept      :",
      round(pca.explained_variance_ratio_.sum(), 4))

print(f"Variance retained        : "
      f"{pca.explained_variance_ratio_.sum() * 100:.2f}%")

# 5. Plot the two principal components
plt.figure(figsize=(7, 5))

for i, name in enumerate(iris.target_names):
    plt.scatter(
        Z[y == i, 0],
        Z[y == i, 1],
        label=name,
        alpha=0.8
    )

plt.xlabel("Principal Component 1 (PC1)")
plt.ylabel("Principal Component 2 (PC2)")
plt.title("PCA of Iris Dataset (2 Components)")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()
