# ==============================================================================
# Q12. Iris: PCA to two components
# ==============================================================================
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

iris = load_iris()
X_std = StandardScaler().fit_transform(iris.data)
pca = PCA(n_components=2)
Z = pca.fit_transform(X_std)

print('Explained variance ratio:', pca.explained_variance_ratio_.round(4))
print('Total variance kept     :', round(pca.explained_variance_ratio_.sum(), 4))

plt.figure(figsize=(6, 4.5))
for i, name in enumerate(iris.target_names):
    plt.scatter(Z[iris.target == i, 0], Z[iris.target == i, 1], label=name, alpha=0.8)
plt.xlabel('PC1'); plt.ylabel('PC2'); plt.title('PCA of Iris (2 components)'); plt.legend(); plt.show()
