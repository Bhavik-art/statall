
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load dataset
iris = load_iris()
X = iris.data

# Apply K-Means clustering
model = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = model.fit_predict(X)

# Calculate Silhouette Score
score = silhouette_score(X, labels)

print("Number of Clusters:", 3)
print("Silhouette Score:", round(score, 4))
