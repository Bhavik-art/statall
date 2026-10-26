import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()

# Create DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Calculate Pearson correlation matrix
corr_matrix = df.corr(method='pearson')

print("Pearson Correlation Matrix:")
print(corr_matrix.round(2))

# Generate heatmap using Matplotlib
plt.figure(figsize=(8, 6))

plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)

plt.colorbar(label="Pearson Correlation")

# Add feature names
plt.xticks(range(len(df.columns)), df.columns,
           rotation=45, ha='right')
plt.yticks(range(len(df.columns)), df.columns)

# Display correlation values
for i in range(len(corr_matrix.columns)):
    for j in range(len(corr_matrix.columns)):
        plt.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                 ha='center', va='center')

plt.title("Pearson Correlation Heatmap - Iris Dataset")
plt.tight_layout()
plt.show()

# Find the strongest absolute correlation
corr_pairs = corr_matrix.abs().where(
    ~__import__('numpy').eye(len(corr_matrix), dtype=bool)
).stack()

strongest_pair = corr_pairs.idxmax()
strongest_value = corr_matrix.loc[strongest_pair[0],
                                  strongest_pair[1]]

print("\nStrongest Correlated Feature Pair:")
print(strongest_pair[0], "and", strongest_pair[1])
print("Correlation:", round(strongest_value, 2))