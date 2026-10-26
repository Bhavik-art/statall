
# ==============================================================================
# Working
# Q3. Iris: Pearson Correlation Heatmap
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# 1. Load the Iris dataset
iris = load_iris()
data = iris.data
features = iris.feature_names
n = len(features)

# 2. Calculate Pearson correlation matrix
corr = np.corrcoef(data, rowvar=False)

# 3. Print correlation matrix
print("=" * 75)
print("PEARSON CORRELATION MATRIX (IRIS DATASET)")
print("=" * 75)

print(f"{'Feature':<25}", end="")
for feature in features:
    print(f"{feature[:12]:>14}", end="")
print()

for i, feature in enumerate(features):
    print(f"{feature:<25}", end="")
    for j in range(n):
        print(f"{corr[i, j]:>14.3f}", end="")
    print()

# 4. Plot the correlation heatmap
plt.figure(figsize=(8, 6))

plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="Pearson correlation (r)")

plt.xticks(range(n), features, rotation=45, ha="right")
plt.yticks(range(n), features)

# Display correlation values in each cell
for i in range(n):
    for j in range(n):
        value = corr[i, j]
        text_color = "white" if abs(value) > 0.65 else "black"
        plt.text(
            j, i, f"{value:.2f}",
            ha="center", va="center",
            color=text_color
        )

plt.title("Pearson Correlation Heatmap (Iris)")
plt.tight_layout()
plt.show()

# 5. Find the strongest correlated pair, excluding diagonal values
upper_triangle = np.triu(np.abs(corr), k=1)
i, j = np.unravel_index(np.argmax(upper_triangle), upper_triangle.shape)

print("\nStrongest correlated pair:")
print(f"{features[i]} vs {features[j]}")
print(f"Pearson correlation (r) = {corr[i, j]:.3f}")
