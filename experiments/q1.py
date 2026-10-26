
# ==============================================================================
#  Working
#  Q1. Iris: Mean, Median, Standard Deviation and Box Plot
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# 1. Load the Iris dataset
iris = load_iris()
data = iris.data
feature_names = iris.feature_names

# 2. Calculate descriptive statistics
mean = np.mean(data, axis=0)
median = np.median(data, axis=0)
std_dev = np.std(data, axis=0, ddof=1)

print("=" * 65)
print("IRIS DATASET: DESCRIPTIVE STATISTICS")
print("=" * 65)
print(f"{'Feature':<25}{'Mean':>12}{'Median':>12}{'Std Dev':>12}")
print("-" * 65)

for i, feature in enumerate(feature_names):
    print(
        f"{feature:<25}"
        f"{mean[i]:>12.3f}"
        f"{median[i]:>12.3f}"
        f"{std_dev[i]:>12.3f}"
    )

# 3. Extract petal length (cm)
petal_length = data[:, 2]

# 4. Create the box plot
plt.figure(figsize=(6, 4))

plt.boxplot(
    petal_length,
    patch_artist=True,
    tick_labels=["Petal Length"]
)
plt.title("Box Plot of Iris Petal Length")
plt.ylabel("Length (cm)")
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()
