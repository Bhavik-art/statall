
# ==============================================================================
# Working
# Q15. Breast Cancer: Permutation Importance (Random Forest)
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

# 1. Load the Breast Cancer dataset
bc = load_breast_cancer()
X = bc.data
y = bc.target
feature_names = bc.feature_names

# 2. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Train the Random Forest classifier
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
rf.fit(X_train, y_train)

# 4. Evaluate test accuracy
accuracy = rf.score(X_test, y_test)

print("=" * 55)
print("RANDOM FOREST: PERMUTATION IMPORTANCE")
print("=" * 55)
print(f"Test accuracy: {accuracy:.4f}")

# 5. Calculate permutation importance on test data
result = permutation_importance(
    rf,
    X_test,
    y_test,
    n_repeats=30,
    random_state=42,
    scoring="accuracy"
)

# 6. Select the five most important features
importance_mean = result.importances_mean
importance_std = result.importances_std

top_indices = np.argsort(importance_mean)[::-1][:5]

top_names = feature_names[top_indices]
top_means = importance_mean[top_indices]
top_stds = importance_std[top_indices]

print("\nTop 5 Features (Permutation Importance):")
print(f"{'Feature':<30}{'Mean':>10}{'Std Dev':>12}")
print("-" * 52)

for name, mean, std in zip(top_names, top_means, top_stds):
    print(f"{name:<30}{mean:>10.4f}{std:>12.4f}")

# 7. Sort features for the horizontal bar chart
order = np.argsort(top_means)

plot_names = top_names[order]
plot_means = top_means[order]
plot_stds = top_stds[order]

# 8. Plot permutation importance
plt.figure(figsize=(8, 5))

plt.barh(
    plot_names,
    plot_means,
    xerr=plot_stds,
    color="darkorange",
    capsize=4,
    edgecolor="black"
)

plt.axvline(0, color="black", linewidth=0.8)
plt.xlabel("Mean Decrease in Accuracy")
plt.ylabel("Feature")
plt.title("Permutation Importance (Test Set)")
plt.grid(axis="x", alpha=0.25)
plt.tight_layout()
plt.show()
