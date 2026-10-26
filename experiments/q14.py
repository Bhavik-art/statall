
# ==============================================================================
# Working
# Q14. Breast Cancer: Decision Tree Feature Importance
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

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

# 3. Train the Decision Tree classifier
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)

# 4. Evaluate test accuracy
accuracy = tree.score(X_test, y_test)

print("=" * 50)
print("DECISION TREE FEATURE IMPORTANCE")
print("=" * 50)
print(f"Test accuracy: {accuracy:.4f}")

# 5. Find the five most important features
importances = tree.feature_importances_
top_indices = np.argsort(importances)[::-1][:5]

top_names = feature_names[top_indices]
top_values = importances[top_indices]

print("\nTop 5 Feature Importances:")
for name, value in zip(top_names, top_values):
    print(f"{name:<30} {value:.4f}")

# 6. Sort for horizontal bar chart
order = np.argsort(top_values)

plot_names = top_names[order]
plot_values = top_values[order]

# 7. Plot feature importance
plt.figure(figsize=(8, 5))

plt.barh(
    plot_names,
    plot_values,
    color="purple",
    edgecolor="black"
)

plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Decision Tree: Top 5 Important Features")
plt.grid(axis="x", alpha=0.25)
plt.tight_layout()
plt.show()
