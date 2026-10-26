
# ==============================================================================
# Working
# Q13. Breast Cancer: 5-Fold Cross-Validation
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

# 1. Load the Breast Cancer dataset
bc = load_breast_cancer()
X = bc.data
y = bc.target

# 2. Create the machine learning pipeline
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

# 3. Configure 5-fold stratified cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# 4. Evaluate model using accuracy
scores = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="accuracy"
)

# 5. Print fold accuracies
print("=" * 45)
print("5-FOLD CROSS-VALIDATION RESULTS")
print("=" * 45)

for i, score in enumerate(scores, start=1):
    print(f"Fold {i} accuracy: {score:.4f}")

# 6. Calculate mean and standard deviation
mean_accuracy = scores.mean()
std_deviation = scores.std()

print(f"\nMean accuracy : {mean_accuracy:.4f}")
print(f"Std deviation : {std_deviation:.4f}")
print(f"Mean accuracy (%): {mean_accuracy * 100:.2f}%")

# 7. Plot fold accuracies
plt.figure(figsize=(7, 4.5))

plt.bar(
    range(1, 6),
    scores,
    color="teal",
    edgecolor="black"
)

plt.axhline(
    mean_accuracy,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Mean = {mean_accuracy:.4f}"
)

plt.xticks(range(1, 6))
plt.ylim(0.90, 1.0)
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.title("5-Fold Cross-Validation")
plt.legend()
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.show()
