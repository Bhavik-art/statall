
# ==============================================================================
# Working
# Q10. Iris: Random Forest Classifier
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report
)

# 1. Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# 2. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# 3. Train the Random Forest classifier
clf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
clf.fit(X_train, y_train)

# 4. Predict the test data
pred = clf.predict(X_test)

# 5. Calculate accuracy
accuracy = accuracy_score(y_test, pred)

print("=" * 50)
print("IRIS RANDOM FOREST CLASSIFIER")
print("=" * 50)
print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

# 6. Display confusion matrix
cm = confusion_matrix(y_test, pred)

print("\nConfusion Matrix:")
print(cm)

# 7. Display classification report
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        pred,
        target_names=iris.target_names,
        zero_division=0
    )
)

# 8. Plot confusion matrix
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names
)

display.plot(cmap="Greens", values_format="d")
plt.title("Random Forest: Confusion Matrix")
plt.tight_layout()
plt.show()
