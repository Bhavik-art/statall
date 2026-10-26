# ==============================================================================
# Q10. Iris: Random Forest classifier
# ==============================================================================
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42, stratify=iris.target)
clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
pred = clf.predict(X_test)

print('Accuracy:', round(accuracy_score(y_test, pred), 4))
print('\nConfusion matrix:'); print(confusion_matrix(y_test, pred))
print('\nClassification report:')
print(classification_report(y_test, pred, target_names=iris.target_names))

ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=iris.target_names, cmap='Greens')
plt.title('Random Forest: confusion matrix'); plt.show()
