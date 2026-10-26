# ==============================================================================
# Q14. Breast Cancer: Decision Tree feature importance
# ==============================================================================
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
bc = load_breast_cancer(as_frame=True)
X, y = bc.data, bc.target
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
tree = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
print('Test accuracy:', round(tree.score(X_test, y_test), 4))

imp = pd.Series(tree.feature_importances_, index=X.columns).sort_values(ascending=False).head(5)
print('\nTop 5 feature importances:'); print(imp.round(4))

plt.figure(figsize=(7, 4))
imp.sort_values().plot.barh(color='purple')
plt.xlabel('Importance'); plt.title('Decision Tree: top 5 features'); plt.tight_layout(); plt.show()
