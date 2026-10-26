# ==============================================================================
# Q15. Breast Cancer: permutation importance (Random Forest)
# ==============================================================================
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
bc = load_breast_cancer(as_frame=True)
X, y = bc.data, bc.target
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
rf = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_train, y_train)
print('Test accuracy:', round(rf.score(X_test, y_test), 4))

r = permutation_importance(rf, X_test, y_test, n_repeats=30, random_state=42, scoring='accuracy')
top = pd.DataFrame({'mean': r.importances_mean, 'std': r.importances_std}, index=X.columns).sort_values('mean', ascending=False).head(5)
print('\nTop 5 features (permutation importance):'); print(top.round(4))

plt.figure(figsize=(7, 4))
plt.barh(top.index[::-1], top['mean'][::-1], xerr=top['std'][::-1], color='darkorange')
plt.xlabel('Mean decrease in accuracy'); plt.title('Permutation importance (test set)'); plt.tight_layout(); plt.show()
