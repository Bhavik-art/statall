# ==============================================================================
# Q13. Breast Cancer: 5-fold cross-validation
# ==============================================================================
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
bc = load_breast_cancer(as_frame=True)
X, y = bc.data, bc.target
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

for i, s in enumerate(scores, 1):
    print(f'Fold {i}: {s:.4f}')
print('\nMean accuracy:', round(scores.mean(), 4))
print('Std deviation:', round(scores.std(), 4))

plt.figure(figsize=(6, 4))
plt.bar(range(1, 6), scores, color='teal'); plt.axhline(scores.mean(), color='red', linestyle='--', label='Mean')
plt.ylim(0.9, 1.0); plt.xlabel('Fold'); plt.ylabel('Accuracy'); plt.title('5-fold cross-validation'); plt.legend(); plt.show()
