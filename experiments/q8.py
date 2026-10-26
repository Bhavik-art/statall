# ==============================================================================
# Q8. Linear vs Ridge regression
# ==============================================================================
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
diabetes = load_diabetes(as_frame=True)
X, y = diabetes.data, diabetes.target
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lin = LinearRegression().fit(X_train, y_train)
ridge = Ridge(alpha=1.0).fit(X_train, y_train)

r2_lin = r2_score(y_test, lin.predict(X_test))
r2_ridge = r2_score(y_test, ridge.predict(X_test))
print('Linear Regression test R2:', round(r2_lin, 4))
print('Ridge Regression  test R2:', round(r2_ridge, 4))
print('Better model:', 'Ridge' if r2_ridge > r2_lin else 'Linear')

plt.figure(figsize=(5, 4))
plt.bar(['Linear', 'Ridge'], [r2_lin, r2_ridge], color=['steelblue', 'darkorange'])
plt.ylabel('Test R2'); plt.title('Linear vs Ridge'); plt.ylim(0, 0.6); plt.show()
