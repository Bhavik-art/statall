# ==============================================================================
# Q7. Diabetes: simple linear regression on BMI
# ==============================================================================
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
diabetes = load_diabetes(as_frame=True)
X, y = diabetes.data, diabetes.target
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

Xb = X[['bmi']]
X_train, X_test, y_train, y_test = train_test_split(Xb, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)
pred = model.predict(X_test)

print('Intercept:', round(model.intercept_, 3))
print('Slope    :', round(model.coef_[0], 3))
print('MAE      :', round(mean_absolute_error(y_test, pred), 3))
print('R2       :', round(r2_score(y_test, pred), 3))

plt.figure(figsize=(6, 4))
plt.scatter(X_test, y_test, alpha=0.7, label='Actual')
plt.plot(X_test.sort_values('bmi'), model.predict(X_test.sort_values('bmi')), color='red', label='Regression line')
plt.xlabel('BMI'); plt.ylabel('Disease progression'); plt.title('Linear regression: BMI vs target'); plt.legend(); plt.show()
