
# ==============================================================================
# Working
# Q7. Diabetes: Simple Linear Regression on BMI
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load the diabetes dataset
diabetes = load_diabetes()

# BMI is feature index 2
X = diabetes.data[:, [2]]
y = diabetes.target

# 2. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Make predictions
pred = model.predict(X_test)

# 5. Display evaluation metrics
print("=" * 45)
print("SIMPLE LINEAR REGRESSION: BMI")
print("=" * 45)

print(f"Intercept : {model.intercept_:.3f}")
print(f"Slope     : {model.coef_[0]:.3f}")
print(f"MAE       : {mean_absolute_error(y_test, pred):.3f}")
print(f"R-squared : {r2_score(y_test, pred):.3f}")

# 6. Sort BMI values for a correctly drawn regression line
order = np.argsort(X_test[:, 0])
X_sorted = X_test[order]
pred_sorted = model.predict(X_sorted)

# 7. Plot actual values and regression line
plt.figure(figsize=(7, 5))

plt.scatter(
    X_test[:, 0], y_test,
    alpha=0.7, label="Actual values"
)

plt.plot(
    X_sorted[:, 0], pred_sorted,
    color="red", linewidth=2,
    label="Regression line"
)

plt.xlabel("BMI (standardized)")
plt.ylabel("Disease progression")
plt.title("Linear Regression: BMI vs Diabetes Progression")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()
