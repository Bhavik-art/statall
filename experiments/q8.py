
# ==============================================================================
# Working
# Q8. Linear vs Ridge Regression
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score

# 1. Load the diabetes dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# 2. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Train Linear Regression
lin = LinearRegression()
lin.fit(X_train, y_train)

# 4. Train Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

# 5. Make predictions
pred_lin = lin.predict(X_test)
pred_ridge = ridge.predict(X_test)

# 6. Calculate test R-squared scores
r2_lin = r2_score(y_test, pred_lin)
r2_ridge = r2_score(y_test, pred_ridge)

print("=" * 45)
print("LINEAR VS RIDGE REGRESSION")
print("=" * 45)

print(f"Linear Regression test R2: {r2_lin:.4f}")
print(f"Ridge Regression  test R2: {r2_ridge:.4f}")

if r2_ridge > r2_lin:
    print("Better model based on test R2: Ridge Regression")
elif r2_lin > r2_ridge:
    print("Better model based on test R2: Linear Regression")
else:
    print("Both models have equal test R2 scores.")

# 7. Plot comparison
models = ["Linear", "Ridge"]
scores = [r2_lin, r2_ridge]

plt.figure(figsize=(6, 4))
bars = plt.bar(models, scores, color=["steelblue", "darkorange"])

plt.ylabel("Test R-squared (R2)")
plt.title("Linear Regression vs Ridge Regression")
plt.axhline(0, color="black", linewidth=0.8)

# Display score values above the bars
for bar, score in zip(bars, scores):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        score,
        f"{score:.4f}",
        ha="center",
        va="bottom" if score >= 0 else "top"
    )

plt.tight_layout()
plt.show()
