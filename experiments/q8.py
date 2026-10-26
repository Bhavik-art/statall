
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load the Diabetes dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_pred = linear_model.predict(X_test)
linear_r2 = r2_score(y_test, linear_pred)

# Ridge Regression
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
ridge_pred = ridge_model.predict(X_test)
ridge_r2 = r2_score(y_test, ridge_pred)

# Compare R2 scores
print("Linear Regression R2:", round(linear_r2, 4))
print("Ridge Regression R2:", round(ridge_r2, 4))

if linear_r2 > ridge_r2:
    print("Linear Regression performs better.")
elif ridge_r2 > linear_r2:
    print("Ridge Regression performs better.")
else:
    print("Both models perform equally.")
