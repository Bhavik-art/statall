
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import pandas as pd

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Calculate permutation importance
result = permutation_importance(
    model, X_test, y_test, n_repeats=10, random_state=42
)

# Display top five features
importance = pd.Series(
    result.importances_mean, index=data.feature_names
)
print("Top 5 Important Features:")
print(importance.sort_values(ascending=False).head(5))
