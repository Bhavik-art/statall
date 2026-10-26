
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Create model
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=5000)
)

# 5-fold cross-validation
scores = cross_val_score(model, X, y, cv=5)

# Display results
print("Fold Scores:", scores)
print("Mean Accuracy:", round(scores.mean(), 4))
