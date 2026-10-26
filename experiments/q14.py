
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Train Decision Tree
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# Find top 5 features
importance = model.feature_importances_
indices = np.argsort(importance)[-5:]

# Plot bar chart
plt.barh(data.feature_names[indices], importance[indices])
plt.xlabel("Feature Importance")
plt.title("Top 5 Important Features")
plt.show()
