# ==============================================================================
# Statistics for Machine Learning and Data Science: Practical Exam
# Complete Python Code for All 15 Experiments
# ==============================================================================

# ==============================================================================
# Q1. Iris: Mean, Median, Standard Deviation and Box Plot
# ==============================================================================

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()

# Convert dataset into a DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Calculate mean, median, and standard deviation
print("Mean:")
print(df.mean())

print("\nMedian:")
print(df.median())

print("\nStandard Deviation:")
print(df.std())

# Generate a box plot for sepal length
plt.boxplot(df["sepal length (cm)"])

plt.title("Box Plot of Sepal Length")
plt.ylabel("Sepal Length (cm)")
plt.show()


# ==============================================================================
# Q2. Diabetes: BMI Distribution and Histogram
# ==============================================================================

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes

# Load the Diabetes dataset
diabetes = load_diabetes()

# Convert dataset into a DataFrame
df = pd.DataFrame(
    diabetes.data,
    columns=diabetes.feature_names
)

# Select BMI feature
bmi = df["bmi"]

# Calculate descriptive statistics
print("Descriptive Statistics of BMI:")
print(bmi.describe())

# Generate histogram
plt.hist(bmi, bins=20, edgecolor="black")

plt.title("Histogram of BMI Feature")
plt.xlabel("BMI (Standardized)")
plt.ylabel("Frequency")
plt.show()


# ==============================================================================
# Q3. Iris: Pearson Correlation and Heatmap
# ==============================================================================

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()

# Create DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Calculate Pearson correlation matrix
corr_matrix = df.corr(method='pearson')

print("Pearson Correlation Matrix:")
print(corr_matrix.round(2))

# Generate heatmap using Matplotlib
plt.figure(figsize=(8, 6))

plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)

plt.colorbar(label="Pearson Correlation")

# Add feature names
plt.xticks(range(len(df.columns)), df.columns,
           rotation=45, ha='right')
plt.yticks(range(len(df.columns)), df.columns)

# Display correlation values
for i in range(len(corr_matrix.columns)):
    for j in range(len(corr_matrix.columns)):
        plt.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                 ha='center', va='center')

plt.title("Pearson Correlation Heatmap - Iris Dataset")
plt.tight_layout()
plt.show()

# Find the strongest absolute correlation
corr_pairs = corr_matrix.abs().where(
    ~__import__('numpy').eye(len(corr_matrix), dtype=bool)
).stack()

strongest_pair = corr_pairs.idxmax()
strongest_value = corr_matrix.loc[strongest_pair[0],
                                  strongest_pair[1]]

print("\nStrongest Correlated Feature Pair:")
print(strongest_pair[0], "and", strongest_pair[1])
print("Correlation:", round(strongest_value, 2))


# ==============================================================================
# Q4. Euclidean Distance Before and After Standardization
# ==============================================================================

import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import euclidean

# Create a dataset with different scales
data = {
    'Age': [20, 25, 30, 35],
    'Salary': [20000, 30000, 40000, 50000]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

# Calculate Euclidean distance before standardization
distance_before = euclidean(df.iloc[0], df.iloc[1])
print("\nDistance Before Standardization:", round(distance_before, 2))

# Apply standardization
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

scaled_df = pd.DataFrame(scaled_data, columns=df.columns)

print("\nStandardized Dataset:")
print(scaled_df.round(2))

# Calculate Euclidean distance after standardization
distance_after = euclidean(scaled_df.iloc[0], scaled_df.iloc[1])
print("\nDistance After Standardization:", round(distance_after, 2))


# ==============================================================================
# Q5. One-Sample t-Test
# ==============================================================================

import numpy as np
from scipy.stats import ttest_1samp

# Given observations
data = [52, 48, 51, 49, 54, 50, 53, 47]

# Perform one-sample t-test
t_stat, p_value = ttest_1samp(data, 50)

print("Sample Mean:", np.mean(data))
print("T-statistic:", round(t_stat, 4))
print("P-value:", round(p_value, 4))

# Decision
if p_value < 0.05:
    print("Reject the null hypothesis.")
    print("The population mean differs significantly from 50.")
else:
    print("Fail to reject the null hypothesis.")
    print("The population mean does not differ significantly from 50.")


# ==============================================================================
# Q6. Bootstrap 95% Confidence Interval for the Mean
# ==============================================================================

import numpy as np

# Given dataset
data = np.array([12, 15, 14, 10, 18, 16, 13, 17, 11, 14])

# Number of bootstrap samples
n_iterations = 10000

# Store bootstrap means
bootstrap_means = []

# Bootstrap resampling
for i in range(n_iterations):
    sample = np.random.choice(data, size=len(data), replace=True)
    bootstrap_means.append(np.mean(sample))

# Calculate 95% confidence interval
lower = np.percentile(bootstrap_means, 2.5)
upper = np.percentile(bootstrap_means, 97.5)

print("Sample Mean:", np.mean(data))
print("95% Confidence Interval:", (round(lower, 2), round(upper, 2)))


# ==============================================================================
# Q7. Diabetes: Simple Linear Regression
# ==============================================================================

from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Load the diabetes dataset
diabetes = load_diabetes()

# Select BMI feature
X = diabetes.data[:, [2]]
y = diabetes.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict target values
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error (MAE):", round(mae, 2))
print("R2 Score:", round(r2, 4))


# ==============================================================================
# Q8. Diabetes: Linear Regression vs Ridge Regression
# ==============================================================================

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


# ==============================================================================
# Q9. Iris: Logistic Regression Classifier
# ==============================================================================

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average="weighted"))
print("Recall:", recall_score(y_test, y_pred, average="weighted"))
print("F1 Score:", f1_score(y_test, y_pred, average="weighted"))


# ==============================================================================
# Q10. Iris: Decision Tree Classifier
# ==============================================================================

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict species
y_pred = model.predict(X_test)

# Evaluate model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average="weighted"))
print("Recall:", recall_score(y_test, y_pred, average="weighted"))
print("F1 Score:", f1_score(y_test, y_pred, average="weighted"))


# ==============================================================================
# Q11. Iris: K-Means Clustering and Silhouette Score
# ==============================================================================

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load dataset
iris = load_iris()
X = iris.data

# Apply K-Means clustering
model = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = model.fit_predict(X)

# Calculate Silhouette Score
score = silhouette_score(X, labels)

print("Number of Clusters:", 3)
print("Silhouette Score:", round(score, 4))


# ==============================================================================
# Q12. Iris: Principal Component Analysis (PCA)
# ==============================================================================

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Standardize features
X = StandardScaler().fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot results
for i, name in enumerate(iris.target_names):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], label=name)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Visualization of Iris Dataset")
plt.legend()
plt.show()


# ==============================================================================
# Q13. Breast Cancer: 5-Fold Cross-Validation
# ==============================================================================

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


# ==============================================================================
# Q14. Breast Cancer: Decision Tree Feature Importance
# ==============================================================================

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


# ==============================================================================
# Q15. Breast Cancer: Random Forest Permutation Importance
# ==============================================================================

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


# ==============================================================================
# Q16. Breast Cancer: Maximum Likelihood Estimation (MLE) - Bernoulli Distribution
# ==============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

# Load Dataset
data = load_breast_cancer(as_frame=True)
df = data.frame
df["target"] = data.target
print("=" * 60)
print("First Five Records")
print("=" * 60)
print(df.head())

# Bernoulli Distribution MLE
target = df["target"]
p_hat = np.mean(target)
print("Maximum Likelihood Estimation (MLE)")
print("Estimated Probability =", round(p_hat, 4))
print("Malignant =", sum(target == 0))
print("Benign =", sum(target == 1))

target.value_counts().plot(kind="bar", color=["green", "red"])
plt.title("Bernoulli Distribution - Target Classes")
plt.xlabel("Target Class")
plt.ylabel("Count")
plt.xticks([0, 1], ["Benign", "Malignant"], rotation=0)
plt.show()


# ==============================================================================
# Q17. California Housing: Central Limit Theorem (CLT) Simulation
# ==============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

# Load Dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame
print("=" * 60)
print("First Five Records")
print("=" * 60)
print(df.head())

# Select Feature
population = df["MedInc"]
print("\nPopulation Size =", len(population))

# CLT Simulation
sample_size = 30
num_samples = 1000
sample_means = []

for i in range(num_samples):
    sample = population.sample(sample_size, replace=True)
    sample_means.append(sample.mean())

sample_means = np.array(sample_means)
print("\nAverage of Sample Means =", sample_means.mean())
print("Standard Deviation =", sample_means.std())


# ==============================================================================
# Q18. Titanic: Data Preprocessing (Deletion, Mean/Median/Mode Imputation)
# ==============================================================================

import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

print("Missing Values Before Preprocessing:")
print(df.isnull().sum())

# Deletion method
df.dropna(subset=["Embarked"], inplace=True)

# Imputation using mean
df["Age"] = df["Age"].fillna(df["Age"].mean())

# Imputation using median
df["Fare"] = df["Fare"].fillna(df["Fare"].median())

# Imputation using mode
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

print("\nMissing Values After Preprocessing:")
print(df.isnull().sum())

print("\nFirst Five Records:")
print(df.head())
