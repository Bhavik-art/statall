# ==============================================================================
# Statistics for Machine Learning and Data Science: Practical Exam
# Complete Code for All 15 Experiments / Questions
# ==============================================================================


# ==============================================================================
# Q1. Iris: Mean, Median, Standard Deviation and Box Plot
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# 1. Load the Iris dataset
iris = load_iris()
data = iris.data
feature_names = iris.feature_names

# 2. Calculate descriptive statistics
mean = np.mean(data, axis=0)
median = np.median(data, axis=0)
std_dev = np.std(data, axis=0, ddof=1)

print("=" * 65)
print("IRIS DATASET: DESCRIPTIVE STATISTICS")
print("=" * 65)
print(f"{'Feature':<25}{'Mean':>12}{'Median':>12}{'Std Dev':>12}")
print("-" * 65)

for i, feature in enumerate(feature_names):
    print(
        f"{feature:<25}"
        f"{mean[i]:>12.3f}"
        f"{median[i]:>12.3f}"
        f"{std_dev[i]:>12.3f}"
    )

# 3. Extract petal length (cm)
petal_length = data[:, 2]

# 4. Create the box plot
plt.figure(figsize=(6, 4))

plt.boxplot(
    petal_length,
    patch_artist=True,
    tick_labels=["Petal Length"]
)
plt.title("Box Plot of Iris Petal Length")
plt.ylabel("Length (cm)")
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()


# ==============================================================================
# Q2. Diabetes: BMI Distribution
# Descriptive Statistics, Skewness, Kurtosis, and Histogram
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes

# 1. Load the diabetes dataset
diabetes = load_diabetes()

# BMI is the second feature (index 2)
bmi = diabetes.data[:, 2]

# 2. Calculate descriptive statistics
count = len(bmi)
mean = np.mean(bmi)
median = np.median(bmi)
std = np.std(bmi, ddof=1)
variance = np.var(bmi, ddof=1)
minimum = np.min(bmi)
maximum = np.max(bmi)
q1 = np.percentile(bmi, 25)
q3 = np.percentile(bmi, 75)

print("=" * 45)
print("DIABETES BMI DISTRIBUTION")
print("=" * 45)

print(f"Count              : {count}")
print(f"Mean               : {mean:.4f}")
print(f"Median             : {median:.4f}")
print(f"Standard Deviation : {std:.4f}")
print(f"Variance           : {variance:.4f}")
print(f"Minimum            : {minimum:.4f}")
print(f"25th Percentile    : {q1:.4f}")
print(f"75th Percentile    : {q3:.4f}")
print(f"Maximum            : {maximum:.4f}")

# 3. Calculate skewness and excess kurtosis
m2 = np.mean((bmi - mean) ** 2)
m3 = np.mean((bmi - mean) ** 3)
m4 = np.mean((bmi - mean) ** 4)

skewness = m3 / (m2 ** 1.5)
kurtosis = (m4 / (m2 ** 2)) - 3

print(f"Skewness           : {skewness:.4f}")
print(f"Excess Kurtosis    : {kurtosis:.4f}")

# 4. Plot the BMI histogram
plt.figure(figsize=(8, 5))

plt.hist(
    bmi,
    bins=25,
    edgecolor="black",
    alpha=0.75
)

plt.axvline(
    mean,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Mean = {mean:.4f}"
)

plt.axvline(
    median,
    color="green",
    linestyle="--",
    linewidth=2,
    label=f"Median = {median:.4f}"
)

plt.title("Histogram of BMI (Diabetes Dataset)")
plt.xlabel("BMI (standardized)")
plt.ylabel("Frequency")
plt.legend()
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()

plt.show()


# ==============================================================================
# Q3. Iris: Pearson Correlation Heatmap
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# 1. Load the Iris dataset
iris = load_iris()
data = iris.data
features = iris.feature_names
n = len(features)

# 2. Calculate Pearson correlation matrix
corr = np.corrcoef(data, rowvar=False)

# 3. Print correlation matrix
print("=" * 75)
print("PEARSON CORRELATION MATRIX (IRIS DATASET)")
print("=" * 75)

print(f"{'Feature':<25}", end="")
for feature in features:
    print(f"{feature[:12]:>14}", end="")
print()

for i, feature in enumerate(features):
    print(f"{feature:<25}", end="")
    for j in range(n):
        print(f"{corr[i, j]:>14.3f}", end="")
    print()

# 4. Plot the correlation heatmap
plt.figure(figsize=(8, 6))

plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="Pearson correlation (r)")

plt.xticks(range(n), features, rotation=45, ha="right")
plt.yticks(range(n), features)

# Display correlation values in each cell
for i in range(n):
    for j in range(n):
        value = corr[i, j]
        text_color = "white" if abs(value) > 0.65 else "black"
        plt.text(
            j, i, f"{value:.2f}",
            ha="center", va="center",
            color=text_color
        )

plt.title("Pearson Correlation Heatmap (Iris)")
plt.tight_layout()
plt.show()

# 5. Find the strongest correlated pair, excluding diagonal values
upper_triangle = np.triu(np.abs(corr), k=1)
i, j = np.unravel_index(np.argmax(upper_triangle), upper_triangle.shape)

print("\nStrongest correlated pair:")
print(f"{features[i]} vs {features[j]}")
print(f"Pearson correlation (r) = {corr[i, j]:.3f}")


# ==============================================================================
# Q4. Euclidean Distance Before and After Standardization
# ==============================================================================

import numpy as np
from sklearn.preprocessing import StandardScaler

# 1. Create the dataset
names = ["A", "B", "C", "D", "E"]

data = np.array([
    [25,  30000],
    [32,  45000],
    [47,  90000],
    [51, 120000],
    [38,  60000]
], dtype=float)

features = ["Age", "Salary"]

# 2. Display original data
print("=" * 50)
print("ORIGINAL DATA")
print("=" * 50)
print(f"{'Person':<12}{'Age':>10}{'Salary':>15}")

for name, row in zip(names, data):
    print(f"{name:<12}{row[0]:>10.0f}{row[1]:>15.0f}")

# 3. Standardize the data
scaler = StandardScaler()
scaled = scaler.fit_transform(data)

print("\nSTANDARDIZED DATA")
print("=" * 50)
print(f"{'Person':<12}{'Age':>12}{'Salary':>12}")

for name, row in zip(names, scaled):
    print(f"{name:<12}{row[0]:>12.3f}{row[1]:>12.3f}")

# 4. Calculate Euclidean distance between A and B
# Before standardization
distance_before = np.linalg.norm(data[0] - data[1])

# After standardization
distance_after = np.linalg.norm(scaled[0] - scaled[1])

print("\nEUCLIDEAN DISTANCE: A vs B")
print("=" * 50)
print(f"Distance before standardization: {distance_before:.3f}")
print(f"Distance after standardization : {distance_after:.3f}")

# 5. Compare the distances
print("\nCONCLUSION")
if distance_before > distance_after:
    print("The numerical distance decreased after standardization.")
else:
    print("The numerical distance increased after standardization.")

print("Standardization puts Age and Salary on comparable scales.")


# ==============================================================================
# Q5. One-Sample t-Test
# ==============================================================================

import numpy as np
import math

# 1. Sample data
data = np.array([52, 48, 51, 49, 54, 50, 53, 47], dtype=float)

mu0 = 50
alpha = 0.05
n = len(data)

# 2. Calculate sample statistics
sample_mean = np.mean(data)
sample_std = np.std(data, ddof=1)
standard_error = sample_std / math.sqrt(n)

# 3. Calculate t-statistic
t_stat = (sample_mean - mu0) / standard_error

# 4. Calculate two-tailed p-value using the t-distribution
# Numerical integration using Simpson's rule
def t_pdf(x, df):
    return (
        math.gamma((df + 1) / 2)
        / (math.sqrt(df * math.pi) * math.gamma(df / 2))
        * (1 + x * x / df) ** (-(df + 1) / 2)
    )

def integrate_simpson(func, a, b, df, steps=10000):
    if steps % 2:
        steps += 1

    h = (b - a) / steps
    total = func(a, df) + func(b, df)

    for i in range(1, steps):
        x = a + i * h
        total += (4 if i % 2 else 2) * func(x, df)

    return total * h / 3

df = n - 1
p_value = 2 * integrate_simpson(
    t_pdf, abs(t_stat), 100, df
)

# 5. Display results
print("=" * 45)
print("ONE-SAMPLE T-TEST")
print("=" * 45)

print(f"Sample size       : {n}")
print(f"Hypothesized mean : {mu0}")
print(f"Sample mean       : {sample_mean:.4f}")
print(f"Sample std        : {sample_std:.4f}")
print(f"Standard error    : {standard_error:.4f}")
print(f"Degrees of freedom: {df}")
print(f"t-statistic       : {t_stat:.4f}")
print(f"p-value           : {p_value:.4f}")
print(f"Significance level: {alpha}")

# 6. Decision
print("\nCONCLUSION")

if p_value < alpha:
    print("Reject H0: the mean differs significantly from 50.")
else:
    print("Fail to reject H0: insufficient evidence that the mean differs from 50.")


# ==============================================================================
# Q6. Bootstrap 95% Confidence Interval for the Mean
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt

# 1. Original sample data
data = np.array([12, 15, 14, 10, 18, 16, 13, 17, 11, 14])

# 2. Bootstrap settings
rng = np.random.default_rng(42)
n_boot = 10000
n = len(data)

# 3. Generate bootstrap samples and calculate their means
boot_means = np.array([
    rng.choice(data, size=n, replace=True).mean()
    for _ in range(n_boot)
])

# 4. Calculate the 95% confidence interval
lo, hi = np.percentile(boot_means, [2.5, 97.5])

# 5. Display results
print("=" * 45)
print("BOOTSTRAP CONFIDENCE INTERVAL")
print("=" * 45)

print(f"Sample size       : {n}")
print(f"Number of samples : {n_boot}")
print(f"Sample mean       : {data.mean():.3f}")
print(f"Bootstrap mean    : {boot_means.mean():.3f}")
print(f"95% Bootstrap CI  : ({lo:.3f}, {hi:.3f})")

# 6. Plot the bootstrap distribution
plt.figure(figsize=(7, 5))

plt.hist(
    boot_means,
    bins=40,
    edgecolor="black",
    alpha=0.75
)

plt.axvline(
    lo, color="red", linestyle="--",
    linewidth=2, label=f"Lower limit = {lo:.3f}"
)

plt.axvline(
    hi, color="red", linestyle="--",
    linewidth=2, label=f"Upper limit = {hi:.3f}"
)

plt.axvline(
    data.mean(), color="green", linestyle="-",
    linewidth=2, label=f"Sample mean = {data.mean():.3f}"
)

plt.title("Bootstrap Distribution of the Mean")
plt.xlabel("Bootstrap Mean")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()


# ==============================================================================
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


# ==============================================================================
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


# ==============================================================================
# Q9. Iris: Decision Tree Classifier
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)

# 1. Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# 2. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# 3. Train the Decision Tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# 4. Predict the test data
pred = clf.predict(X_test)

# 5. Calculate accuracy
accuracy = accuracy_score(y_test, pred)

print("=" * 50)
print("IRIS DECISION TREE CLASSIFIER")
print("=" * 50)
print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

# 6. Display confusion matrix
cm = confusion_matrix(y_test, pred)

print("\nConfusion Matrix:")
print(cm)

# 7. Display classification report
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        pred,
        target_names=iris.target_names,
        zero_division=0
    )
)

# 8. Plot confusion matrix
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names
)

display.plot(cmap="Blues", values_format="d")
plt.title("Decision Tree: Confusion Matrix")
plt.tight_layout()
plt.show()


# ==============================================================================
# Q10. Iris: Random Forest Classifier
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report
)

# 1. Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# 2. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# 3. Train the Random Forest classifier
clf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
clf.fit(X_train, y_train)

# 4. Predict the test data
pred = clf.predict(X_test)

# 5. Calculate accuracy
accuracy = accuracy_score(y_test, pred)

print("=" * 50)
print("IRIS RANDOM FOREST CLASSIFIER")
print("=" * 50)
print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

# 6. Display confusion matrix
cm = confusion_matrix(y_test, pred)

print("\nConfusion Matrix:")
print(cm)

# 7. Display classification report
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        pred,
        target_names=iris.target_names,
        zero_division=0
    )
)

# 8. Plot confusion matrix
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names
)

display.plot(cmap="Greens", values_format="d")
plt.title("Random Forest: Confusion Matrix")
plt.tight_layout()
plt.show()


# ==============================================================================
# Q11. Iris: K-Means with Silhouette Score
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. Load the Iris dataset
iris = load_iris()
X = iris.data
features = iris.feature_names

# 2. Apply K-Means clustering
km = KMeans(
    n_clusters=3,
    n_init=10,
    random_state=42
)

labels = km.fit_predict(X)

# 3. Calculate cluster sizes
cluster_sizes = np.bincount(labels)

print("=" * 45)
print("IRIS K-MEANS CLUSTERING")
print("=" * 45)

for i, size in enumerate(cluster_sizes):
    print(f"Cluster {i}: {size} samples")

# 4. Calculate Silhouette Score
score = silhouette_score(X, labels)
print(f"\nSilhouette Score: {score:.4f}")

# 5. Plot clusters using petal length and petal width
plt.figure(figsize=(7, 5))

plt.scatter(
    X[:, 2],
    X[:, 3],
    c=labels,
    cmap="viridis",
    alpha=0.8,
    edgecolors="black",
    label="Data points"
)

# Plot cluster centroids
plt.scatter(
    km.cluster_centers_[:, 2],
    km.cluster_centers_[:, 3],
    c="red",
    marker="X",
    s=200,
    edgecolors="black",
    label="Centroids"
)

plt.xlabel("Petal length (cm)")
plt.ylabel("Petal width (cm)")
plt.title("K-Means Clustering (k = 3)")
plt.legend()
plt.tight_layout()
plt.show()


# ==============================================================================
# Q12. Iris: PCA to Two Components
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# 2. Standardize the features
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# 3. Apply PCA to reduce 4 features to 2 components
pca = PCA(n_components=2)
Z = pca.fit_transform(X_std)

# 4. Display explained variance
print("=" * 45)
print("PCA: IRIS DATASET")
print("=" * 45)

print("Explained variance ratio:",
      np.round(pca.explained_variance_ratio_, 4))

print("Total variance kept      :",
      round(pca.explained_variance_ratio_.sum(), 4))

print(f"Variance retained        : "
      f"{pca.explained_variance_ratio_.sum() * 100:.2f}%")

# 5. Plot the two principal components
plt.figure(figsize=(7, 5))

for i, name in enumerate(iris.target_names):
    plt.scatter(
        Z[y == i, 0],
        Z[y == i, 1],
        label=name,
        alpha=0.8
    )

plt.xlabel("Principal Component 1 (PC1)")
plt.ylabel("Principal Component 2 (PC2)")
plt.title("PCA of Iris Dataset (2 Components)")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()


# ==============================================================================
# Q13. Breast Cancer: 5-Fold Cross-Validation
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

# 1. Load the Breast Cancer dataset
bc = load_breast_cancer()
X = bc.data
y = bc.target

# 2. Create the machine learning pipeline
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

# 3. Configure 5-fold stratified cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# 4. Evaluate model using accuracy
scores = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="accuracy"
)

# 5. Print fold accuracies
print("=" * 45)
print("5-FOLD CROSS-VALIDATION RESULTS")
print("=" * 45)

for i, score in enumerate(scores, start=1):
    print(f"Fold {i} accuracy: {score:.4f}")

# 6. Calculate mean and standard deviation
mean_accuracy = scores.mean()
std_deviation = scores.std()

print(f"\nMean accuracy : {mean_accuracy:.4f}")
print(f"Std deviation : {std_deviation:.4f}")
print(f"Mean accuracy (%): {mean_accuracy * 100:.2f}%")

# 7. Plot fold accuracies
plt.figure(figsize=(7, 4.5))

plt.bar(
    range(1, 6),
    scores,
    color="teal",
    edgecolor="black"
)

plt.axhline(
    mean_accuracy,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Mean = {mean_accuracy:.4f}"
)

plt.xticks(range(1, 6))
plt.ylim(0.90, 1.0)
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.title("5-Fold Cross-Validation")
plt.legend()
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.show()


# ==============================================================================
# Q14. Breast Cancer: Decision Tree Feature Importance
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# 1. Load the Breast Cancer dataset
bc = load_breast_cancer()
X = bc.data
y = bc.target
feature_names = bc.feature_names

# 2. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Train the Decision Tree classifier
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)

# 4. Evaluate test accuracy
accuracy = tree.score(X_test, y_test)

print("=" * 50)
print("DECISION TREE FEATURE IMPORTANCE")
print("=" * 50)
print(f"Test accuracy: {accuracy:.4f}")

# 5. Find the five most important features
importances = tree.feature_importances_
top_indices = np.argsort(importances)[::-1][:5]

top_names = feature_names[top_indices]
top_values = importances[top_indices]

print("\nTop 5 Feature Importances:")
for name, value in zip(top_names, top_values):
    print(f"{name:<30} {value:.4f}")

# 6. Sort for horizontal bar chart
order = np.argsort(top_values)

plot_names = top_names[order]
plot_values = top_values[order]

# 7. Plot feature importance
plt.figure(figsize=(8, 5))

plt.barh(
    plot_names,
    plot_values,
    color="purple",
    edgecolor="black"
)

plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Decision Tree: Top 5 Important Features")
plt.grid(axis="x", alpha=0.25)
plt.tight_layout()
plt.show()


# ==============================================================================
# Q15. Breast Cancer: Permutation Importance (Random Forest)
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

# 1. Load the Breast Cancer dataset
bc = load_breast_cancer()
X = bc.data
y = bc.target
feature_names = bc.feature_names

# 2. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Train the Random Forest classifier
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
rf.fit(X_train, y_train)

# 4. Evaluate test accuracy
accuracy = rf.score(X_test, y_test)

print("=" * 55)
print("RANDOM FOREST: PERMUTATION IMPORTANCE")
print("=" * 55)
print(f"Test accuracy: {accuracy:.4f}")

# 5. Calculate permutation importance on test data
result = permutation_importance(
    rf,
    X_test,
    y_test,
    n_repeats=30,
    random_state=42,
    scoring="accuracy"
)

# 6. Select the five most important features
importance_mean = result.importances_mean
importance_std = result.importances_std

top_indices = np.argsort(importance_mean)[::-1][:5]

top_names = feature_names[top_indices]
top_means = importance_mean[top_indices]
top_stds = importance_std[top_indices]

print("\nTop 5 Features (Permutation Importance):")
print(f"{'Feature':<30}{'Mean':>10}{'Std Dev':>12}")
print("-" * 52)

for name, mean, std in zip(top_names, top_means, top_stds):
    print(f"{name:<30}{mean:>10.4f}{std:>12.4f}")

# 7. Sort features for the horizontal bar chart
order = np.argsort(top_means)

plot_names = top_names[order]
plot_means = top_means[order]
plot_stds = top_stds[order]

# 8. Plot permutation importance
plt.figure(figsize=(8, 5))

plt.barh(
    plot_names,
    plot_means,
    xerr=plot_stds,
    color="darkorange",
    capsize=4,
    edgecolor="black"
)

plt.axvline(0, color="black", linewidth=0.8)
plt.xlabel("Mean Decrease in Accuracy")
plt.ylabel("Feature")
plt.title("Permutation Importance (Test Set)")
plt.grid(axis="x", alpha=0.25)
plt.tight_layout()
plt.show()
