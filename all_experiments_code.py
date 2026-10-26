# ==============================================================================
# Statistics for Machine Learning and Data Science: Practical Exam
# Complete Code for All 15 Experiments / Questions
# ==============================================================================


# ==============================================================================
# Q1. Iris: mean, median, standard deviation and box plot
# ==============================================================================
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.datasets import load_iris
iris = load_iris(as_frame=True)
df = iris.data
stats = pd.DataFrame({'Mean': df.mean(), 'Median': df.median(), 'Std Dev': df.std()}).round(3)
print(stats)

plt.figure(figsize=(5, 4))
plt.boxplot(df['petal length (cm)'], patch_artist=True)
plt.title('Box plot of petal length (cm)')
plt.ylabel('cm')
plt.show()


# ==============================================================================
# Q2. Diabetes: BMI distribution
# ==============================================================================
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
diabetes = load_diabetes(as_frame=True)
X, y = diabetes.data, diabetes.target
bmi = X['bmi']
print(bmi.describe().round(4))
print('Skewness:', round(bmi.skew(), 4))
print('Kurtosis:', round(bmi.kurt(), 4))

plt.figure(figsize=(6, 4))
plt.hist(bmi, bins=25, edgecolor='black')
plt.axvline(bmi.mean(), color='red', linestyle='--', label='Mean')
plt.axvline(bmi.median(), color='green', linestyle='--', label='Median')
plt.title('Histogram of BMI (diabetes dataset)')
plt.xlabel('BMI (standardized)'); plt.ylabel('Frequency'); plt.legend()
plt.show()


# ==============================================================================
# Q3. Iris: Pearson correlation heatmap
# ==============================================================================
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.datasets import load_iris
iris = load_iris(as_frame=True)
df = iris.data
corr = df.corr(method='pearson')
print(corr.round(3))

plt.figure(figsize=(6, 5))
plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(label='Pearson r')
plt.xticks(range(4), corr.columns, rotation=45, ha='right'); plt.yticks(range(4), corr.columns)
for i in range(4):
    for j in range(4):
        plt.text(j, i, f'{corr.iloc[i, j]:.2f}', ha='center', va='center')
plt.title('Correlation heatmap (Iris)'); plt.tight_layout(); plt.show()

pairs = corr.abs().where(~np.eye(4, dtype=bool)).stack()
a, b = pairs.idxmax()
print(f'\nStrongest pair: {a} vs {b}  (r = {corr.loc[a, b]:.3f})')


# ==============================================================================
# Q4. Euclidean distance before and after standardization
# ==============================================================================
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import euclidean

data = pd.DataFrame({'Age': [25, 32, 47, 51, 38], 'Salary': [30000, 45000, 90000, 120000, 60000]},
                    index=['A', 'B', 'C', 'D', 'E'])
print('Original data:'); print(data)

scaled = pd.DataFrame(StandardScaler().fit_transform(data), columns=data.columns, index=data.index)
print('\nStandardized data:'); print(scaled.round(3))

d_before = euclidean(data.loc['A'], data.loc['B'])
d_after = euclidean(scaled.loc['A'], scaled.loc['B'])
print(f'\nDistance A-B before scaling: {d_before:.3f}')
print(f'Distance A-B after scaling : {d_after:.3f}')


# ==============================================================================
# Q5. One-sample t-test
# ==============================================================================
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy import stats

data = np.array([52, 48, 51, 49, 54, 50, 53, 47])
mu0, alpha = 50, 0.05
t_stat, p_value = stats.ttest_1samp(data, popmean=mu0)

print('Sample mean :', data.mean())
print('Sample std  :', round(data.std(ddof=1), 4))
print('t-statistic :', round(t_stat, 4))
print('p-value     :', round(p_value, 4))
print()
if p_value < alpha:
    print('Reject H0: the mean differs significantly from 50.')
else:
    print('Fail to reject H0: no significant difference from 50 at the 5% level.')


# ==============================================================================
# Q6. Bootstrap 95% confidence interval for the mean
# ==============================================================================
import numpy as np, pandas as pd, matplotlib.pyplot as plt
data = np.array([12, 15, 14, 10, 18, 16, 13, 17, 11, 14])
rng = np.random.default_rng(42)
n_boot = 10000
boot_means = np.array([rng.choice(data, size=len(data), replace=True).mean() for _ in range(n_boot)])

lo, hi = np.percentile(boot_means, [2.5, 97.5])
print('Sample mean            :', data.mean())
print('Bootstrap mean         :', round(boot_means.mean(), 3))
print(f'95% bootstrap CI       : ({lo:.3f}, {hi:.3f})')

plt.figure(figsize=(6, 4))
plt.hist(boot_means, bins=40, edgecolor='black')
plt.axvline(lo, color='red', linestyle='--'); plt.axvline(hi, color='red', linestyle='--', label='95% CI')
plt.title('Bootstrap distribution of the mean'); plt.xlabel('Mean'); plt.legend(); plt.show()


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


# ==============================================================================
# Q9. Iris: Decision Tree classifier
# ==============================================================================
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42, stratify=iris.target)
clf = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
pred = clf.predict(X_test)

print('Accuracy:', round(accuracy_score(y_test, pred), 4))
print('\nConfusion matrix:'); print(confusion_matrix(y_test, pred))
print('\nClassification report:')
print(classification_report(y_test, pred, target_names=iris.target_names))

ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=iris.target_names, cmap='Blues')
plt.title('Decision Tree: confusion matrix'); plt.show()


# ==============================================================================
# Q10. Iris: Random Forest classifier
# ==============================================================================
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42, stratify=iris.target)
clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
pred = clf.predict(X_test)

print('Accuracy:', round(accuracy_score(y_test, pred), 4))
print('\nConfusion matrix:'); print(confusion_matrix(y_test, pred))
print('\nClassification report:')
print(classification_report(y_test, pred, target_names=iris.target_names))

ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=iris.target_names, cmap='Greens')
plt.title('Random Forest: confusion matrix'); plt.show()


# ==============================================================================
# Q11. Iris: K-Means with Silhouette Score
# ==============================================================================
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.datasets import load_iris
iris = load_iris(as_frame=True)
df = iris.data
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

km = KMeans(n_clusters=3, n_init=10, random_state=42)
labels = km.fit_predict(df)
print('Cluster sizes:', np.bincount(labels))
print('Silhouette Score:', round(silhouette_score(df, labels), 4))

plt.figure(figsize=(6, 4.5))
plt.scatter(df.iloc[:, 2], df.iloc[:, 3], c=labels, cmap='viridis', alpha=0.8)
plt.scatter(km.cluster_centers_[:, 2], km.cluster_centers_[:, 3], c='red', marker='X', s=200, label='Centroids')
plt.xlabel('petal length (cm)'); plt.ylabel('petal width (cm)'); plt.title('K-Means (k = 3)'); plt.legend(); plt.show()


# ==============================================================================
# Q12. Iris: PCA to two components
# ==============================================================================
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

iris = load_iris()
X_std = StandardScaler().fit_transform(iris.data)
pca = PCA(n_components=2)
Z = pca.fit_transform(X_std)

print('Explained variance ratio:', pca.explained_variance_ratio_.round(4))
print('Total variance kept     :', round(pca.explained_variance_ratio_.sum(), 4))

plt.figure(figsize=(6, 4.5))
for i, name in enumerate(iris.target_names):
    plt.scatter(Z[iris.target == i, 0], Z[iris.target == i, 1], label=name, alpha=0.8)
plt.xlabel('PC1'); plt.ylabel('PC2'); plt.title('PCA of Iris (2 components)'); plt.legend(); plt.show()


# ==============================================================================
# Q13. Breast Cancer: 5-fold cross-validation
# ==============================================================================
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
bc = load_breast_cancer(as_frame=True)
X, y = bc.data, bc.target
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

for i, s in enumerate(scores, 1):
    print(f'Fold {i}: {s:.4f}')
print('\nMean accuracy:', round(scores.mean(), 4))
print('Std deviation:', round(scores.std(), 4))

plt.figure(figsize=(6, 4))
plt.bar(range(1, 6), scores, color='teal'); plt.axhline(scores.mean(), color='red', linestyle='--', label='Mean')
plt.ylim(0.9, 1.0); plt.xlabel('Fold'); plt.ylabel('Accuracy'); plt.title('5-fold cross-validation'); plt.legend(); plt.show()


# ==============================================================================
# Q14. Breast Cancer: Decision Tree feature importance
# ==============================================================================
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
bc = load_breast_cancer(as_frame=True)
X, y = bc.data, bc.target
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
tree = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
print('Test accuracy:', round(tree.score(X_test, y_test), 4))

imp = pd.Series(tree.feature_importances_, index=X.columns).sort_values(ascending=False).head(5)
print('\nTop 5 feature importances:'); print(imp.round(4))

plt.figure(figsize=(7, 4))
imp.sort_values().plot.barh(color='purple')
plt.xlabel('Importance'); plt.title('Decision Tree: top 5 features'); plt.tight_layout(); plt.show()


# ==============================================================================
# Q15. Breast Cancer: permutation importance (Random Forest)
# ==============================================================================
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
bc = load_breast_cancer(as_frame=True)
X, y = bc.data, bc.target
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
rf = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_train, y_train)
print('Test accuracy:', round(rf.score(X_test, y_test), 4))

r = permutation_importance(rf, X_test, y_test, n_repeats=30, random_state=42, scoring='accuracy')
top = pd.DataFrame({'mean': r.importances_mean, 'std': r.importances_std}, index=X.columns).sort_values('mean', ascending=False).head(5)
print('\nTop 5 features (permutation importance):'); print(top.round(4))

plt.figure(figsize=(7, 4))
plt.barh(top.index[::-1], top['mean'][::-1], xerr=top['std'][::-1], color='darkorange')
plt.xlabel('Mean decrease in accuracy'); plt.title('Permutation importance (test set)'); plt.tight_layout(); plt.show()
