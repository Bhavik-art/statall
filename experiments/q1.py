
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
