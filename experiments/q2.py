
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
