
# ==============================================================================
# Working
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
