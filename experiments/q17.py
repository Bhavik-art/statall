# =========================================================
# Experiment 04 / 8
# Central Limit Theorem (CLT) Simulation
# =========================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

# --------------------------------------------
# Load Dataset
# --------------------------------------------
housing = fetch_california_housing(as_frame=True)
df = housing.frame
print("=" * 60)
print("First Five Records")
print("=" * 60)
print(df.head())

# --------------------------------------------
# Select Feature
# --------------------------------------------
population = df["MedInc"]
print("\nPopulation Size =", len(population))

# --------------------------------------------
# CLT Simulation
# --------------------------------------------
sample_size = 30
num_samples = 1000
sample_means = []

for i in range(num_samples):
    sample = population.sample(sample_size, replace=True)
    sample_means.append(sample.mean())

sample_means = np.array(sample_means)
print("\nAverage of Sample Means =", sample_means.mean())
print("Standard Deviation =", sample_means.std())
