# ======================================================
# Experiment 07 / 15
# Maximum Likelihood Estimation (MLE)
# ======================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

# -----------------------------------------------------
# Load Dataset
# -----------------------------------------------------
data = load_breast_cancer(as_frame=True)
df = data.frame
df["target"] = data.target
print("=" * 60)
print("First Five Records")
print("=" * 60)
print(df.head())

# -----------------------------------------------------
# Bernoulli Distribution MLE
# -----------------------------------------------------
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
