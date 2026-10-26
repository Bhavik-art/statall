
# ==============================================================================
# Working
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
