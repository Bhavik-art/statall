
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
