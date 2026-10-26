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
