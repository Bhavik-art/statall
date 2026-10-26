import numpy as np
from scipy.stats import ttest_1samp

# Given observations
data = [52, 48, 51, 49, 54, 50, 53, 47]

# Perform one-sample t-test
t_stat, p_value = ttest_1samp(data, 50)

print("Sample Mean:", np.mean(data))
print("T-statistic:", round(t_stat, 4))
print("P-value:", round(p_value, 4))

# Decision
if p_value < 0.05:
    print("Reject the null hypothesis.")
    print("The population mean differs significantly from 50.")
else:
    print("Fail to reject the null hypothesis.")
    print("The population mean does not differ significantly from 50.")