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
