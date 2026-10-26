
# ==============================================================================
# Working
# Q5. One-Sample t-Test
# ==============================================================================

import numpy as np
import math

# 1. Sample data
data = np.array([52, 48, 51, 49, 54, 50, 53, 47], dtype=float)

mu0 = 50
alpha = 0.05
n = len(data)

# 2. Calculate sample statistics
sample_mean = np.mean(data)
sample_std = np.std(data, ddof=1)
standard_error = sample_std / math.sqrt(n)

# 3. Calculate t-statistic
t_stat = (sample_mean - mu0) / standard_error

# 4. Calculate two-tailed p-value using the t-distribution
# Numerical integration using Simpson's rule
def t_pdf(x, df):
    return (
        math.gamma((df + 1) / 2)
        / (math.sqrt(df * math.pi) * math.gamma(df / 2))
        * (1 + x * x / df) ** (-(df + 1) / 2)
    )

def integrate_simpson(func, a, b, df, steps=10000):
    if steps % 2:
        steps += 1

    h = (b - a) / steps
    total = func(a, df) + func(b, df)

    for i in range(1, steps):
        x = a + i * h
        total += (4 if i % 2 else 2) * func(x, df)

    return total * h / 3

df = n - 1
p_value = 2 * integrate_simpson(
    t_pdf, abs(t_stat), 100, df
)

# 5. Display results
print("=" * 45)
print("ONE-SAMPLE T-TEST")
print("=" * 45)

print(f"Sample size       : {n}")
print(f"Hypothesized mean : {mu0}")
print(f"Sample mean       : {sample_mean:.4f}")
print(f"Sample std        : {sample_std:.4f}")
print(f"Standard error    : {standard_error:.4f}")
print(f"Degrees of freedom: {df}")
print(f"t-statistic       : {t_stat:.4f}")
print(f"p-value           : {p_value:.4f}")
print(f"Significance level: {alpha}")

# 6. Decision
print("\nCONCLUSION")

if p_value < alpha:
    print("Reject H0: the mean differs significantly from 50.")
else:
    print("Fail to reject H0: insufficient evidence that the mean differs from 50.")
