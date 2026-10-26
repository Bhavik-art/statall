# ==============================================================================
# Q2. Diabetes: BMI distribution
# ==============================================================================
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
diabetes = load_diabetes(as_frame=True)
X, y = diabetes.data, diabetes.target
bmi = X['bmi']
print(bmi.describe().round(4))
print('Skewness:', round(bmi.skew(), 4))
print('Kurtosis:', round(bmi.kurt(), 4))

plt.figure(figsize=(6, 4))
plt.hist(bmi, bins=25, edgecolor='black')
plt.axvline(bmi.mean(), color='red', linestyle='--', label='Mean')
plt.axvline(bmi.median(), color='green', linestyle='--', label='Median')
plt.title('Histogram of BMI (diabetes dataset)')
plt.xlabel('BMI (standardized)'); plt.ylabel('Frequency'); plt.legend()
plt.show()
