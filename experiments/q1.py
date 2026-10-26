# ==============================================================================
# Q1. Iris: mean, median, standard deviation and box plot
# ==============================================================================
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.datasets import load_iris
iris = load_iris(as_frame=True)
df = iris.data
stats = pd.DataFrame({'Mean': df.mean(), 'Median': df.median(), 'Std Dev': df.std()}).round(3)
print(stats)

plt.figure(figsize=(5, 4))
plt.boxplot(df['petal length (cm)'], patch_artist=True)
plt.title('Box plot of petal length (cm)')
plt.ylabel('cm')
plt.show()
