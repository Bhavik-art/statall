# ==============================================================================
# Q3. Iris: Pearson correlation heatmap
# ==============================================================================
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.datasets import load_iris
iris = load_iris(as_frame=True)
df = iris.data
corr = df.corr(method='pearson')
print(corr.round(3))

plt.figure(figsize=(6, 5))
plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(label='Pearson r')
plt.xticks(range(4), corr.columns, rotation=45, ha='right'); plt.yticks(range(4), corr.columns)
for i in range(4):
    for j in range(4):
        plt.text(j, i, f'{corr.iloc[i, j]:.2f}', ha='center', va='center')
plt.title('Correlation heatmap (Iris)'); plt.tight_layout(); plt.show()

pairs = corr.abs().where(~np.eye(4, dtype=bool)).stack()
a, b = pairs.idxmax()
print(f'\nStrongest pair: {a} vs {b}  (r = {corr.loc[a, b]:.3f})')
