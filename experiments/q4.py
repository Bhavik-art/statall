# ==============================================================================
# Q4. Euclidean distance before and after standardization
# ==============================================================================
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import euclidean

data = pd.DataFrame({'Age': [25, 32, 47, 51, 38], 'Salary': [30000, 45000, 90000, 120000, 60000]},
                    index=['A', 'B', 'C', 'D', 'E'])
print('Original data:'); print(data)

scaled = pd.DataFrame(StandardScaler().fit_transform(data), columns=data.columns, index=data.index)
print('\nStandardized data:'); print(scaled.round(3))

d_before = euclidean(data.loc['A'], data.loc['B'])
d_after = euclidean(scaled.loc['A'], scaled.loc['B'])
print(f'\nDistance A-B before scaling: {d_before:.3f}')
print(f'Distance A-B after scaling : {d_after:.3f}')
