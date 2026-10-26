
# ==============================================================================
# Working
# Q4. Euclidean Distance Before and After Standardization
# ==============================================================================

import numpy as np
from sklearn.preprocessing import StandardScaler

# 1. Create the dataset
names = ["A", "B", "C", "D", "E"]

data = np.array([
    [25,  30000],
    [32,  45000],
    [47,  90000],
    [51, 120000],
    [38,  60000]
], dtype=float)

features = ["Age", "Salary"]

# 2. Display original data
print("=" * 50)
print("ORIGINAL DATA")
print("=" * 50)
print(f"{'Person':<12}{'Age':>10}{'Salary':>15}")

for name, row in zip(names, data):
    print(f"{name:<12}{row[0]:>10.0f}{row[1]:>15.0f}")

# 3. Standardize the data
scaler = StandardScaler()
scaled = scaler.fit_transform(data)

print("\nSTANDARDIZED DATA")
print("=" * 50)
print(f"{'Person':<12}{'Age':>12}{'Salary':>12}")

for name, row in zip(names, scaled):
    print(f"{name:<12}{row[0]:>12.3f}{row[1]:>12.3f}")

# 4. Calculate Euclidean distance between A and B
# Before standardization
distance_before = np.linalg.norm(data[0] - data[1])

# After standardization
distance_after = np.linalg.norm(scaled[0] - scaled[1])

print("\nEUCLIDEAN DISTANCE: A vs B")
print("=" * 50)
print(f"Distance before standardization: {distance_before:.3f}")
print(f"Distance after standardization : {distance_after:.3f}")

# 5. Compare the distances
print("\nCONCLUSION")
if distance_before > distance_after:
    print("The numerical distance decreased after standardization.")
else:
    print("The numerical distance increased after standardization.")

print("Standardization puts Age and Salary on comparable scales.")
