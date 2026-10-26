import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import euclidean

# Create a dataset with different scales
data = {
    'Age': [20, 25, 30, 35],
    'Salary': [20000, 30000, 40000, 50000]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

# Calculate Euclidean distance before standardization
distance_before = euclidean(df.iloc[0], df.iloc[1])
print("\nDistance Before Standardization:", round(distance_before, 2))

# Apply standardization
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

scaled_df = pd.DataFrame(scaled_data, columns=df.columns)

print("\nStandardized Dataset:")
print(scaled_df.round(2))

# Calculate Euclidean distance after standardization
distance_after = euclidean(scaled_df.iloc[0], scaled_df.iloc[1])
print("\nDistance After Standardization:", round(distance_after, 2))