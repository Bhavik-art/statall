import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

print("Missing Values Before Preprocessing:")
print(df.isnull().sum())

# Deletion method
df.dropna(subset=["Embarked"], inplace=True)

# Imputation using mean
df["Age"] = df["Age"].fillna(df["Age"].mean())

# Imputation using median
df["Fare"] = df["Fare"].fillna(df["Fare"].median())

# Imputation using mode
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

print("\nMissing Values After Preprocessing:")
print(df.isnull().sum())

print("\nFirst Five Records:")
print(df.head())
