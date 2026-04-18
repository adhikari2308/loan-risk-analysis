import pandas as pd

# Load dataset

df = pd.read_csv(r"C:\Users\harsh\OneDrive\Desktop\cs2 python\loan_approval_dataset.csv")
df.columns = df.columns.str.strip()

print("First 5 rows:\n", df.head())
print("\nShape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())
