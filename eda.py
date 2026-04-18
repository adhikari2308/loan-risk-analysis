import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv(r"C:\Users\harsh\OneDrive\Desktop\cs2 python\loan_approval_dataset.csv")

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# ==============================
# Summary Statistics
# ==============================
print("First 5 rows:")
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nSummary Statistics:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nCorrelation Matrix:")
print(df.corr(numeric_only=True))

# ==============================
# Graph 1: Loan Approval Status
# ==============================
plt.figure(figsize=(8, 5))
sns.countplot(x='loan_status', data=df)
plt.title('Loan Approval Status')
plt.xlabel('Loan Status')
plt.ylabel('Count')
plt.show()

# ==============================
# Graph 2: Income Distribution
# ==============================
plt.figure(figsize=(10, 5))
sns.histplot(df['income_annum'], bins=30, kde=True)
plt.title('Income Distribution')
plt.xlabel('Annual Income')
plt.ylabel('Frequency')
plt.show()

# ==============================
# Graph 3: Loan Amount Distribution
# ==============================
plt.figure(figsize=(10, 5))
sns.histplot(df['loan_amount'], bins=30, kde=True)
plt.title('Loan Amount Distribution')
plt.xlabel('Loan Amount')
plt.ylabel('Frequency')
plt.show()

# ==============================
# Graph 4: CIBIL Score Boxplot
# ==============================
plt.figure(figsize=(10, 5))
sns.boxplot(x=df['cibil_score'])
plt.title('CIBIL Score Boxplot')
plt.xlabel('CIBIL Score')
plt.show()

# ==============================
# Graph 5: Correlation Heatmap
# ==============================
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# ==============================
# Graph 6: Income vs Loan Amount
# ==============================
plt.figure(figsize=(8, 5))
sns.scatterplot(x='income_annum', y='loan_amount', hue='loan_status', data=df)
plt.title('Income vs Loan Amount')
plt.xlabel('Annual Income')
plt.ylabel('Loan Amount')
plt.show()
