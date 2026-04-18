import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==============================
# 1. LOAD DATASET
# ==============================
df = pd.read_csv(r"C:\Users\harsh\OneDrive\Desktop\cs2 python\loan_approval_dataset.csv")

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

print("First 5 rows:")
print(df.head())

print("\nDataset Shape:", df.shape)

# ==============================
# 2. DATA CLEANING
# ==============================
# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Encode categorical columns
le = LabelEncoder()

for col in ['education', 'self_employed', 'loan_status']:
    df[col] = le.fit_transform(df[col].astype(str).str.strip())

# ==============================
# 3. DEFINE FEATURES AND TARGET
# ==============================
X = df.drop(['loan_id', 'loan_status'], axis=1)
y = df['loan_status']

# ==============================
# 4. FEATURE SCALING
# ==============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==============================
# 5. TRAIN-TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ==============================
# 6. TRAIN LOGISTIC REGRESSION MODEL
# ==============================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ==============================
# 7. PREDICTIONS
# ==============================
y_pred = model.predict(X_test)

# ==============================
# 8. MODEL EVALUATION
# ==============================
print("\n===== MODEL PERFORMANCE =====")
print("Accuracy Score:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ==============================
# 9. CONFUSION MATRIX
# ==============================
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
