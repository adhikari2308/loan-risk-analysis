import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv(r"C:\Users\harsh\OneDrive\Desktop\cs2 python\loan_approval_dataset.csv")


df.columns = df.columns.str.strip()

print("First 5 rows:")
print(df.head())

print("\nDataset Shape:", df.shape)


df.drop_duplicates(inplace=True)


le = LabelEncoder()

for col in ['education', 'self_employed', 'loan_status']:
    df[col] = le.fit_transform(df[col].astype(str).str.strip())


X = df.drop(['loan_id', 'loan_status'], axis=1)
y = df['loan_status']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)


model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print("\n===== MODEL PERFORMANCE =====")
print("Accuracy Score:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
