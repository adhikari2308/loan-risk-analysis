# 📊 Loan Approval Prediction using Data Science & Machine Learning

## 🚀 Project Overview
This project focuses on analyzing a Loan Approval dataset to uncover key insights and build a predictive model for loan approval. The goal is to understand the factors influencing loan decisions and apply data-driven techniques for better financial decision-making.

---

## 📁 Dataset
- Dataset: Loan Approval Dataset  
- Contains information about applicants such as:
  - Income
  - Loan Amount
  - CIBIL Score
  - Education
  - Employment Status
  - Loan Status (Approved/Rejected)

---

## 🔍 Project Workflow

### 1. Data Loading
- Loaded dataset using Pandas  
- Checked dataset structure, shape, and missing values  

### 2. Data Cleaning & Preprocessing
- Removed duplicate records  
- Handled missing values  
- Cleaned column names  
- Encoded categorical variables using Label Encoding  
- Feature scaling using StandardScaler  

---

### 3. Exploratory Data Analysis (EDA)
- Visualized loan approval distribution  
- Income distribution analysis  
- Loan amount trends  
- CIBIL score analysis (Boxplot)  
- Correlation heatmap  
- Income vs Loan amount (Scatter plot)

---

### 4. Statistical Analysis
- Applied **Z-Test** to validate income-based insights  
- Tested hypothesis using sample mean vs assumed population mean  

---

### 5. Machine Learning Model
- Model Used: **Logistic Regression**  
- Train-Test Split: 80% Training / 20% Testing  
- Feature Scaling applied  

---

### 6. Model Evaluation
- Accuracy Score  
- Classification Report  
- Confusion Matrix  

---

## 🛠️ Technologies Used
- Python 🐍  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-learn  
- SciPy  

---

## 📈 Results
- Successfully analyzed key factors affecting loan approval  
- Built a predictive model to classify loan status  
- Gained insights into income, loan amount, and credit score relationships  

---

## 💡 Key Learnings
- Data preprocessing is crucial for accurate modeling  
- Visualization helps uncover hidden patterns  
- Statistical testing validates assumptions  
- Machine learning improves decision-making  

---

## 📂 Project Structure
├── load.py # Data loading & basic inspection


├── eda.py # Exploratory Data Analysis & visualization


├── ztest.py # Statistical testing (Z-Test)


├── ml.py # Machine Learning model


├── loan_approval_dataset.csv


└── README.md
