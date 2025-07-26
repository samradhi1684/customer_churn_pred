# 📉 Customer Churn Prediction ML App

A full-stack Machine Learning project to **predict telecom customer churn**.  
This project includes end-to-end steps from **EDA, feature engineering, model building** to deploying a working **interactive web app** using `Streamlit`.

---

## 📌 Problem Statement

Customer churn is a costly issue for telecom companies. The goal is to develop a system that identifies which customers are at risk of leaving, based on their service usage, payment patterns, and contract details.

---

## 🖼️ Preview


## 🧠 Key Features

✅ Upload your own customer data (.csv)  
✅ Auto preprocessing and alignment with trained model  
✅ Predict churn label and churn probability  
✅ Download predictions as a .csv  
✅ Beautiful, responsive UI powered by Streamlit  

---

## 📊 Dataset Overview

- 📦 7,032 customers  
- 🧮 32 preprocessed features including:
  - Demographics (SeniorCitizen, gender, etc.)
  - Services (PhoneService, Internet, TechSupport, etc.)
  - Charges (Monthly, Total, charge ratio)
  - Contract type, payment method, etc.

---

## 🧪 How to Use This App

> 🔗 The app runs locally and accepts customer `.csv` files for prediction.

### 📂 Setup Instructions

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/customer-churn-predictor.git
cd customer-churn-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the Streamlit app
streamlit run app.py
```

## 🧾 Testing the App
You can immediately test the app using the sample file test_customer.csv included in the repo.

## What I learnt
📌 Exploratory Data Analysis (EDA)
  - Customer Churn distribution
  - Key patterns with tenure, contract type, monthly charges, etc.

📌 Feature Engineering
  - Created charge_ratio = TotalCharges / tenure
  - One-hot encoding of all categorical variables
  - Handled missing and inconsistent data

📌 Modeling & Evaluation
  - Tried 3 models: Logistic Regression, XGBoost, Random Forest
  - Selected Logistic Regression with class_weight=balanced for best recall
  - Evaluated using: Accuracy, F1-score, Confusion Matrix, Precision/Recall

📌 App Development
  - Built the UI using Streamlit
  - Accepts dynamic CSV uploads
  - Aligns uploaded features to model’s input
  - Returns predictions and churn probability instantly

## 🧰 Tech Stack 
Python | Pandas | Numpy | Scikit-learn | Streamlit | Matplotlib | Seaborn | Jupyter Notebook

## 🚀 Future Enhancements
  - Add SHAP/Explainable AI integration
  - Upload .xlsx files support
  - Deploy on Streamlit Cloud or HuggingFace
  - Model switcher: try different algorithms live


## ✅ Conclusion 
This project is a complete ML pipeline that not only delivers accurate churn predictions but also demonstrates critical real-world ML development skills:
  - Data Cleaning & Preprocessing
  - Feature Engineering
  - Model Building & Evaluation
  - Frontend Deployment with Streamlit
  - Real-world file input/output & automation


