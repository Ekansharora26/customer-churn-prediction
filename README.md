📊 Customer Churn Prediction System

An end-to-end Machine Learning project that predicts whether a customer is likely to churn (leave the service) using historical customer data.
The project includes data preprocessing, feature engineering, model training with XGBoost, evaluation, and a modern Streamlit web application.

🚀 Project Highlights

✅ Real-world Telco Customer Churn dataset

✅ Robust data cleaning & preprocessing

✅ Feature engineering with one-hot encoding

✅ XGBoost (industry-grade ML model)

✅ Model evaluation with confusion matrix

✅ Feature importance (top churn drivers)

✅ Interactive Streamlit web app

✅ Dark UI with animations & progress indicators

✅ Real-time churn probability prediction

🧠 Problem Statement

Customer churn directly impacts business revenue.
Predicting churn in advance allows companies to:

Retain high-value customers

Reduce customer acquisition costs

Improve service and satisfaction

This project builds a predictive system that identifies customers at high risk of churn.

📂 Project Structure
customer-churn-prediction/
│
├── data/
│   └── churn_data.csv
│
├── notebooks/
│   └── churn_analysis.py
│
├── app.py
├── README.md

🛠️ Tech Stack

Language: Python

Data Analysis: Pandas, NumPy

Visualization: Matplotlib, Seaborn

Machine Learning: Scikit-learn, XGBoost

Web App: Streamlit

UI Styling: Custom CSS & animations

📊 Dataset

Source: Telco Customer Churn Dataset

Rows: ~5,000 (after cleaning)

Features include:

Customer demographics

Contract type

Internet & service usage

Billing and payment methods

⚙️ Methodology
1️⃣ Data Cleaning

Removed invalid records

Converted numeric fields correctly

Standardized churn labels

2️⃣ Feature Engineering

One-hot encoding for categorical features

Selected business-relevant variables

3️⃣ Model Training

Baseline model: Logistic Regression

Final model: XGBoost Classifier

4️⃣ Model Evaluation

Accuracy ≈ 81%

Confusion matrix analysis

Churn risk interpretation

5️⃣ Explainability

Feature importance to identify key churn drivers

6️⃣ Deployment

Streamlit web app for real-time predictions

Modern UI with animations and progress bars

📈 Model Performance

Model: XGBoost

Accuracy: ~81%

Strengths:

Captures non-linear relationships

Handles mixed feature types efficiently

🔮 Example Prediction
Churn Prediction: YES
Churn Probability: 84.32%

🌐 Streamlit Web App

The app allows users to:

Enter customer details

Click Predict Churn

Instantly see:

Churn / No-Churn decision

Probability score

Visual progress indicator

Run Locally
pip install -r requirements.txt
streamlit run app.py

🧩 Key Learnings

Handling real-world noisy data

Feature engineering for ML models

Model comparison & evaluation

Building ML-powered web applications

End-to-end ML project deployment

🚀 Future Enhancements

Deploy app on Streamlit Cloud

Threshold tuning for higher churn recall

Database integration for prediction storage

Business dashboard integration
