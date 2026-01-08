# ================== IMPORTS ==================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier, plot_importance


# ================== LOAD DATA ==================
df = pd.read_csv(
    r"C:\Users\HP\OneDrive\Desktop\customer churn analysis\data\churn_data.csv"
)

df = df.drop(columns=['Unnamed: 0'], errors='ignore')

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

df['Churn'] = df['Churn'].replace({
    'Yes': 1, 'No': 0, True: 1, False: 0
})

df['Churn'] = pd.to_numeric(df['Churn'], errors='coerce')
df = df.dropna()

print("Total rows after cleaning:", df.shape[0])


# ================== FEATURE ENGINEERING ==================
features = [
    'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges'
]

X = df[features]
y = df['Churn']

X = pd.get_dummies(X, drop_first=True)
print("Total features after encoding:", X.shape[1])


# ================== TRAIN TEST SPLIT ==================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)


# ================== XGBOOST MODEL ==================
xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    random_state=42
)

xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)

xgb_accuracy = accuracy_score(y_test, xgb_pred)
print("\nXGBoost Accuracy:", xgb_accuracy)

xgb_cm = confusion_matrix(y_test, xgb_pred)
print("\nXGBoost Confusion Matrix:\n", xgb_cm)

sns.heatmap(xgb_cm, annot=True, fmt='d', cmap='Greens')
plt.title("XGBoost Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# ================== NEW CUSTOMER PREDICTION ==================
new_customer = {
    'SeniorCitizen': 0,
    'Partner': 'No',
    'Dependents': 'No',
    'tenure': 5,
    'PhoneService': 'Yes',
    'MultipleLines': 'No',
    'InternetService': 'Fiber optic',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'No',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'Yes',
    'StreamingMovies': 'Yes',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 85.5,
    'TotalCharges': 420.3
}

new_df = pd.DataFrame([new_customer])
new_df = pd.get_dummies(new_df)
new_df = new_df.reindex(columns=X.columns, fill_value=0)

prediction = xgb_model.predict(new_df)[0]
probability = xgb_model.predict_proba(new_df)[0][1]

print("\nNEW CUSTOMER PREDICTION")
print("Churn Prediction:", "YES (Will Churn)" if prediction == 1 else "NO (Will Stay)")
print("Churn Probability:", round(probability * 100, 2), "%")


# ================== TOP CHURN DRIVERS ==================
plt.figure(figsize=(10, 6))
plot_importance(xgb_model, max_num_features=8)
plt.title("Top Factors Causing Customer Churn")
plt.show()
