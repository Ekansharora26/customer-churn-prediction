# ================== IMPORTS ==================
import pandas as pd
import streamlit as st
from xgboost import XGBClassifier

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# ================== DARK + ANIMATION CSS ==================
st.markdown("""
<style>
/* Background */
.stApp {
    background-color: #0f172a;
    color: #e5e7eb;
}

/* Headings */
h1, h2, h3 {
    color: #f8fafc;
}

/* Fade-in animation */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(15px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Pulse animation */
@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(59,130,246,0.6); }
    70% { box-shadow: 0 0 0 15px rgba(59,130,246,0); }
    100% { box-shadow: 0 0 0 0 rgba(59,130,246,0); }
}

/* Result cards */
.card {
    padding: 22px;
    border-radius: 16px;
    background-color: #020617;
    box-shadow: 0 8px 25px rgba(0,0,0,0.7);
    animation: fadeIn 0.8s ease-in-out;
    margin-top: 20px;
}

/* Risk styles */
.success {
    border-left: 6px solid #22c55e;
}
.danger {
    border-left: 6px solid #ef4444;
    animation: pulse 2s infinite;
}

/* Inputs */
div[data-baseweb="select"] > div,
input {
    background-color: #020617 !important;
    color: #f8fafc !important;
    border-radius: 10px;
}

/* Button */
button {
    background: linear-gradient(135deg, #2563eb, #1e40af) !important;
    color: white !important;
    border-radius: 14px !important;
    height: 3em;
    font-weight: 700;
    transition: transform 0.2s ease;
}
button:hover {
    transform: scale(1.03);
}
</style>
""", unsafe_allow_html=True)

# ================== LOAD & TRAIN MODEL ==================
@st.cache_data
def load_model():
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

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=42
    )

    model.fit(X, y)
    return model, X.columns


model, feature_columns = load_model()

# ================== HEADER ==================
st.markdown("<h1>📊 Customer Churn Prediction System</h1>", unsafe_allow_html=True)
st.write("Dark-themed ML app with smooth animations and real-time churn prediction.")
st.divider()

# ================== INPUT FORM ==================
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Customer Profile")
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 5)
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])

with col2:
    st.subheader("📡 Services & Billing")
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    TechSupport = st.selectbox("Tech Support", ["Yes", "No"])
    StreamingTV = st.selectbox("Streaming TV", ["Yes", "No"])
    PaymentMethod = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )
    MonthlyCharges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
    TotalCharges = st.number_input("Total Charges", 0.0, 10000.0, 500.0)

# ================== PREDICTION ==================
if st.button("🔮 Predict Churn"):
    customer = pd.DataFrame([{
        'SeniorCitizen': SeniorCitizen,
        'Partner': Partner,
        'Dependents': Dependents,
        'tenure': tenure,
        'PhoneService': PhoneService,
        'MultipleLines': "No",
        'InternetService': InternetService,
        'OnlineSecurity': "No",
        'OnlineBackup': "No",
        'DeviceProtection': "No",
        'TechSupport': TechSupport,
        'StreamingTV': StreamingTV,
        'StreamingMovies': "No",
        'Contract': Contract,
        'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges
    }])

    customer = pd.get_dummies(customer)
    customer = customer.reindex(columns=feature_columns, fill_value=0)

    prediction = model.predict(customer)[0]
    probability = float(model.predict_proba(customer)[0][1])

    st.divider()

    if prediction == 1:
        st.markdown(
            f"<div class='card danger'><h3>⚠️ High Churn Risk</h3>"
            f"<p>Churn Probability: <b>{probability*100:.2f}%</b></p></div>",
            unsafe_allow_html=True
        )
        st.progress(probability)
    else:
        st.markdown(
            f"<div class='card success'><h3>✅ Low Churn Risk</h3>"
            f"<p>Retention Probability: <b>{(1-probability)*100:.2f}%</b></p></div>",
            unsafe_allow_html=True
        )
        st.progress(1 - probability)
