import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="centered"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #eff6ff 0%, #ffffff 50%, #dbeafe 100%);
}

.main-title {
    font-size: 46px;
    font-weight: 800;
    color: #1e3a8a;
    text-align: center;
}

.subtitle {
    font-size: 18px;
    color: #475569;
    text-align: center;
    margin-bottom: 25px;
}

.section-card {
    background-color: white;
    padding: 24px;
    border-radius: 18px;
    box-shadow: 0px 6px 18px rgba(30, 58, 138, 0.12);
    margin-bottom: 22px;
}

.small-text {
    color: #64748b;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# Load dataset
url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
df = pd.read_csv(url)

# Train model
features = ["Glucose", "BMI", "Age", "Pregnancies"]
X = df[features]
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# Dataset averages
averages = df.groupby("Outcome")[features].mean()

# Header
st.markdown('<div class="main-title">🩺 Diabetes Risk Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Estimate diabetes likelihood using a machine learning model trained on health data.</div>',
    unsafe_allow_html=True
)

st.info("Educational project only — not a medical diagnosis.")

# Inputs
with st.container():
    st.subheader("Enter Health Information")

    glucose = st.slider("Glucose Level", 0, 250, 120)
    bmi = st.slider("BMI", 0.0, 80.0, 30.0)
    age = st.slider("Age", 0, 120, 30)
    pregnancies = st.slider("Pregnancies", 0, 20, 1)

    predict_button = st.button("Predict Risk")

if predict_button:
    user_data = pd.DataFrame({
        "Glucose": [glucose],
        "BMI": [bmi],
        "Age": [age],
        "Pregnancies": [pregnancies]
    })

    prediction = model.predict(user_data)[0]
    probability = model.predict_proba(user_data)[0][1]
    risk_percent = round(probability * 100, 2)

    st.divider()
    st.subheader("Prediction Result")

    if risk_percent < 35:
        st.success("🟢 Lower Diabetes Likelihood")
        risk_level = "Low"
    elif risk_percent < 65:
        st.warning("🟡 Moderate Diabetes Likelihood")
        risk_level = "Moderate"
    else:
        st.error("🔴 Higher Diabetes Likelihood")
        risk_level = "High"

    st.metric("Estimated Risk", f"{risk_percent}%")
    st.progress(probability)
    st.write(f"**Risk Level:** {risk_level}")

    st.divider()
    st.subheader("How Your Values Compare to the Dataset")

    comparison = pd.DataFrame({
        "Your Value": [glucose, bmi, age, pregnancies],
        "Avg. Non-Diabetic": [
            round(averages.loc[0, "Glucose"], 2),
            round(averages.loc[0, "BMI"], 2),
            round(averages.loc[0, "Age"], 2),
            round(averages.loc[0, "Pregnancies"], 2),
        ],
        "Avg. Diabetic": [
            round(averages.loc[1, "Glucose"], 2),
            round(averages.loc[1, "BMI"], 2),
            round(averages.loc[1, "Age"], 2),
            round(averages.loc[1, "Pregnancies"], 2),
        ],
    }, index=["Glucose", "BMI", "Age", "Pregnancies"])

    st.dataframe(comparison)

    st.divider()
    st.subheader("Model Information")
    st.write("**Algorithm:** Logistic Regression")
    st.write(f"**Model Accuracy:** {round(accuracy * 100, 2)}%")
    st.write("**Features Used:** Glucose, BMI, Age, Pregnancies")

st.divider()
st.caption("Built by Diya Senthil using Python, pandas, scikit-learn, and Streamlit.")
