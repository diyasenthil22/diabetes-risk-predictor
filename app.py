import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="GlucoSense",
    page_icon="💙",
    layout="centered"
)

st.markdown("""
<style>
.main {
    background-color: #0f172a;
}
.big-title {
    font-size: 48px;
    font-weight: 800;
    color: #e0f2fe;
}
.subtitle {
    font-size: 18px;
    color: #cbd5e1;
}
.card {
    background-color: #1e293b;
    padding: 25px;
    border-radius: 18px;
    margin-top: 20px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.25);
}
.result-card {
    background-color: #172554;
    padding: 25px;
    border-radius: 18px;
    margin-top: 25px;
}
.small-text {
    color: #94a3b8;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# Load dataset
url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
df = pd.read_csv(url)

# Train model
X = df[["Glucose", "BMI", "Age"]]
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# Header
st.markdown('<div class="big-title">💙 GlucoSense</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">A simple machine learning web app that estimates diabetes likelihood using glucose, BMI, and age.</div>',
    unsafe_allow_html=True
)

st.markdown("")

st.info("Educational project only — not a medical diagnosis.")

st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("Enter Your Health Information")

glucose = st.slider("Glucose Level", 0, 250, 120)
bmi = st.slider("BMI", 0.0, 80.0, 30.0)
age = st.slider("Age", 0, 120, 30)

st.markdown("</div>", unsafe_allow_html=True)

if st.button("✨ Predict Risk"):
    user_data = pd.DataFrame({
        "Glucose": [glucose],
        "BMI": [bmi],
        "Age": [age]
    })

    prediction = model.predict(user_data)[0]
    probability = model.predict_proba(user_data)[0][1]
    risk_percent = round(probability * 100, 2)

    st.markdown('<div class="result-card">', unsafe_allow_html=True)
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

    st.markdown("### Input Summary")
    st.write(f"🧪 **Glucose:** {glucose}")
    st.write(f"⚖️ **BMI:** {bmi}")
    st.write(f"🎂 **Age:** {age}")

    st.caption(f"Model accuracy on test data: {round(accuracy * 100, 2)}%")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    '<p class="small-text">Built by Diya Senthil using Python, pandas, scikit-learn, and Streamlit.</p>',
    unsafe_allow_html=True
)
