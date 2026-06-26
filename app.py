import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #e0f2fe 0%, #f8fafc 45%, #dbeafe 100%);
    color: #0f172a;
}
.main-title {
    font-size: 52px;
    font-weight: 800;
    color: #0f172a;
}
.subtitle {
    font-size: 19px;
    color: #334155;
    margin-bottom: 18px;
}
.card {
    background-color: white;
    padding: 28px;
    border-radius: 22px;
    box-shadow: 0px 8px 25px rgba(15, 23, 42, 0.12);
    margin-bottom: 22px;
}
.result-card {
    background-color: white;
    padding: 28px;
    border-radius: 22px;
    box-shadow: 0px 8px 25px rgba(15, 23, 42, 0.12);
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

X = df[["Glucose", "BMI", "Age"]]
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

left, center, right = st.columns([1.2, 2.2, 1.2])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 About the Project")
    st.write(
        "This app uses a Logistic Regression model trained on a diabetes dataset "
        "to estimate diabetes likelihood from glucose, BMI, and age."
    )
    st.write("**Tools used:**")
    st.write("Python · pandas · scikit-learn · Streamlit")
    st.metric("Model Accuracy", f"{round(accuracy * 100, 2)}%")
    st.markdown("</div>", unsafe_allow_html=True)

with center:
    st.markdown('<div class="main-title">🩺 Diabetes Risk Predictor</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">A machine learning web app that estimates diabetes likelihood using glucose, BMI, and age.</div>',
        unsafe_allow_html=True
    )

    st.info("Educational project only — not a medical diagnosis.")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Enter Your Health Information")

    glucose = st.slider("Glucose Level", 0, 250, 120)
    bmi = st.slider("BMI", 0.0, 80.0, 30.0)
    age = st.slider("Age", 0, 120, 30)

    predict_button = st.button("Predict Risk")
    st.markdown("</div>", unsafe_allow_html=True)

    if predict_button:
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
        st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🧠 How It Works")
    st.write("The model looks at three inputs:")
    st.write("🧪 **Glucose**")
    st.write("⚖️ **BMI**")
    st.write("🎂 **Age**")
    st.write(
        "It then compares these values to patterns learned from the dataset "
        "and estimates diabetes likelihood."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("⚠️ Reminder")
    st.write(
        "This app is for learning and portfolio purposes only. "
        "It should not be used for medical decisions."
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    '<p class="small-text">Built by Diya Senthil using Python, pandas, scikit-learn, and Streamlit.</p>',
    unsafe_allow_html=True
)
