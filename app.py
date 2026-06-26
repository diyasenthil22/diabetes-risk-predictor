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

url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
df = pd.read_csv(url)

features = ["Glucose", "BMI", "Age", "Pregnancies"]
X = df[features]
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))
averages = df.groupby("Outcome")[features].mean()

st.title("🩺 Diabetes Risk Predictor")
st.info("Educational project only, not a medical diagnosis.")
with st.sidebar:
    st.header("📊 Project Info")
    st.write("**Model:** Logistic Regression")
    st.write(f"**Accuracy:** {round(accuracy * 100, 2)}%")
    st.write("**Features Used:**")
    st.write("• Glucose")
    st.write("• BMI")
    st.write("• Age")
    st.write("• Pregnancies")

    st.divider()

    st.header("🧠 What I Built")
    st.write(
        "I explored a diabetes dataset, created visualizations, tested prediction rules, "
        "and built a machine learning web app using Streamlit."
    )

    st.divider()

    st.header("⚠️ Disclaimer")
    st.write("This app is for educational purposes only and is not medical advice.")

st.header("Enter Health Information")

glucose = st.slider("Glucose Level", 0, 250, 120)
bmi = st.slider("BMI", 0.0, 80.0, 30.0)
age = st.slider("Age", 0, 120, 30)
pregnancies = st.slider("Pregnancies", 0, 20, 1)

if st.button("Predict Risk"):
    user_data = pd.DataFrame({
        "Glucose": [glucose],
        "BMI": [bmi],
        "Age": [age],
        "Pregnancies": [pregnancies]
    })

    prediction = model.predict(user_data)[0]
    probability = model.predict_proba(user_data)[0][1]
    risk_percent = round(probability * 100, 2)

    st.header("Prediction Result")

    if risk_percent < 35:
        st.success("🟢 Lower Diabetes Likelihood")
    elif risk_percent < 65:
        st.warning("🟡 Moderate Diabetes Likelihood")
    else:
        st.error("🔴 Higher Diabetes Likelihood")

    st.metric("Estimated Risk", f"{risk_percent}%")
    st.progress(probability)

    st.subheader("How Your Values Compare")

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
    }, index=features)

    st.dataframe(comparison)

st.divider()
st.write("**Model:** Logistic Regression")
st.write(f"**Model Accuracy:** {round(accuracy * 100, 2)}%")
st.caption("Built by Diya Senthil using Python, pandas, scikit-learn, and Streamlit.")
