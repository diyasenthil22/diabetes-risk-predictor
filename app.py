
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

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

# Web app
st.title("Diabetes Risk Predictor")

st.write("Enter health information to estimate diabetes likelihood.")

glucose = st.number_input("Glucose", min_value=0, max_value=250, value=120)
bmi = st.number_input("BMI", min_value=0.0, max_value=80.0, value=30.0)
age = st.number_input("Age", min_value=0, max_value=120, value=30)

if st.button("Predict"):
    user_data = pd.DataFrame({
        "Glucose": [glucose],
        "BMI": [bmi],
        "Age": [age]
    })

    prediction = model.predict(user_data)[0]
    probability = model.predict_proba(user_data)[0][1]

    if prediction == 1:
        st.warning("Higher likelihood of diabetes")
    else:
        st.success("Lower likelihood of diabetes")

    st.write("Probability:", round(probability * 100, 2), "%")
