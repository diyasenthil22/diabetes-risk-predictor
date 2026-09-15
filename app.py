import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# --------------------------------------------------
# PAGE SETUP
# --------------------------------------------------

st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="centered"
)

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
df = pd.read_csv(url)

features = ["Glucose", "BMI", "Age", "Pregnancies"]

X = df[features]
y = df["Outcome"]

# --------------------------------------------------
# TRAIN / TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# --------------------------------------------------
# EVALUATE MODEL
# --------------------------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)

# Average values for comparison
averages = df.groupby("Outcome")[features].mean()

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🩺 Diabetes Risk Predictor")

st.info(
    "Educational machine learning project only. "
    "This application does not provide a medical diagnosis."
)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("📊 Project Info")

    st.write("**Model:** Logistic Regression")

    st.write(f"**Accuracy:** {round(accuracy * 100, 2)}%")
    st.write(f"**Precision:** {round(precision * 100, 2)}%")
    st.write(f"**Recall:** {round(recall * 100, 2)}%")
    st.write(f"**F1 Score:** {round(f1 * 100, 2)}%")

    st.write("**Features Used:**")
    st.write("• Glucose")
    st.write("• BMI")
    st.write("• Age")
    st.write("• Pregnancies")

    st.divider()

    st.header("🧠 What I Built")

    st.write(
        "I explored a diabetes dataset, analyzed health variables, "
        "trained a machine learning model, evaluated its performance, "
        "and built an interactive web application using Streamlit."
    )

    st.divider()

    st.header("⚠️ Disclaimer")

    st.write(
        "This application is for educational purposes only "
        "and should not be used for medical decisions."
    )

# --------------------------------------------------
# USER INPUT
# --------------------------------------------------

st.header("Enter Health Information")

glucose = st.slider(
    "Glucose Level",
    0,
    250,
    120
)

bmi = st.slider(
    "BMI",
    0.0,
    80.0,
    30.0
)

age = st.slider(
    "Age",
    0,
    120,
    30
)

pregnancies = st.slider(
    "Pregnancies",
    0,
    20,
    1
)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if st.button("🩺 Check Diabetes Risk"):

    user_data = pd.DataFrame({
        "Glucose": [glucose],
        "BMI": [bmi],
        "Age": [age],
        "Pregnancies": [pregnancies]
    })

    probability = model.predict_proba(user_data)[0][1]

    risk_percent = round(
        probability * 100,
        2
    )

    st.header("Prediction Result")

    # These are model probability categories,
    # not medical risk categories.

    if risk_percent < 35:

        st.success(
            "🟢 Lower Model-Predicted Probability"
        )

    elif risk_percent < 65:

        st.warning(
            "🟡 Moderate Model-Predicted Probability"
        )

    else:

        st.error(
            "🔴 Higher Model-Predicted Probability"
        )

    st.metric(
        "Model-Predicted Probability",
        f"{risk_percent}%"
    )

    st.progress(probability)

    # --------------------------------------------------
    # DATASET COMPARISON
    # --------------------------------------------------

    st.subheader(
        "📊 How Your Values Compare With the Dataset"
    )

    if glucose > averages.loc[1, "Glucose"]:

        st.write(
            "🧪 **Glucose:** Your glucose level is higher "
            "than the average value observed among diabetic "
            "patients in the dataset."
        )

    elif glucose > averages.loc[0, "Glucose"]:

        st.write(
            "🧪 **Glucose:** Your glucose level is higher "
            "than the average value observed among "
            "non-diabetic patients in the dataset."
        )

    else:

        st.write(
            "🧪 **Glucose:** Your glucose level is closer "
            "to the average value observed among "
            "non-diabetic patients in the dataset."
        )

    if bmi > averages.loc[1, "BMI"]:

        st.write(
            "⚖️ **BMI:** Your BMI is higher than the "
            "average value observed among diabetic "
            "patients in the dataset."
        )

    elif bmi > averages.loc[0, "BMI"]:

        st.write(
            "⚖️ **BMI:** Your BMI is higher than the "
            "average value observed among non-diabetic "
            "patients in the dataset."
        )

    else:

        st.write(
            "⚖️ **BMI:** Your BMI is closer to the "
            "average value observed among non-diabetic "
            "patients in the dataset."
        )

    if age > averages.loc[1, "Age"]:

        st.write(
            "🎂 **Age:** Your age is higher than the "
            "average age observed among diabetic "
            "patients in the dataset."
        )

    elif age > averages.loc[0, "Age"]:

        st.write(
            "🎂 **Age:** Your age is higher than the "
            "average age observed among non-diabetic "
            "patients in the dataset."
        )

    else:

        st.write(
            "🎂 **Age:** Your age is closer to the "
            "average age observed among non-diabetic "
            "patients in the dataset."
        )

    if pregnancies > averages.loc[1, "Pregnancies"]:

        st.write(
            "🤰 **Pregnancies:** Your pregnancy count is "
            "higher than the average value observed among "
            "diabetic patients in the dataset."
        )

    elif pregnancies > averages.loc[0, "Pregnancies"]:

        st.write(
            "🤰 **Pregnancies:** Your pregnancy count is "
            "higher than the average value observed among "
            "non-diabetic patients in the dataset."
        )

    else:

        st.write(
            "🤰 **Pregnancies:** Your pregnancy count is "
            "closer to the average value observed among "
            "non-diabetic patients in the dataset."
        )

    # --------------------------------------------------
    # COMPARISON TABLE
    # --------------------------------------------------

    st.subheader("📊 Compare Your Health Measurements")

    comparison = pd.DataFrame({

        "Your Value": [
            glucose,
            bmi,
            age,
            pregnancies
        ],

        "Avg. Non-Diabetic": [
            round(averages.loc[0, "Glucose"], 2),
            round(averages.loc[0, "BMI"], 2),
            round(averages.loc[0, "Age"], 2),
            round(averages.loc[0, "Pregnancies"], 2)
        ],

        "Avg. Diabetic": [
            round(averages.loc[1, "Glucose"], 2),
            round(averages.loc[1, "BMI"], 2),
            round(averages.loc[1, "Age"], 2),
            round(averages.loc[1, "Pregnancies"], 2)
        ]

    }, index=features)

    st.dataframe(comparison)

# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

st.divider()

st.header("🧠 Model Performance")

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Accuracy",
        f"{accuracy * 100:.1f}%"
    )

    st.metric(
        "Precision",
        f"{precision * 100:.1f}%"
    )

with col2:

    st.metric(
        "Recall",
        f"{recall * 100:.1f}%"
    )

    st.metric(
        "F1 Score",
        f"{f1 * 100:.1f}%"
    )

st.write(
    "**Accuracy** measures how often the model was correct overall. "
    "**Precision** measures how often a predicted diabetes case was "
    "actually diabetic. **Recall** measures how many actual diabetes "
    "cases the model successfully identified. **F1 score** balances "
    "precision and recall."
)

# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------

st.subheader("Confusion Matrix")

confusion_df = pd.DataFrame(
    cm,
    index=[
        "Actually Non-Diabetic",
        "Actually Diabetic"
    ],
    columns=[
        "Predicted Non-Diabetic",
        "Predicted Diabetic"
    ]
)

st.dataframe(confusion_df)

st.write(
    "The confusion matrix shows exactly where the model "
    "made correct predictions and where it made mistakes."
)

# --------------------------------------------------
# FEATURE COEFFICIENTS
# --------------------------------------------------

st.divider()

st.header("📊 Model Coefficients")

coefficients = pd.DataFrame({

    "Feature": features,

    "Coefficient": model.coef_[0]

})

coefficients["Absolute Coefficient"] = abs(
    coefficients["Coefficient"]
)

coefficients = coefficients.sort_values(
    by="Absolute Coefficient",
    ascending=True
)

st.bar_chart(
    coefficients.set_index("Feature")[
        "Absolute Coefficient"
    ]
)

st.caption(
    "Larger coefficients indicate a stronger effect on the model's predictions, "
    "but values are affected by each feature's measurement scale."
)

# --------------------------------------------------
# DATASET INFORMATION
# --------------------------------------------------

st.divider()

st.header("About the Dataset")

st.write(
    f"This project uses a diabetes dataset containing "
    f"{len(df)} patient records. The model currently uses "
    f"glucose level, BMI, age, and pregnancy count to "
    f"estimate the probability associated with the diabetes "
    f"outcome in the dataset."
)

# --------------------------------------------------
# KEY FINDINGS
# --------------------------------------------------

st.header("📈 Key Findings")

st.markdown(
    f"""
- The Logistic Regression model achieved **{accuracy * 100:.1f}% accuracy** on the test data.
- The model achieved **{recall * 100:.1f}% recall**, meaning it identified this percentage of the actual diabetes cases in the test set.
- Glucose and BMI differed between the diabetic and non-diabetic groups in the dataset.
- Model performance cannot be understood from accuracy alone, so precision, recall, and F1 score were also evaluated.
"""
)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.caption(
    "Built by Diya Senthil using Python, pandas, "
    "scikit-learn, and Streamlit."
)
