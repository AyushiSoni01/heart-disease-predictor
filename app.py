import streamlit as st
import pandas as pd
import joblib

# Load model, scaler and columns
model = joblib.load("LR_heart.pkl")
scaler = joblib.load("LR_scaler.pkl")
columns = joblib.load("LR_columns.pkl")

# -----------------------------
# Streamlit UI
# -----------------------------

st.title("Heart Disease Prediction by Ayushi 🧸")
st.markdown("Provide the following patient details")

# User Inputs
age = st.slider("Age", 18, 100, 40)

sex = st.selectbox(
    "Sex",
    ["M", "F"]
)

chest_pain = st.selectbox(
    "Chest Pain Type",
    ["ASY", "ATA", "NAP", "TA"]
)

resting_bp = st.number_input(
    "Resting Blood Pressure (mm Hg)",
    80, 250, 120
)

cholesterol = st.number_input(
    "Cholesterol (mg/dL)",
    0, 700, 200
)

fasting_bs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dL",
    [0, 1]
)

resting_ecg = st.selectbox(
    "Resting ECG",
    ["Normal", "ST", "LVH"]
)

max_hr = st.slider(
    "Maximum Heart Rate",
    60, 220, 150
)

exercise_angina = st.selectbox(
    "Exercise Induced Angina",
    ["Y", "N"]
)

oldpeak = st.slider(
    "Oldpeak (ST Depression)",
    0.0, 6.0, 1.0
)

st_slope = st.selectbox(
    "ST Slope",
    ["Up", "Flat", "Down"]
)

# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict"):

    # Risk score feature
    risk_score = (
        age * 0.02 +
        resting_bp * 0.01 +
        cholesterol * 0.005 +
        oldpeak * 2
    )

    # Create full input dictionary
    raw_input = {

        # Numerical Features
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'risk_score': risk_score,

        # Gender
        'Gender_F': 1 if sex == "F" else 0,
        'Gender_M': 1 if sex == "M" else 0,

        # Chest Pain
        'ChestPainType_ASY': 1 if chest_pain == "ASY" else 0,
        'ChestPainType_ATA': 1 if chest_pain == "ATA" else 0,
        'ChestPainType_NAP': 1 if chest_pain == "NAP" else 0,
        'ChestPainType_TA': 1 if chest_pain == "TA" else 0,

        # ECG
        'RestingECG_LVH': 1 if resting_ecg == "LVH" else 0,
        'RestingECG_Normal': 1 if resting_ecg == "Normal" else 0,
        'RestingECG_ST': 1 if resting_ecg == "ST" else 0,

        # Exercise Angina
        'ExerciseAngina_N': 1 if exercise_angina == "N" else 0,
        'ExerciseAngina_Y': 1 if exercise_angina == "Y" else 0,

        # ST Slope
        'ST_Slope_Down': 1 if st_slope == "Down" else 0,
        'ST_Slope_Flat': 1 if st_slope == "Flat" else 0,
        'ST_Slope_Up': 1 if st_slope == "Up" else 0
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([raw_input])

    # Add missing columns
    for col in columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Remove target column if present
    if 'HeartDisease' in input_df.columns:
        input_df = input_df.drop('HeartDisease', axis=1)

    # Correct column order
    model_columns = [col for col in columns if col != 'HeartDisease']
    input_df = input_df[model_columns]

    # Scale input
    scaled_input = scaler.transform(input_df)

    # Prediction
    prediction = model.predict(scaled_input)[0]

    # Probability
    probability = model.predict_proba(scaled_input)[0][1]

    # -----------------------------
    # Output
    # -----------------------------

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Disease ({probability*100:.2f}%)")
    else:
        st.success(f"✅ Low Risk of Heart Disease ({(1-probability)*100:.2f}%)")

    # Optional debugging
    with st.expander("Show Processed Input Data"):
        st.write(input_df)