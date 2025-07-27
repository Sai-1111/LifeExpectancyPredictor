import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.models import load_model

# Title
st.set_page_config(page_title="Life Expectancy Predictor", layout="centered")
st.title("🌍 Life Expectancy Predictor (with Pretrained ANN)")
st.write("This app predicts life expectancy using a pre-trained Artificial Neural Network based on WHO data.")

# Load data
df = pd.read_csv("Life Expectancy Data.csv")
df.columns = df.columns.str.strip()  # Clean column names

# Imputation
imp = SimpleImputer(strategy="mean")
columns_to_impute = [
    'Life expectancy ', 'Adult Mortality', 'Alcohol', 'Hepatitis B', ' BMI ', 'Polio',
    'Total expenditure', 'Diphtheria ', 'GDP', 'Population', ' thinness  1-19 years',
    ' thinness 5-9 years', 'Income composition of resources', 'Schooling'
]
df[columns_to_impute] = imp.fit_transform(df[columns_to_impute])

# Encode Status
le = LabelEncoder()
df["Status"] = le.fit_transform(df["Status"])

# Features and label
X = df.drop(["Country", "Year", "Life expectancy "], axis=1)
feature_names = X.columns.tolist()

# Handle outliers (same as training)
cols_to_handle_outliers = [
    'Adult Mortality', 'infant deaths', 'Alcohol', 'percentage expenditure',
    'Hepatitis B', 'Measles ', ' BMI ', 'under-five deaths ', 'Polio',
    'Total expenditure', 'Diphtheria ', ' HIV/AIDS', 'GDP', 'Population',
    ' thinness  1-19 years', ' thinness 5-9 years',
    'Income composition of resources', 'Schooling'
]
for col in cols_to_handle_outliers:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    mean_val = df[col].mean()
    df[col] = np.where((df[col] < lower_bound) | (df[col] > upper_bound), mean_val, df[col])

# Scale
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Load model
model = load_model("regression_model.keras")

# Streamlit UI
st.header("🔢 Enter Health & Socioeconomic Indicators")

user_input = []
for col in feature_names:
    col_min = float(df[col].min())
    col_max = float(df[col].max())
    col_mean = float(df[col].mean())

    if col == "Status":
        val = st.selectbox("Status", ["Developed", "Developing"])
        user_input.append(0 if val == "Developed" else 1)
    else:
        val = st.slider(col, col_min, col_max, float(col_mean))
        user_input.append(val)

# Convert to DataFrame
user_df = pd.DataFrame([user_input], columns=feature_names)

# Preprocess user input
user_scaled = scaler.transform(user_df)

# Predict
prediction = model.predict(user_scaled)[0][0]

# Display result
st.success(f"🎯 Predicted Life Expectancy: **{prediction:.2f} years**")

# Optional: Show sample data
if st.checkbox("📊 Show Sample Data"):
    st.dataframe(df.head())
