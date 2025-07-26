import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("Life Expectancy Data.csv")
    return df

df = load_data()
st.title("🌍 Life Expectancy Predictor (with Pretrained ANN)")
st.write("This app uses a pre-trained Keras regression model to predict life expectancy based on WHO health indicators.")

# Preprocess
df.dropna(subset=["Life expectancy "], inplace=True)
X = df.drop(["Country", "Year", "Life expectancy "], axis=1)
y = df["Life expectancy "]

# Encode 'Status'
le = LabelEncoder()
X["Status"] = le.fit_transform(X["Status"])

# Imputer & Scaler (fitted on entire dataset for consistent transform)
imp = SimpleImputer(strategy="mean")
X_imp = imp.fit_transform(X)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_imp)

# Load Pretrained Model
model = load_model("regression_model.keras")

# User Input Form
st.header("🔢 Enter values to predict life expectancy")
user_input = []

for col in X.columns:
    col_min = float(df[col].min())
    col_max = float(df[col].max())
    col_mean = float(df[col].mean())
    
    if col == "Status":
        status_val = st.selectbox("Status", ["Developed", "Developing"])
        user_input.append(0 if status_val == "Developed" else 1)
    else:
        val = st.slider(col, col_min, col_max, float(col_mean))
        user_input.append(val)

# Make Prediction
input_array = scaler.transform(imp.transform([user_input]))
predicted_life_expectancy = model.predict(input_array)[0][0]

st.success(f"🎯 Predicted Life Expectancy: **{predicted_life_expectancy:.2f} years**")

# Show Dataset Summary
if st.checkbox("📊 Show Sample of Dataset"):
    st.dataframe(df.head())
