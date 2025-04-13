import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Dummy user credentials
USER_CREDENTIALS = {"saloni": "password123", "admin": "admin@123"}

# Initialize session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "theme" not in st.session_state:
    st.session_state.theme = "Dark"

# 🎨 **Theme Toggle**
def switch_theme():
    st.session_state.theme = "Light" if st.session_state.theme == "Dark" else "Dark"
    st.rerun()

# 🔐 **Login Page**
def login():
    st.title("🔐 Login to Dashboard")
    username = st.text_input("👤 Username", placeholder="Enter your username")
    password = st.text_input("🔑 Password", type="password", placeholder="Enter your password")
    
    if st.button("Login", use_container_width=True):
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state.logged_in = True
            st.success("✅ Login Successful!")
            st.rerun()
        else:
            st.error("❌ Invalid Username or Password")

# 🏆 **LSTM Model Training**
def train_lstm(X_train, X_test, y_train, y_test):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
    X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

    model = Sequential([
        LSTM(50, activation="relu", return_sequences=True, input_shape=(X_train_scaled.shape[1], 1)),
        LSTM(50, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train_scaled, y_train, epochs=20, batch_size=16, verbose=0)

    y_pred = model.predict(X_test_scaled)
    mse = np.mean((y_pred.flatten() - y_test) ** 2)
    return y_pred.flatten(), mse

# 📊 **Prediction Function**
def predict_with_model(model_name, X_train, X_test, y_train, y_test):
    if model_name == "Linear Regression":
        model = LinearRegression()
    elif model_name == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_name == "XGBoost":
        model = XGBRegressor(n_estimators=100, random_state=42)
    elif model_name == "LSTM (Deep Learning)":
        return train_lstm(X_train, X_test, y_train, y_test)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = np.mean((y_pred - y_test) ** 2)
    return y_pred, mse

# 🚀 **Dashboard**
def dashboard():
    st.sidebar.header("⚙️ Settings")
    st.sidebar.button("🌗 Toggle Theme", on_click=switch_theme)

    st.sidebar.header("📂 Upload Your CSV")
    dataset = st.sidebar.selectbox("Choose Dataset", ["Petroleum", "Gas"])
    model_name = st.sidebar.selectbox("Choose Model", ["Linear Regression", "Random Forest", "XGBoost", "LSTM (Deep Learning)"])
    uploaded_file = st.sidebar.file_uploader(f"Upload {dataset} CSV", type=["csv"])

    if st.sidebar.button("🚪 Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.rerun()

    # 📥 Load Data
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")
    else:
        st.warning("⚠️ No file uploaded. Using sample data.")
        df = pd.read_csv("clean_petroleumflowdata.csv") if dataset == "Petroleum" else pd.read_csv("clean_gasProduction.csv")

    # 📈 **Preprocess Data**
    if dataset == "Petroleum":
        st.subheader("🛢 Petroleum Data")
        df['Date'] = pd.to_datetime(df['Date'])
        df['Date_Ordinal'] = df['Date'].map(pd.Timestamp.toordinal)
        X = df[['Date_Ordinal']]
        y = df['Total flow']

        fig = px.line(df, x='Date', y='Total flow', title="Petroleum Flow Trend", template="plotly_dark" if st.session_state.theme == "Dark" else "plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.subheader("🔥 Gas Production Data")
        df['Time'] = pd.to_datetime(df['Time'])
        df['Time_Ordinal'] = df['Time'].map(pd.Timestamp.toordinal)
        X = df[['Time_Ordinal']]
        y = df['Production']

        fig = px.line(df, x='Time', y='Production', title="Gas Production Trend", template="plotly_dark" if st.session_state.theme == "Dark" else "plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    # 📊 **Train Model & Predict**
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred, mse = predict_with_model(model_name, X_train, X_test, y_train, y_test)

    st.success(f"✅ {model_name} Model Trained! MSE: {mse:.2f}")

    # 📈 **Prediction vs Actual**
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_test.values, mode="lines", name="Actual", line=dict(color="blue")))
    fig.add_trace(go.Scatter(y=y_pred, mode="lines", name="Predicted", line=dict(color="red")))
    fig.update_layout(title="Actual vs Predicted", template="plotly_dark" if st.session_state.theme == "Dark" else "plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # 📥 **Download Predictions**
    results_df = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
    csv = results_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Predictions", data=csv, file_name="predictions.csv", mime="text/csv")

# 🚀 **Run Login or Dashboard**
if not st.session_state.logged_in:
    login()
else:
    dashboard()
