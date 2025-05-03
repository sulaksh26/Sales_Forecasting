import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

st.title("Sales Forecasting using XGBoost")

DEFAULT_CSV_PATH = "train.csv"

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.success("Custom file uploaded successfully.")
else:
    st.info("Using default dataset: train.csv")
    data = pd.read_csv(DEFAULT_CSV_PATH)

st.subheader("Raw Data Preview")
st.dataframe(data.head())

# Parse Order Date
data['Order Date'] = pd.to_datetime(data['Order Date'], format='%d/%m/%Y')

# Plot Sales Trend
st.subheader("Sales Trend Over Time")
sales_by_date = data.groupby('Order Date')['Sales'].sum().reset_index()

fig1, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(sales_by_date['Order Date'], sales_by_date['Sales'], label='Sales', color='red')
ax1.set_title('Sales Trend Over Time')
ax1.set_xlabel('Date')
ax1.set_ylabel('Sales')
ax1.grid(True)
ax1.legend()
fig1.autofmt_xdate()
st.pyplot(fig1)

# Lag Feature Creation
def create_lagged_features(data, lag=1):
    lagged_data = data.copy()
    for i in range(1, lag + 1):
        lagged_data[f'lag_{i}'] = lagged_data['Sales'].shift(i)
    return lagged_data

lag = st.slider("Select number of lag days for feature creation", 1, 30, 5)
sales_with_lags = create_lagged_features(data[['Order Date', 'Sales']], lag)
sales_with_lags = sales_with_lags.dropna()

# Train-Test Split
X = sales_with_lags.drop(columns=['Order Date', 'Sales'])
y = sales_with_lags['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# XGBoost Model Training
model_xgb = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5)
model_xgb.fit(X_train, y_train)

# Prediction and Evaluation
predictions_xgb = model_xgb.predict(X_test)
rmse_xgb = np.sqrt(mean_squared_error(y_test, predictions_xgb))

st.subheader(f"XGBoost RMSE: {rmse_xgb:.2f}")

# Plot Predictions
st.subheader("Actual vs Predicted Sales")
fig2, ax2 = plt.subplots(figsize=(12, 6))
ax2.plot(y_test.index, y_test, label='Actual Sales', color='red')
ax2.plot(y_test.index, predictions_xgb, label='Predicted Sales', color='green')
ax2.set_title('Sales Forecasting using XGBoost')
ax2.set_xlabel('Index')
ax2.set_ylabel('Sales')
ax2.legend()
ax2.grid(True)
st.pyplot(fig2)
