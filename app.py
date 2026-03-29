import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import yfinance as yf
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

# -----------------------------
# UI
# -----------------------------
st.title('📈 Stock Price Prediction App')

stocks = ["AAPL", "TSLA", "GOOG", "MSFT", "RELIANCE.NS", "TCS.NS", "INFY.NS"]
user_input = st.selectbox("Select Stock", stocks)

# -----------------------------
# Data Load
# -----------------------------
start = '2010-01-01'

df = yf.download(user_input, start=start, end=None)

if df.empty:
    st.error("Invalid ticker or no data found!")
    st.stop()

# -----------------------------
# Data Overview
# -----------------------------
st.subheader('📊 Data Summary')
st.write(df.describe())

# -----------------------------
# Price Chart
# -----------------------------
st.subheader('📈 Closing Price vs Time')
fig = plt.figure(figsize=(12, 6))
plt.plot(df['Close'], label='Close Price')
plt.legend()
st.pyplot(fig)

# -----------------------------
# Moving Averages
# -----------------------------
st.subheader('📉 Moving Averages (MA100 & MA200)')
ma100 = df['Close'].rolling(100).mean()
ma200 = df['Close'].rolling(200).mean()

fig2 = plt.figure(figsize=(12, 6))
plt.plot(df['Close'], 'b', label='Close')
plt.plot(ma100, 'r', label='MA100')
plt.plot(ma200, 'g', label='MA200')
plt.legend()
st.pyplot(fig2)

# -----------------------------
# Train-Test Split
# -----------------------------
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):])

# -----------------------------
# Scaling
# -----------------------------
scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_training)

# -----------------------------
# Load Model
# -----------------------------
model = load_model('keras_model.h5')

# -----------------------------
# Prepare Test Data
# -----------------------------
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

input_data = scaler.transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# -----------------------------
# Prediction (Past)
# -----------------------------
y_predicted = model.predict(x_test)

# ✅ FIXED INVERSE SCALING
y_predicted = scaler.inverse_transform(y_predicted)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# -----------------------------
# Plot Prediction
# -----------------------------
st.subheader('📊 Predicted vs Original Price')
fig3 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig3)

# -----------------------------
# Accuracy
# -----------------------------
mae = mean_absolute_error(y_test, y_predicted)

st.subheader("📉 Model Accuracy")
st.write(f"Mean Absolute Error: {mae:.2f}")

# -----------------------------
# Confidence Score
# -----------------------------
confidence = max(0, 100 - (mae / np.mean(y_test)) * 100)

st.subheader("🎯 Confidence Score")
st.write(f"{confidence:.2f}%")

# -----------------------------
# Tomorrow Prediction
# -----------------------------
last_100_days = final_df.tail(100)
last_100_scaled = scaler.transform(last_100_days)

X_input = np.array([last_100_scaled])

tomorrow_pred = model.predict(X_input)

# ✅ FIXED HERE
tomorrow_price = scaler.inverse_transform(tomorrow_pred)[0][0]

st.subheader("📅 Tomorrow's Predicted Price")
st.write(f"💰 {tomorrow_price:.2f}")

# -----------------------------
# Buy/Sell Signal
# -----------------------------
last_close = float(df['Close'].iloc[-1])

st.subheader("📊 Trading Signal")
st.write("Last Close:", last_close)
st.write("Tomorrow Predicted:", tomorrow_price)

st.warning("⚠️ This model is for educational purposes and may not accurately predict real market movements.")

if tomorrow_price > last_close:
    st.success("🟢 BUY Signal (Expected Increase)")
elif tomorrow_price < last_close:
    st.error("🔴 SELL Signal (Expected Decrease)")
else:
    st.info("⚪ HOLD")