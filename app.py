import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt

model = load_model('Stock_Price_Prediction_Model.keras')

st.header('Stock Market Predictor')

stock =st.text_input('Enter Stock Symbol', 'TCS.NS')
start = '2014-01-01'
end = '2025-03-31'

data = yf.download(stock, start ,end)

st.subheader('Stock Data')
st.write(data)

data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

st.subheader('Price vs MA50')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r',label='Moving Avg of 50 days')
plt.plot(data.Close, 'g',label='Stock Closing Price')
plt.legend()
plt.show()
st.pyplot(fig1)

st.subheader('Price vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r',label='Moving Avg of 50 days')
plt.plot(ma_100_days, 'b',label='Moving Avg of 100 days')
plt.plot(data.Close, 'g',label='Stock Closing Price')
plt.legend()
plt.show()
st.pyplot(fig2)

st.subheader('Price vs MA100 vs MA200')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8,6))
plt.plot(ma_100_days, 'r',label='Moving Avg of 100 days')
plt.plot(ma_200_days, 'b',label='Moving Avg of 200 days')
plt.plot(data.Close, 'g',label='Stock Closing Price')
plt.legend()
plt.show()
st.pyplot(fig3)

x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])

x,y = np.array(x), np.array(y)

predict = model.predict(x)

scale = 1/scaler.scale_

predict = predict * scale
y = y * scale

st.subheader('Original Price vs Predicted Price')

min_len = min(len(predict), len(y))
predict = predict[:min_len]
y = y[:min_len]

fig4 = plt.figure(figsize=(8,6))
plt.plot(data.index[-min_len:], predict, 'r', label='Predicted Price')  
plt.plot(data.index[-min_len:], y, 'g', label='Original Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)  
plt.show()
st.pyplot(fig4)


future_days = 30
future_predictions = []

last_100_days = data_test_scale[-100:]  
current_input = last_100_days.reshape(1, 100, 1)

for _ in range(future_days):
    next_price = model.predict(current_input)[0, 0]  
    future_predictions.append(next_price)
    
    current_input = np.append(current_input[:, 1:, :], [[[next_price]]], axis=1)

future_predictions = np.array(future_predictions).reshape(-1, 1)
future_predictions = scaler.inverse_transform(future_predictions).flatten()

last_date = data.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days)

# Future predictions
st.subheader('Future Predicted Prices')
fig5 = plt.figure(figsize=(10,6)) 
plt.plot(future_dates, future_predictions, 'r', marker='o', linestyle='-', label='Predicted Future Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
st.pyplot(fig5)
