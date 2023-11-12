import os
import pandas as pd
import numpy as np
from datetime import date 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import pandas_datareader.data as pdr
import yfinance as yfin
import streamlit as st

# Override yfinance with pandas_datareader
yfin.pdr_override()

# Set start and end dates
start = '2015-01-01'
end = '2025-12-31'

# Streamlit setup
st.title('Stock Price Prediction')
user_input = st.text_input('Enter Stock Ticker', 'AAPL')
df = pdr.get_data_yahoo(user_input, start=start, end=end)

# Display basic statistics
st.subheader('Data from 2015-2025')
st.write(df.describe())

# Visualization
st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig)

# Split data into training and testing
data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70): int(len(df))])

print(data_training.shape)
print(data_testing.shape)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_training)

# Print the current working directory for debugging
print("Current Working Directory:", os.getcwd())

# Load the Keras model
script_directory = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_directory, 'keras_models.h5')

print("Model Path:", model_path)

if os.path.exists(model_path):
    model = load_model(model_path)
else:
    st.error(f"Model file '{model_path}' not found.")

# ... (rest of the code remains unchanged)


# Prepare data for prediction
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

# Prepare test data
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i - 100:i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)

# Invert scaling to get the actual prices
scaler = scaler.scale_
scale_factor = 1 / scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Final graph
st.subheader('Prediction vs Original')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig2)
