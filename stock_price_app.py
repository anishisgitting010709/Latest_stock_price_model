import streamlit as st
import yfinance as yf
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Load the pre-trained model
model_path = "Latest_stock_price_model.keras"
model = load_model(model_path)

# Function to predict future stock prices
def predict_future(no_of_days, prev_100, model, scaler):
    future_predictions = []
    for i in range(no_of_days):
        next_day = model.predict(prev_100)
        prev_100 = np.append(prev_100[:, 1:, :], [[next_day[0]]], axis=1)
        future_predictions.append(float(scaler.inverse_transform(next_day)[0][0]))
    return future_predictions

# Streamlit UI
st.title("Stock Price Prediction using LSTM")
st.sidebar.header("User Input")

# Sidebar inputs
stock_symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., ^NSEI):", "^NSEI")
prediction_days = st.sidebar.slider("Number of days to predict:", 1, 30, 5)

# Load stock data
start = datetime.now() - pd.DateOffset(years=20)
end = datetime.now()

st.write(f"Fetching data for {stock_symbol} from {start.date()} to {end.date()}")
data = yf.download(stock_symbol, start, end)

if not data.empty:
    st.write("## Stock Data")
    st.line_chart(data['Adj Close'])

    # Preprocess data for predictions
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Adj Close']].values)
    last_100_scaled = scaled_data[-100:].reshape(1, -1, 1)

    # Predict future prices
    future_results = predict_future(prediction_days, last_100_scaled, model, scaler)

    # Display results
    st.write("## Future Predictions")
    prediction_df = pd.DataFrame({
        "Day": [f"Day {i+1}" for i in range(prediction_days)],
        "Predicted Price": future_results
    })

    st.write(prediction_df)

    # Plot predictions
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prediction_df["Day"], y=prediction_df["Predicted Price"], mode='lines+markers'))
    fig.update_layout(title="Future Price Predictions", xaxis_title="Day", yaxis_title="Price")
    st.plotly_chart(fig)
else:
    st.error("Failed to fetch stock data. Please check the symbol and try again.")
