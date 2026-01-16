from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

app = FastAPI()

@app.get("/")
def home():
    return {"status": "AI Stock Predictor Running"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = tf.keras.models.load_model("AlphaStocks AI.keras")

@app.get("/predict/{symbol}")
def predict(symbol: str):
    symbol = symbol.upper()

    data = yf.download(symbol, period="1y")

    if len(data) < 120:
        return {"error": "Not enough data to predict"}

    close_series = data["Close"]
    close_prices = close_series.values.reshape(-1, 1)

    # ---------------- MA100 ----------------
    ma100 = close_series.rolling(window=100).mean().iloc[-1]

    # ---------------- RSI (14 days) ----------------
    delta = close_series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_value = rsi.iloc[-1]

    # ---------------- Volume (latest) ----------------
    volume = float(data["Volume"].iloc[-1])

    # ---------------- Volatility (30-day) ----------------
    returns = close_series.pct_change()
    volatility = returns.rolling(30).std().iloc[-1]

    # ---------------- Model Prediction ----------------
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled = scaler.fit_transform(close_prices)

    X = scaled[-100:].reshape(1, 100, 1)
    pred_scaled = model.predict(X)
    pred = scaler.inverse_transform(pred_scaled)

    return {
        "symbol": symbol,
        "last_price": float(close_prices[-1][0]),
        "predicted_price": float(pred[0][0]),
        "ma100": float(ma100),
        "rsi": float(rsi_value),
        "volume": volume,
        "volatility": float(volatility)
    }
