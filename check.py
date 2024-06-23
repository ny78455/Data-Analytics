import pandas as pd
import alpaca_trade_api as tradeapi
import time
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta, timezone

load_dotenv()
# Alpaca API credentials
API_KEY = os.getenv('APCA_API_KEY_ID')
API_SECRET = os.getenv('APCA_API_SECRET_KEY')
BASE_URL = 'https://paper-api.alpaca.markets'   # Use 'https://api.alpaca.markets' for live trading

# Initialize Alpaca API
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL)

# Function to read the CSV file
def read_csv(file_path):
    return pd.read_csv(file_path)

# Function to execute buy orders
def buy_stock(symbol, quantity):
    try:
        api.submit_order(
            symbol=symbol,
            qty=quantity,
            side='buy',
            type='market',
            time_in_force='gtc'
        )
        print(f"Buy order placed for {symbol}")
    except Exception as e:
        print(f"Error placing buy order: {e}")

# Function to execute sell orders
def sell_stock(symbol, quantity):
    try:
        api.submit_order(
            symbol=symbol,
            qty=quantity,
            side='sell',
            type='market',
            time_in_force='gtc'
        )
        print(f"Sell order placed for {symbol}")
    except Exception as e:
        print(f"Error placing sell order: {e}")

# Function to process the CSV and trade based on signals
def process_and_trade(symbol, quantity):
    df = pd.read_csv('check1.csv').iloc[-1]

    prediction = int(df['Prediction'])
    if prediction == 1:
        # Sell signal
        sell_stock(symbol, quantity)
    elif prediction in [0, 2, 3, 4, 5]:
        # Buy signal
        buy_stock(symbol, quantity)
        # Wait for the specified number of candles before selling
        candles_to_wait = prediction + 1
        time.sleep(candles_to_wait * 60)  # Assuming each candle represents 1 minute
        sell_stock(symbol, quantity)

# Main loop to continuously check for updates
def main():
    symbol = 'AVAX/USD'  # Example stock symbol
    quantity = 1   # Number of shares to trade
    while True:
        df = pd.read_csv('check1.csv')
        last_row = df.iloc[-1]
        prediction_time = pd.to_datetime(last_row['Datetime'])  # Assuming the datetime column is named 'Datetime'

        if prediction_time.tzinfo is None:
            prediction_time = prediction_time.replace(tzinfo=timezone.utc)

        current_time = datetime.now(timezone.utc)  # Use UTC time for comparison
        time_difference = abs(current_time - prediction_time)
        
        if time_difference <= timedelta(minutes=2):
            process_and_trade(symbol, quantity)
        
        time.sleep(60)  # Wait 1 minute before checking again

if __name__ == "__main__":
    main()
