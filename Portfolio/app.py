import logging
import traceback
from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import REST as StockClient
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
import os
from dotenv import load_dotenv
import time
import requests
from datetime import datetime, timedelta
import pandas as pd
import threading

# Setup logging
logging.basicConfig(level=logging.INFO)

load_dotenv()  # Load environment variables from .env file

app = Flask(__name__)
CORS(app)

# Alpaca API setup
API_KEY = os.getenv('APCA_API_KEY_ID')
API_SECRET = os.getenv('APCA_API_SECRET_KEY')
BASE_URL = 'https://paper-api.alpaca.markets'  # For paper trading

stock_client = StockClient(API_KEY, API_SECRET, BASE_URL)
crypto_client = CryptoHistoricalDataClient()

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/portfolio', methods=['GET'])
def portfolio():
    try:
        account = stock_client.get_account()
        cash = float(account.cash)
        equity = float(account.equity)

        holdings = stock_client.list_positions()
        holdings_list = []

        if not holdings:
            return jsonify({'error': 'No holdings found'}), 404

        for holding in holdings:
            asset_symbol = holding.symbol
            asset_quantity = float(holding.qty)
            asset_market_value = float(holding.market_value)
            current_price = float(holding.current_price)
            profitloss = float(holding.unrealized_pl)

            holdings_list.append({
                'symbol': asset_symbol,
                'quantity': asset_quantity,
                'market_value': asset_market_value,
                'current_price': current_price,
                'profit_and_loss': profitloss
            })

        transactions = stock_client.get_activities(activity_types='FILL')
        transactions_list = []

        if not transactions:
            return jsonify({'error': 'No transactions found'}), 404

        for transaction in transactions:
            asset_symbol = transaction.symbol
            trade_type = transaction.side
            trade_quantity = float(transaction.qty)
            average_cost = float(transaction.price)
            amount = float(trade_quantity * average_cost)
            status = transaction.order_status
            date = transaction.transaction_time

            transactions_list.append({
                'symbol': asset_symbol,
                'type': trade_type,
                'quantity': trade_quantity,
                'average_cost': average_cost,
                'amount': amount,
                'status': status,
                'date': date
            })

        portfolio_data = {
            'cash': cash,
            'equity': equity,
            'holdings': holdings_list,
            'transactions': transactions_list
        }
        return jsonify(portfolio_data)

    except Exception as e:
        logging.error(f"Error fetching portfolio data: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': 'Error fetching portfolio data'}), 500

@app.route('/api/price/<path:symbol>', methods=['GET'])
def get_price(symbol):
    try:
        # Validate and split the symbol correctly for cryptocurrency symbols
        if '/' in symbol:
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=180)
            request_params = CryptoBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=TimeFrame.Minute,
                start=start_time.isoformat(),
                end=end_time.isoformat()
            )
            bars = crypto_client.get_crypto_bars(request_params)
            if bars.df.empty:
                raise ValueError(f"No bars found for symbol '{symbol}'")
            
            price = bars.df['close'].iloc[-1]
        else:
            # It's a stock symbol
            latest_trade = stock_client.get_latest_trade(symbol)
            if latest_trade is None:
                raise ValueError(f"Unable to fetch latest trade for symbol '{symbol}'")

            price = latest_trade.price

        if price is None:
            raise ValueError(f"Unable to fetch price for the symbol '{symbol}'")

        # Return the price as JSON
        return jsonify({'symbol': symbol, 'price': price})

    except Exception as e:
        logging.error(f"Error fetching price for {symbol}: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f"Error fetching price for {symbol}"}), 500

@app.route('/api/buy', methods=['POST'])
def buy():
    try:
        data = request.get_json()
        symbol = data['symbol']
        amount = float(data['amount'])
        order_type = data['order_type']

        if "/" in symbol:  # Check if the symbol is for a cryptocurrency
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=180)
            request_params = CryptoBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=TimeFrame.Minute,
                start=start_time.isoformat(),
                end=end_time.isoformat()
            )
            bars = crypto_client.get_crypto_bars(request_params)
            last_price = bars.df['close'].iloc[-1]
        else:  # It's a stock symbol
            last_price = stock_client.get_latest_trade(symbol)

        if last_price is None:
            raise ValueError("Unable to fetch last price for the symbol")

        price = last_price.price

        logging.info(f"Received buy request: Symbol={symbol}, Amount={amount}, Order Type={order_type}")
        logging.info(f"Fetched price for {symbol}: {price}")

        qty = float(amount / price)
        qty = round(qty, 2)
        logging.info(f"Calculated quantity: {qty}")

        if '/' in symbol:
            time_in_force = 'ioc'  # For symbols containing '/', use IOC
        else:
            time_in_force = 'day' if qty != int(qty) else 'gtc'

        if order_type == 'market':
            order = stock_client.submit_order(
                symbol=symbol,
                qty=qty,
                side='buy',
                type='market',
                time_in_force=time_in_force
            )
        elif order_type == 'limit':
            order = stock_client.submit_order(
                symbol=symbol,
                qty=qty,
                side='buy',
                type='limit',
                time_in_force=time_in_force,
                limit_price=price
            )

        time.sleep(2)

        return jsonify({'message': 'Buy order placed successfully!'})

    except KeyError as e:
        logging.error(f"Invalid symbol '{symbol}': {str(e)}")
        return jsonify({'error': f"Invalid symbol '{symbol}'"}), 400

    except ValueError as e:
        logging.error(f"Invalid amount '{data['amount']}': {str(e)}")
        return jsonify({'error': f"Invalid amount '{data['amount']}'"}), 400

    except tradeapi.rest.APIError as e:
        error_msg = str(e)
        logging.error(f"Alpaca API error: {error_msg}")
        if 'insufficient' in error_msg.lower():
            return jsonify({'error': 'Insufficient funds to place order'}), 400
        elif 'market is closed' in error_msg.lower():
            return jsonify({'error': 'Market is closed, cannot place order'}), 400
        else:
            return jsonify({'error': 'Error placing order'}), 500

    except Exception as e:
        logging.error(f"Unexpected error in buy endpoint: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': 'Error processing buy request'}), 500

@app.route('/api/sell', methods=['POST'])
def sell():
    try:
        data = request.get_json()
        symbol = data['symbol']
        amount = float(data['amount'])
        order_type = data['order_type']

        if "/" in symbol:  # Check if the symbol is for a cryptocurrency
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=180)
            request_params = CryptoBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=TimeFrame.Minute,
                start=start_time.isoformat(),
                end=end_time.isoformat()
            )
            bars = crypto_client.get_crypto_bars(request_params)
            last_price = bars.df['close'].iloc[-1]
        else:  # It's a stock symbol
            last_price = stock_client.get_latest_trade(symbol) 

        if last_price is None:
            raise ValueError("Unable to fetch last price for the symbol")

        price = last_price.price

        logging.info(f"Received buy request: Symbol={symbol}, Amount={amount}, Order Type={order_type}")
        logging.info(f"Fetched price for {symbol}: {price}")

        qty = float(amount / price)
        qty = round(qty, 2)
        logging.info(f"Calculated quantity: {qty}")

        if '/' in symbol:
            time_in_force = 'ioc'  # For symbols containing '/', use IOC
        else:
            time_in_force = 'day' if qty != int(qty) else 'gtc'

        if order_type == 'market':
            order = stock_client.submit_order(
                symbol=symbol,
                qty=qty,
                side='sell',
                type='market',
                time_in_force=time_in_force
            )
        elif order_type == 'limit':
            order = stock_client.submit_order(
                symbol=symbol,
                qty=qty,
                side='sell',
                type='limit',
                time_in_force=time_in_force,
                limit_price=price
            )

        time.sleep(2)

        return jsonify({'message': 'sell order placed successfully!'})

    except KeyError as e:
        logging.error(f"Invalid symbol '{symbol}': {str(e)}")
        return jsonify({'error': f"Invalid symbol '{symbol}'"}), 400

    except ValueError as e:
        logging.error(f"Invalid amount '{data['amount']}': {str(e)}")
        return jsonify({'error': f"Invalid amount '{data['amount']}'"}), 400

    except tradeapi.rest.APIError as e:
        error_msg = str(e)
        logging.error(f"Alpaca API error: {error_msg}")
        if 'insufficient' in error_msg.lower():
            return jsonify({'error': 'Insufficient funds to place order'}), 400
        elif 'market is closed' in error_msg.lower():
            return jsonify({'error': 'Market is closed, cannot place order'}), 400
        else:
            return jsonify({'error': 'Error placing order'}), 500

    except Exception as e:
        logging.error(f"Unexpected error in buy endpoint: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': 'Error processing sell request'}), 500

# Load the prediction data
data = pd.read_csv('check1.csv')

def start_trading(amount, hold_period, symbol):
    cash = amount
    position = 0

    for i in range(len(data)):
        prediction = data['Prediction'].iloc[i]
        close_price = data['Close'].iloc[i]
        
        if prediction == 1 and position > 0:
            # Sell the current position
            sell_data = {
                'symbol': symbol,
                'amount': position * close_price,
                'order_type': 'market'
            }
            response = requests.post('http://127.0.0.1:5000/api/sell', json=sell_data)
            if response.status_code == 200:
                cash = response.json().get('cash')
                position = 0
            else:
                logging.error(f"Failed to sell: {response.json().get('error')}")

            buy_index = i + hold_period
            if buy_index < len(data):
                buy_price = data['Close'].iloc[buy_index]
                qty_to_buy = (cash) / buy_price
                buy_data = {
                    'symbol': symbol,
                    'amount': qty_to_buy * buy_price,
                    'order_type': 'market'
                }
                response = requests.post('http://127.0.0.1:5000/api/buy', json=buy_data)
                if response.status_code == 200:
                    position = qty_to_buy
                    cash = 0
                else:
                    logging.error(f"Failed to buy: {response.json().get('error')}")

        elif prediction in [2, 3, 4, 5] and cash > 0:
            buy_price = close_price
            qty_to_buy = (cash) / buy_price
            buy_data = {
                'symbol': symbol,
                'amount': qty_to_buy * buy_price,
                'order_type': 'market'
            }
            response = requests.post('http://127.0.0.1:5000/api/buy', json=buy_data)
            if response.status_code == 200:
                position = qty_to_buy
                cash = 0
            else:
                logging.error(f"Failed to buy: {response.json().get('error')}")

            sell_index = i + hold_period + (prediction - 2)
            if sell_index < len(data):
                sell_price = data['Close'].iloc[sell_index]
                sell_data = {
                    'symbol': symbol,
                    'amount': position * sell_price,
                    'order_type': 'market'
                }
                response = requests.post('http://127.0.0.1:5000/api/sell', json=sell_data)
                if response.status_code == 200:
                    cash = response.json().get('cash')
                    position = 0
                else:
                    logging.error(f"Failed to sell: {response.json().get('error')}")

@app.route('/start-automation', methods=['POST'])
def start_automation():
    try:
        data = request.json
        amount = float(data['amount'])
        hold_period = int(data['holdPeriod'])
        symbol = data['symbol']

        thread = threading.Thread(target=start_trading, args=(amount, hold_period, symbol))
        thread.start()

        return jsonify({'status': 'success', 'message': 'Trading automation started'}), 200

    except KeyError as e:
        logging.error(f"Missing key in request JSON: {str(e)}")
        return jsonify({'error': 'Missing key in request JSON'}), 400

    except ValueError as e:
        logging.error(f"Invalid value in request JSON: {str(e)}")
        return jsonify({'error': f"Invalid value in request JSON: {str(e)}"}), 400

    except Exception as e:
        logging.error(f"Unexpected error in start_automation endpoint: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': 'Error processing start automation request'}), 500

if __name__ == '__main__':
    app.run(debug=True)

