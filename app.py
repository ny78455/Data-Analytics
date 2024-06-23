from flask import Flask, request, render_template, send_from_directory, jsonify, redirect, url_for
from apscheduler.schedulers.background import BackgroundScheduler
from src.pipeline.pre_pipeline import StockDataDownloader, StockDataPipeline
from src.pipeline.train_pipeline import TrainPipeline
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.pipeline.visualization_pipeline import RunVisPipeline
import pandas as pd
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
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import threading
import requests
from dateutil import parser

# Setup logging
logging.basicConfig(level=logging.INFO)

load_dotenv() 

app = Flask(__name__)

CORS(app)


# Alpaca API setup
API_KEY = os.getenv('APCA_API_KEY_ID')
API_SECRET = os.getenv('APCA_API_SECRET_KEY')
BASE_URL = 'https://paper-api.alpaca.markets'  # For paper trading

stock_client = StockClient(API_KEY, API_SECRET, BASE_URL)
crypto_client = CryptoHistoricalDataClient()

# Ticker, period, and interval variables
ticker = 'AAPL'  # Default ticker
period = '3mo'   # Default period
interval = '1h'  # Default interval
refresh_task_enabled = False

# Initialize scheduler
scheduler = BackgroundScheduler(daemon=True)

def scheduled_job():
    if refresh_task_enabled:
        global ticker, period, interval

        try:
        # Download stock data
            downloader = StockDataDownloader(ticker=ticker, period=period, interval=interval)
            downloader.download_data()

        # Process the downloaded data
            pipeline = StockDataPipeline(ticker, period, interval)
            pipeline.run_pipeline()

        # Train the model
            train_pipeline = TrainPipeline()
            train_pipeline.run_pipeline()

        # Visualize the data
            vis_pipeline = RunVisPipeline()
            vis_pipeline.run_pipeline(MF=1.0, x=None, y=None)

            graph_filename = 'plot.html'  # Adjust this based on your actual filename
            return jsonify({'graphUrl': f'/renderplot/{graph_filename}'})
        except Exception as e:
            print("Error:", e)
            return jsonify({'error': str(e)}), 500

print("Scheduled job executed")


@app.route('/')
def index():
    return render_template('index_new.html')  # Render the form

@app.route('/process', methods=['POST'])
def process():
    global ticker, period, interval
    try:
        data = request.json  # Access JSON data sent from frontend
        ticker = data.get('ticker', 'AAPL')
        period = data.get('period', '3mo')
        interval = data.get('interval', '1h')
        
        # Process once
        return scheduled_job()
    except Exception as e:
        print("Error:", e)
        return jsonify({'error': str(e)}), 500

@app.route('/toggle_refresh', methods=['POST'])
def toggle_refresh():
    global refresh_task_enabled
    refresh_task_enabled = not refresh_task_enabled
    if refresh_task_enabled:
        if not scheduler.get_job('scheduled_job'):
            scheduler.add_job(scheduled_job, 'interval', seconds=80, id='scheduled_job')
        scheduler.start()
    else:
        scheduler.remove_all_jobs()
    return jsonify({'status': 'success', 'refresh_task_enabled': refresh_task_enabled})

@app.route('/update_plot', methods=['POST'])
def update_plot():
    try:
        data = request.json
        MF = float(data.get('MF'))
        x = pd.to_datetime(data.get('x'), format='%Y-%m-%dT%H:%M', utc=True)
        y = pd.to_datetime(data.get('y'), format='%Y-%m-%dT%H:%M', utc=True)

        vis_pipeline = RunVisPipeline()
        vis_pipeline.run_pipeline(MF=MF, x=x, y=y)

        graph_filename = 'plot.html'  # Adjust this based on your actual filename
        return jsonify({'graphUrl': f'/renderplot/{graph_filename}'})

    except Exception as e:
        print("Error:", e)
        return jsonify({'error': str(e)}), 500

@app.route('/renderplot/<path:plot_filename>')
def render_plot(plot_filename):
    try:
        # Serve the generated plot file from the templates directory
        return send_from_directory('templates', plot_filename)

    except Exception as e:
        print("Error:", e)
        return render_template('error.html', error="An error occurred while rendering the plot.")

@app.route('/form')
def home():
    return render_template('form_new.html')  # Render the home page or dashboard

@app.route('/predictdata', methods=['POST'])
def predict_datapoint():
    try:
        # Mock data processing and prediction (replace with actual logic)
        data = CustomData(
            Year=int(request.form.get('year', 0)),
            Month=int(request.form.get('month', 0)),
            Day=int(request.form.get('day', 0)),
            Hour=int(request.form.get('hour', 0)),
            Minute=int(request.form.get('minute', 0)),
            EMASignal=int(request.form.get('emasignal', 0)),
            isPivot=int(request.form.get('ispivot', 0)),
            CHOCH_pattern_detected=int(request.form.get('choch_pattern_detected', 0)),
            fibonacci_signal=int(request.form.get('fibonacci_signal', 0)),
            SL=float(request.form.get('sl', 0)),
            TP=float(request.form.get('tp', 0)),
            MinSwing=float(request.form.get('minswing', 0)),
            MaxSwing=float(request.form.get('maxswing', 0)),
            LBD_detected=int(request.form.get('lbd_detected', 0)),
            LBH_detected=int(request.form.get('lbh_detected', 0)),
            SR_signal=int(request.form.get('sr_signal', 0)),
            isBreakOut=int(request.form.get('isbreakout', 0)),
            candlestick_signal=int(request.form.get('candlestick_signal', 0)),
            result=int(request.form.get('result', 0)),
            signal1=int(request.form.get('signal1', 0)),
            buy_signal=int(request.form.get('buy_signal', 0)),
            Position=int(request.form.get('position', 0)),
            sell_signal=int(request.form.get('sell_signal', 0)),
            fractal_high=float(request.form.get('fractal_high', 0)),
            fractal_low=float(request.form.get('fractal_low', 0)),
            buy_signal1=int(request.form.get('buy_signal1', 0)),
            sell_signal1=int(request.form.get('sell_signal1', 0)),
            fractals_high=int(request.form.get('fractals_high', 0)),
            fractals_low=int(request.form.get('fractals_low', 0)),
            VSignal=int(request.form.get('vsignal', 0)),
            PriceSignal=int(request.form.get('pricesignal', 0)),
            TotSignal=int(request.form.get('totsignal', 0)),
            SLSignal=int(request.form.get('slsignal', 0)),
            grid_signal=int(request.form.get('grid_signal', 0)),
            ordersignal=int(request.form.get('ordersignal', 0)),
            SLSignal_heiken=float(request.form.get('slsignal_heiken', 0)),
            EMASignal1=int(request.form.get('emasignal1', 0)),
            long_signal=int(request.form.get('long_signal', 0)),
            martiangle_signal=int(request.form.get('martiangle_signal', 0)),
            Candle_direction=int(request.form.get('Candle_direction', 0))
        )
        
        # Perform prediction
        predict_pipeline = PredictPipeline()
        input_data = data.get_data_as_array().astype(float)  # Convert to float array
        results = predict_pipeline.predict(input_data)
        print("Predicted Results:", results)

        return render_template('form_new.html', results=int(results[0]))

    except Exception as e:
        print("Error:", e)
        return render_template('error.html', error=str(e))

@app.route('/login')
def login():
    return render_template('login.html')  # Render the login page

@app.route('/login/google')
def login_google():
    return redirect(url_for('index'))  # Redirect to the index page

@app.route('/global_market')
def global_market():
    return render_template('global_market.html')

@app.route('/portfolio')
def portfolio_index():
    return render_template('portfolio.html')

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
            last_price = stock_client.get_latest_bar(symbol).c

        if last_price is None:
            raise ValueError("Unable to fetch last price for the symbol")

        price = last_price

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
            last_price = stock_client.get_latest_bar(symbol).c

        if last_price is None:
            raise ValueError("Unable to fetch last price for the symbol")

        price = last_price

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


def read_last_row(file_path):
    return pd.read_csv(file_path).iloc[-1]

# Function to execute buy orders
def buy_stock(symbol, amount):
    buy_data = {
        'symbol': symbol,
        'amount': amount,
        'order_type': 'market'
    }
    response = requests.post('http://127.0.0.1:5000/api/buy', json=buy_data)
    return response

# Function to execute sell orders
def sell_stock(symbol, amount):
    sell_data = {
        'symbol': symbol,
        'amount': amount,
        'order_type': 'market'
    }
    response = requests.post('http://127.0.0.1:5000/api/sell', json=sell_data)
    return response

# Function to check if the current datetime is within 20 seconds of the signal datetime
def is_within_timeframe(signal_time):
    current_time = datetime.now()
    return abs((current_time - signal_time).total_seconds()) <= 60

# Function to start trading based on signals
def start_trading(amount, symbol):
    try:
        cash = amount
        position = 0
        buy_price = None
        sell_due_time = None
        missed_sell_attempt = False

        while True:
            try:
                # Read the last row of the CSV
                last_row = read_last_row('check1.csv')
                prediction = int(last_row['Prediction'])
                close_price = float(last_row['Close'])
                signal_time = parser.parse(last_row['Datetime'])

                if position > 0 and datetime.now() >= sell_due_time:
                    # Try to sell the current position after the hold period
                    response = sell_stock(symbol, position * close_price)
                    if response.status_code == 200:
                        cash += response.json().get('cash')
                        position = 0
                        sell_due_time = None
                        missed_sell_attempt = False
                        print(f"Selling at {close_price} on {datetime.now()}")
                    else:
                        logging.error(f"Failed to sell: {response.json().get('error')}")
                        missed_sell_attempt = True

                if cash > 0 and prediction in [2, 3, 4, 5] and is_within_timeframe(signal_time):
                    # Buy shares if the prediction is valid
                    buy_price = close_price
                    qty_to_buy = cash / buy_price
                    response = buy_stock(symbol, qty_to_buy * buy_price)
                    if response.status_code == 200:
                        position = qty_to_buy
                        cash -= qty_to_buy * buy_price
                        hold_period = prediction + 1
                        sell_due_time = datetime.now() + timedelta(minutes=hold_period)
                        missed_sell_attempt = False
                        print(f"Buying at {buy_price} on {datetime.now()}, hold for {hold_period} minutes")

                time.sleep(60)

            except Exception as e:
                logging.error(f"Error in trading loop: {e}")
                traceback.print_exc()

    except Exception as e:
        logging.error(f"Error in start_trading function: {e}")
        traceback.print_exc()

def start_automation(amount, symbol):
    try:
        thread = threading.Thread(target=start_trading, args=(amount, symbol))
        thread.start()

        return jsonify({'status': 'success', 'message': 'Trading automation started'}), 200

    except Exception as e:
        logging.error(f"Unexpected error in start_automation endpoint: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': 'Error processing start automation request'}), 500

@app.route('/start-automation', methods=['POST'])
def handle_start_automation():
    try:
        data = request.json
        amount = float(data['amount'])
        symbol = data['symbol']

        return start_automation(amount, symbol)

    except KeyError as e:
        logging.error(f"Missing key in request JSON: {str(e)}")
        return jsonify({'error': 'Missing key in request JSON'}), 400

    except ValueError as e:
        logging.error(f"Invalid value in request JSON: {str(e)}")
        return jsonify({'error': f"Invalid value in request JSON: {str(e)}"}), 400

    except Exception as e:
        logging.error(f"Unexpected error in handle_start_automation endpoint: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': 'Error processing start automation request'}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
