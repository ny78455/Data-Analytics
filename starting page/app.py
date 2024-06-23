from flask import Flask, jsonify, request
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)  # Enable CORS

# Replace 'YOUR_API_KEY' with your actual Alpha Vantage API key
API_KEY = 'K0ZXNXSG6S88VW59'
BASE_URL = 'https://www.alphavantage.co/query'

@app.route('/api/stock/<symbol>', methods=['GET'])
def get_stock_data(symbol):
    try:
        response = requests.get(BASE_URL, params={
            'function': 'TIME_SERIES_INTRADAY',
            'symbol': symbol,
            'interval': '5min',
            'apikey': API_KEY
        })
        data = response.json()
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
