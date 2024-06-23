import requests

url = "https://api.alpaca.markets/v2/account"

headers = {
    "accept": "application/json",
    "APCA-API-KEY-ID": "AKSQ7NJJ7DPVHTKEEFO6",
    "APCA-API-SECRET-KEY": "vhH9BjTwfcA0mX2RvpIVs285UWFbgNjeGCAkUxgn"
}

response = requests.get(url, headers=headers)

print(response.text)