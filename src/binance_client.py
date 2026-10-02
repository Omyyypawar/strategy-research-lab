import os
from binance.client import Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Fetch credentials
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")
USE_TESTNET = os.getenv("BINANCE_TESTNET", "True").lower() == "true"

# Initialize client
client = Client(API_KEY, API_SECRET, testnet=USE_TESTNET)

def get_latest_price(symbol="BTCUSDT"):
    """Fetch the latest market price for a given symbol."""
    ticker = client.get_symbol_ticker(symbol=symbol)
    return float(ticker["price"])

def get_recent_klines(symbol="BTCUSDT", interval="1m", limit=100):
    """Fetch recent historical candle data."""
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    return klines

if __name__ == "__main__":
    price = get_latest_price("BTCUSDT")
    print(f"✅ Connected to Binance Testnet! Latest BTCUSDT price: {price}")
