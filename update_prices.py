"""
Refresh prices.csv with real daily closing prices from Yahoo Finance.

Run this ONCE on your own laptop before the demo (not on Streamlit Cloud):

    pip install yfinance
    python update_prices.py

The app itself never goes online: it only reads the saved prices.csv.
"""

from pathlib import Path

import pandas as pd
import yfinance as yf

COINS = ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "USDC"]
PERIOD = "2y"
OUT = Path(__file__).parent / "prices.csv"

tickers = {f"{c}-USD": c for c in COINS}
raw = yf.download(list(tickers), period=PERIOD, interval="1d", auto_adjust=True, progress=False)["Close"]
prices = raw.rename(columns=tickers)[COINS].dropna()
prices.index = pd.to_datetime(prices.index).date
prices.index.name = "date"

if len(prices) < 300:
    raise SystemExit(f"Only {len(prices)} days downloaded - check your internet connection and try again.")

prices.to_csv(OUT, float_format="%.6f")
print(f"Saved {len(prices)} days ({prices.index.min()} to {prices.index.max()}) to {OUT}")
