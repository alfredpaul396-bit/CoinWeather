# 🌦️ CoinWeather

**A plain-English risk forecast for your crypto portfolio.** (Crypto Portfolio Risk Checker)

NEOMA MSc Fintech & DeFi – Final Project

CoinWeather turns risk into a weather forecast: ☀️ Calm (Low), 🌦️ Choppy (Medium) or ⛈️ Stormy (High). Enter your crypto holdings and see, in plain English, how risky the portfolio is, how diversified it really is, and one concrete way to improve it.

## Features

1. **Portfolio input**: pick coins (BTC, ETH, SOL, BNB, XRP, ADA, DOGE, USDC) and enter amounts in EUR or percentages. Weights are normalized automatically.
2. **Allocation chart**: donut chart of the split.
3. **Risk metrics**: annualized volatility, maximum drawdown, risk score (Low / Medium / High) with a gauge.
4. **Diversification check**: correlation heatmap plus a warning such as "78% of your portfolio is in assets that move closely together".
5. **Plain-English summary**: automatic explanation with one calculated suggestion (e.g. adding USDC).

The sidebar has one-click examples (Bitcoin only, Bitcoin + USDC, equal 5-coin split, Dogecoin only) for a quick demo.

Extras: 1-day Value at Risk, risk contribution per coin, a lower-risk mix suggestion, a past-performance chart, and a downloadable CSV report.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Create a new **public** GitHub repository (e.g. `coinweather`).
2. Upload every file in this folder, keeping the structure:
   ```
   app.py
   prices.csv
   requirements.txt
   README.md
   update_prices.py
   .streamlit/config.toml
   ```
   (On github.com: **Add file → Upload files**, then drag the files in. For `.streamlit/config.toml`, use **Add file → Create new file** and type `.streamlit/config.toml` as the name.)
3. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, click **Create app**.
4. Choose the repository, branch `main`, main file `app.py`, then **Deploy**.
5. Copy the app URL for the slides and the course upload.

## Refreshing the price data

`prices.csv` holds daily closing prices (USD) for the 8 coins. To replace it with fresh Yahoo Finance data, run on your laptop:

```bash
pip install yfinance
python update_prices.py
```

Then upload the new `prices.csv` to GitHub. The app redeploys automatically.

## Method

- Daily return = % change of daily closing price
- Portfolio return = weighted sum of coin returns (constant weights)
- Annualized volatility = std of daily portfolio returns × √365
- Max drawdown = largest fall from a running peak
- "Moves closely together" = correlation ≥ 0.60
- Risk score: Low / Calm < 40%, Medium / Choppy 40–80%, High / Stormy > 80% volatility

*Educational tool, not financial advice. Results are based on past prices.*
