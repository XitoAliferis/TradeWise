import yfinance as yf
import pandas as pd
from tqdm import tqdm
import requests
import os

# Download the data from the official link
def download_tickers(url):
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for bad status
    lines = response.text.strip().split('\n')
    tickers = [line.split('|')[0].strip() for line in lines[1:]]  # Skip the header line
    return tickers

# Fetch market cap data with progress bar
def fetch_market_caps(tickers):
    market_caps = []
    for ticker in tqdm(tickers, desc="Fetching market caps"):
        try:
            stock = yf.Ticker(ticker)
            market_cap = stock.info.get('marketCap')
            if market_cap:
                market_caps.append((ticker, market_cap))
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
    return market_caps

# URL to the official link
url = 'https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt'

# Download and read tickers from the provided URL
tickers = download_tickers(url)

# Fetch market caps
market_caps = fetch_market_caps(tickers)

# Sort by market cap
sorted_market_caps = sorted(market_caps, key=lambda x: x[1], reverse=True)

# Limit to top 1000 tickers
top_1000_market_caps = sorted_market_caps[:1000]

# Convert to DataFrame and save to a text file
df = pd.DataFrame(top_1000_market_caps, columns=["Ticker", "MarketCap"])

# Define the output directory and create it if it does not exist
output_dir = "./data_files"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save the DataFrame to a CSV file
output_file = os.path.join(output_dir, "top_1000_tickers_by_market_cap.txt")
df.to_csv(output_file, index=False)