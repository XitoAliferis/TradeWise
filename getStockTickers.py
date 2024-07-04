import yfinance as yf
import pandas as pd
from tqdm import tqdm

# Read the file and extract tickers
def read_tickers(file_path):
    tickers = []
    with open(file_path, 'r') as file:
        next(file)  # Skip the header line
        for line in file:
            parts = line.split('|')
            if len(parts) > 0:
                tickers.append(parts[0].strip())
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

# Path to the stocks.txt file
file_path = 'stocks.txt'  # Assuming stocks.txt is in the same directory as this script

# Read tickers from the provided file
tickers = read_tickers(file_path)

# Fetch market caps
market_caps = fetch_market_caps(tickers)

# Sort by market cap
sorted_market_caps = sorted(market_caps, key=lambda x: x[1], reverse=True)

# Limit to top 1000 tickers
top_1000_market_caps = sorted_market_caps[:1000]

# Convert to DataFrame and save to a text file
df = pd.DataFrame(top_1000_market_caps, columns=["Ticker", "MarketCap"])
df.to_csv("top_1000_tickers_by_market_cap.txt", index=False)

print(df.head())
