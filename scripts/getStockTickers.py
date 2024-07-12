import pandas as pd
import requests
from tqdm import tqdm
import yfinance as yf
import os
import concurrent.futures

# Function to download and parse NASDAQ tickers
def download_nasdaq_tickers(url):
    response = requests.get(url)
    response.raise_for_status()
    lines = response.text.strip().split('\n')
    tickers = [line.split('|')[0].strip() for line in lines[1:]]  # Skip the header line
    return tickers

# Function to download and parse TSX tickers
def download_tsx_tickers(url):
    response = requests.get(url)
    response.raise_for_status()
    lines = response.text.strip().split('\n')[2:]  # Skip the header lines
    tickers = [line.split('\t')[2].strip() for line in lines if len(line.split('\t')) > 2]  # Extract US Symbol
    return tickers

# Function to download and parse NYSE tickers from a text file
def download_nyse_tickers(url):
    response = requests.get(url)
    response.raise_for_status()
    lines = response.text.strip().split('\n')
    tickers = [line.split('\t')[1].strip() for line in lines[1:] if len(line.split('\t')) > 1]  # Extract Symbol
    return tickers

# Function to fetch market cap for a single ticker
def fetch_market_cap(ticker):
    try:
        stock = yf.Ticker(ticker)
        market_cap = stock.info.get('marketCap')
        if market_cap:
            return (ticker, market_cap)
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
    return None

# URLs to the official links
nasdaq_url = 'https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt'
tsx_url = 'https://www.tsx.com/files/trading/interlisted-companies.txt'
nyse_url = 'https://www.nyse.com/publicdocs/nyse/markets/nyse/NYSE_and_NYSE_MKT_Trading_Units_Daily_File.xls'

# Download and read tickers from the provided URLs
with concurrent.futures.ThreadPoolExecutor() as executor:
    future_to_tickers = {
        executor.submit(download_nasdaq_tickers, nasdaq_url): 'nasdaq',
        executor.submit(download_tsx_tickers, tsx_url): 'tsx',
        executor.submit(download_nyse_tickers, nyse_url): 'nyse'
    }
    tickers_result = {}
    for future in concurrent.futures.as_completed(future_to_tickers):
        exchange = future_to_tickers[future]
        try:
            tickers_result[exchange] = future.result()
        except Exception as e:
            print(f"Error fetching tickers from {exchange}: {e}")

# Combine tickers from all exchanges
all_tickers = tickers_result.get('nasdaq', []) + tickers_result.get('tsx', []) + tickers_result.get('nyse', [])

# Remove duplicates
all_tickers = list(set(all_tickers))

# Fetch market caps
with concurrent.futures.ThreadPoolExecutor() as executor:
    future_to_market_cap = {executor.submit(fetch_market_cap, ticker): ticker for ticker in all_tickers}
    market_caps = []
    for future in tqdm(concurrent.futures.as_completed(future_to_market_cap), total=len(future_to_market_cap), desc="Fetching market caps"):
        result = future.result()
        if result:
            market_caps.append(result)

# Sort by market cap
sorted_market_caps = sorted(market_caps, key=lambda x: x[1], reverse=True)

# Convert to DataFrame and save to a CSV file
df = pd.DataFrame(sorted_market_caps, columns=["Ticker", "MarketCap"])

# Define the output directory and create it if it does not exist
output_dir = "./data_files"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save the DataFrame to a CSV file
output_file = os.path.join(output_dir, "tickers_sorted_by_market_cap.txt")
df.to_csv(output_file, index=False)

print("Tickers sorted by market cap have been saved successfully.")
