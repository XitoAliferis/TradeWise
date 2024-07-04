# Stock Trader Project

## Overview

The Stock Trader project leverages machine learning to identify the top 10 stocks to buy and sell over a one-week period. The project is structured into five main scripts, each handling a crucial part of the data processing and analysis workflow. The final output provides actionable insights for traders by analyzing stock tickers, prices, news, and analyst opinions.

## Prerequisites

- Python 3.x
- Required Python packages (see `requirements.txt`)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/XitoAliferis/Stock-Trader.git
    cd Stock-Trader
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the scripts in the following order for a comprehensive analysis:

### 1. Get Stock Tickers

```bash
python getStockTickers.py
```
Description:
This script downloads the list of stock tickers from the NASDAQ and retrieves their market capitalization data using the Yahoo Finance API. It filters and stores the top 1000 tickers based on market cap to focus on the most significant stocks.

Key Functions:

- download_tickers(url): Downloads stock tickers from the provided URL.
- fetch_market_caps(tickers): Fetches market capitalization for each ticker using Yahoo Finance API.
- Saves the top 1000 tickers by market cap to data_files/top_1000_tickers_by_market_cap.txt.
### 2. Analyze Stock Prices
```bash
python analyseStockPrices.py
```
Description:
This script analyzes historical stock prices using technical indicators and prepares the data for machine learning models.

Key Steps:

-Data Collection: Downloads historical price data for each ticker using the Yahoo Finance API.
-Technical Indicators: Computes various technical indicators such as moving averages (MA), relative strength index (RSI), moving average convergence divergence (MACD), and more.
-Feature Engineering: Prepares features and targets for machine learning by combining technical indicators.
-Data Balancing: Balances the dataset using SMOTE (Synthetic Minority Over-sampling Technique) to address class imbalance.
-Normalization: Scales the features using StandardScaler.
-Saving Data: Saves the processed data to data_files/merged_data_with_indicators.csv.
Technical Indicators and Their Importance:

-Moving Averages (MA): Smoothens price data to identify trends. Common types include simple moving average (SMA) and exponential moving average (EMA).
-Relative Strength Index (RSI): Measures the magnitude of recent price changes to evaluate overbought or oversold conditions.
-Moving Average Convergence Divergence (MACD): Indicates trend direction and strength by comparing short-term and long-term EMAs.
-Bollinger Bands: Measures price volatility by plotting bands a certain number of standard deviations away from a moving average.
###3. Analyze Stock News
