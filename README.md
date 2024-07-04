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
