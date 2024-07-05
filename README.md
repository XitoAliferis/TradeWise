# TradeWise

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

 - Data Collection: Downloads historical price data for each ticker using the Yahoo Finance API.
 - Technical Indicators: Computes various technical indicators such as moving averages (MA), relative strength index (RSI), moving average convergence divergence (MACD), and more.
 - Feature Engineering: Prepares features and targets for machine learning by combining technical indicators.
 - Data Balancing: Balances the dataset using SMOTE (Synthetic Minority Over-sampling Technique) to address class imbalance.
 - Normalization: Scales the features using StandardScaler.
 - Saving Data: Saves the processed data to data_files/merged_data_with_indicators.csv.
 
Technical Indicators and Their Importance:

 - Moving Averages (MA): Smoothens price data to identify trends. Common types include simple moving average (SMA) and exponential moving average (EMA).
 - Relative Strength Index (RSI): Measures the magnitude of recent price changes to evaluate overbought or oversold conditions.
 - Moving Average Convergence Divergence (MACD): Indicates trend direction and strength by comparing short-term and long-term EMAs.
 - Bollinger Bands: Measures price volatility by plotting bands a certain number of standard deviations away from a moving average.
   
### 3. Analyze Stock News
```bash
python analyseStockNews.py
```
Description:
This script analyzes recent news articles related to the stocks to assess sentiment and its potential impact on stock prices.

Key Steps:
 - Data Collection: Downloads news articles related to the stocks using a news API.
 - Sentiment Analysis: Performs sentiment analysis on the articles using Natural Language Processing (NLP) techniques to classify the sentiment as positive, negative, or neutral.
 - Aggregation: Aggregates sentiment scores for each stock over a specified period.
 - Merging Data: Merges sentiment data with the stock price data.
 - Saving Data: Saves the data to data_files/news_sentiment.csv.
   
### 4. Analyze Stock Analysts
```bash
python analyseStockAnalysts.py
```
Description:
This script compiles analysts' opinions and recommendations for each stock, which are crucial for understanding market sentiment from financial experts.

Key Steps:
 - Data Collection: Downloads analysts' ratings and target prices from financial APIs.
 - Aggregation: Aggregates and averages the ratings and target prices for each stock.
 - Merging Data: Merges analysts' data with the existing dataset.
 - Saving Data: Saves the data to data_files/analyst_ratings.csv.

### 5. Analyze Stocks
```bash
python analyseStock.py
```
Description:
This script integrates all previous analyses to predict the best stocks to buy and sell using a machine learning model.

Key Steps:
 - Data Merging: Merges data from stock prices, news sentiment, and analysts' opinions.
 - Feature Scaling: Scales the features using StandardScaler.
 - Model Training: Trains a neural network model using TensorFlow. The model architecture includes dense layers with ReLU activation and dropout for regularization.
 - Model Evaluation: Evaluates the model using accuracy metrics and validation split.
 - Predictions: Makes predictions and computes confidence scores.
 - Output: Identifies and outputs the top 10 best and worst stocks based on prediction confidence.
 - Saving Results: Saves the final predictions to data_files/final_stock_ratings_new.csv.

Machine Learning Model:
- Neural Network: A sequential neural network model is used for multi-class classification (Buy, Hold, Sell).
  - Layers: The model includes multiple dense layers with ReLU activation and a softmax output layer for classification.
  - Optimizer: Adam optimizer is used for training.
  - Loss Function: Sparse categorical crossentropy is used as the loss function.
  - Regularization: Dropout layers are included to prevent overfitting.

### Goal
The primary goal of this project is to use comprehensive data analysis and machine learning techniques to identify the top 10 stocks for buying and selling over a one-week period, providing actionable insights for traders.

## Example Data
please refer to the files below
 - [worst stocks](data_files/top_10_worst_stocks.csv).
 - [best stocks](data_files/top_10_best_stocks.csv).
 - [history](https://docs.google.com/spreadsheets/d/1RSpStVAhkKcI8T9QOmz3OHmUel3v7_7ZYwCgVprjmGw/edit?usp=sharing)
### License
This project is licensed under the MIT License.

### Acknowledgements
Contributors: Xito Aliferis
Data Sources: Various financial and news APIs

