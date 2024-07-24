import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
import yfinance as yf
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from keras_tuner.tuners import BayesianOptimization
from keras_tuner import Objective
import ta  # Technical Analysis library
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.mixed_precision import set_global_policy, Policy
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import concurrent.futures
import joblib
from datetime import datetime, timedelta
import time
tf.config.optimizer.set_jit(True)


# Enable mixed precision training
policy = Policy('mixed_float16')
set_global_policy(policy)

# Helper functions
def fetch_ticker_data(ticker, start=None, end=None, max_period=False):
    ticker = str(ticker)
    stock = yf.Ticker(ticker)
    try:
        if max_period:
            hist = stock.history(period="5d")
        else:
            hist = stock.history(start=start, end=end)
        if not hist.empty:
            return ticker, hist
        else:
            print(f"Warning: No data fetched for ticker {ticker}")
            return ticker, None
    except Exception as e:
        print(f"Error fetching data for ticker {ticker}: {e}")
        return ticker, None

def fetch_historical_data(tickers, start=None, end=None, max_period=False):
    historical_data = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_ticker = {
            executor.submit(fetch_ticker_data, ticker, start, end, max_period): ticker
            for ticker in tickers
        }
        for future in tqdm(concurrent.futures.as_completed(future_to_ticker), total=len(tickers), desc="Fetching historical data"):
            ticker, data = future.result()
            if data is not None:
                historical_data[ticker] = data
    return historical_data

def preprocess_data(historical_data):
    processed_data = []
    for ticker, data in historical_data.items():
        data['Ticker'] = ticker
        processed_data.append(data)
    return pd.concat(processed_data)

def add_features(df):
    # Ensure there are enough data points to calculate the indicators
    min_data_points = 50  # Change this based on the longest window size you use
    if len(df) < min_data_points:
        print(f"Not enough data to calculate features for ticker {df['Ticker'].iloc[0]}. Required: {min_data_points}, Available: {len(df)}")
        return pd.DataFrame()  # Return empty DataFrame if not enough data

    df['RollingMean20'] = df['Close'].rolling(window=20).mean()
    df['RollingMean50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    df['MACD'] = ta.trend.MACD(df['Close']).macd_diff()
    df['BollingerHigh'] = ta.volatility.BollingerBands(df['Close']).bollinger_hband()
    df['BollingerLow'] = ta.volatility.BollingerBands(df['Close']).bollinger_lband()
    df['Stochastic'] = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close']).stoch()
    df['CCI'] = ta.trend.CCIIndicator(df['High'], df['Low'], df['Close']).cci()
    df['WilliamsR'] = ta.momentum.WilliamsRIndicator(df['High'], df['Low'], df['Close']).williams_r()
    df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
    df = df.dropna()
    return df

def clean_data(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    return df

def prepare_training_data(df):
    df['FutureReturn'] = df['Close'].pct_change(periods=5).shift(-5)  # Predict 1 week future return
    df = df.dropna(subset=['FutureReturn'])
    X = df[['RollingMean20', 'RollingMean50', 'RSI', 'MACD', 'BollingerHigh', 'BollingerLow', 'Stochastic', 'CCI', 'WilliamsR', 'ATR']].values
    y = df['FutureReturn'].apply(lambda x: 1 if x > 0 else 0).values  # Binary classification: Buy (1), Sell (0)
    tickers = df['Ticker'].values
    return X, y, tickers

# Example usage:
file_path = './data_files/tickers_sorted_by_market_cap.txt'
df_tickers = pd.read_csv(file_path)
tickers = df_tickers['Ticker'].tolist()



def incremental_training(tickers):
    scaler_X = joblib.load("./data_files/scaler_X.pkl")
    best_class_model = load_model("./data_files/best_class_model.h5")

    today = datetime.today()
    start_date = (today - pd.tseries.offsets.BDay(75)).strftime('%Y-%m-%d')
    end_date = today.strftime('%Y-%m-%d')

    # Load existing results
    existing_results = pd.read_csv("./data_files/results.csv")
    existing_tickers = existing_results['Ticker'].tolist()

    # Separate tickers into those in existing_tickers and those not in existing_tickers
    in_existing_tickers = [ticker for ticker in tickers if ticker in existing_tickers]
    not_in_existing_tickers = [ticker for ticker in tickers if ticker not in existing_tickers]

    new_historical_data = {}

    if in_existing_tickers:
        historical_data_existing = fetch_historical_data(in_existing_tickers, start=start_date, end=end_date)
        new_historical_data.update(historical_data_existing)
        
    if not_in_existing_tickers:
        historical_data_new = fetch_historical_data(not_in_existing_tickers, max_period=True)
        new_historical_data.update(historical_data_new)

    if not new_historical_data:
        print("No new historical data fetched. Exiting incremental training.")
        return

    new_processed_data = preprocess_data(new_historical_data)
    new_processed_data = new_processed_data.groupby('Ticker').apply(add_features).reset_index(drop=True)
    new_processed_data = clean_data(new_processed_data)

    if new_processed_data.empty:
        print("No new processed data available after preprocessing. Exiting incremental training.")
        return

    # Removing duplicates
    existing_processed_data = preprocess_data(fetch_historical_data(existing_tickers, max_period=True))
    combined_processed_data = pd.concat([existing_processed_data, new_processed_data]).drop_duplicates(subset=['Date', 'Ticker'], keep='last')

    X_new, y_new, tickers_array = prepare_training_data(combined_processed_data)

    if X_new.shape[0] == 0:
        print("No valid samples in the new data. Exiting incremental training.")
        return

    X_new_scaled = scaler_X.transform(X_new)

    smote = SMOTE(random_state=42)
    X_new_resampled, y_new_resampled = smote.fit_resample(X_new_scaled, y_new)

    early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True, mode='max', verbose=1)
    best_class_model.fit(X_new_resampled, y_new_resampled, epochs=10, validation_split=0.2, callbacks=[early_stopping])

    best_class_model.save("./data_files/best_class_model.h5")

    # Make new predictions
    class_predictions = best_class_model.predict(X_new_scaled).flatten()
    fpr, tpr, thresholds = roc_curve(y_new, class_predictions)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    predicted_labels = (class_predictions > optimal_threshold).astype(int)
    confidence_scores = class_predictions

    new_results = pd.DataFrame({
        'Ticker': tickers_array,
        'BuyHoldSell': predicted_labels,
        'Confidence': confidence_scores
    })
    new_results['BuyHoldSell'] = new_results['BuyHoldSell'].map({0: 'Sell', 1: 'Buy'})
    new_results = new_results.loc[new_results.groupby('Ticker')['Confidence'].idxmax()]

    # Combine existing results with new results
    combined_results = pd.concat([existing_results, new_results]).drop_duplicates(subset='Ticker', keep='last')

    combined_results.to_csv("./testdata_files/results.csv", index=False)

    top_buys = combined_results[combined_results['BuyHoldSell'] == 'Buy'].nlargest(10, 'Confidence')
    top_sells = combined_results[combined_results['BuyHoldSell'] == 'Sell'].nsmallest(10, 'Confidence')

    top_buys.to_csv("./testdata_files/top_buys.csv", index=False)
    top_sells.to_csv("./testdata_files/top_sells.csv", index=False)

    return combined_results


incremental_training(tickers)