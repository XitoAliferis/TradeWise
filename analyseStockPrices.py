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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from keras_tuner.tuners import BayesianOptimization
from keras_tuner import Objective
import ta  # Technical Analysis library
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.mixed_precision import set_global_policy, Policy
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

# Enable mixed precision training
policy = Policy('mixed_float16')
set_global_policy(policy)

# Load the stock list from your file
file_path = 'top_1000_tickers_by_market_cap.txt'
df_tickers = pd.read_csv(file_path)
tickers = df_tickers['Ticker'].tolist()  # Only take the first 2 tickers for now

# Fetch historical data for the past 5 years
def fetch_historical_data(tickers):
    historical_data = {}
    for ticker in tqdm(tickers, desc="Fetching historical data"):
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1mo")
        if not hist.empty:
            historical_data[ticker] = hist
        else:
            print(f"Warning: No data fetched for ticker {ticker}")
    return historical_data

historical_data = fetch_historical_data(tickers)

# Preprocess data
def preprocess_data(historical_data):
    processed_data = []
    for ticker, data in historical_data.items():
        data['Ticker'] = ticker
        processed_data.append(data)
    return pd.concat(processed_data)

processed_data = preprocess_data(historical_data)
print(f"Processed data for tickers: {processed_data['Ticker'].unique()}")

# Add more features
def add_features(df):
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

processed_data = add_features(processed_data)

# Check for and handle NaN, inf values
def clean_data(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    return df

processed_data = clean_data(processed_data)

# Prepare the data for training
def prepare_training_data(df):
    df['FutureReturn'] = df['Close'].pct_change(periods=5).shift(-5)  # Predict 1 week future return
    df = df.dropna(subset=['FutureReturn'])
    X = df[['RollingMean20', 'RollingMean50', 'RSI', 'MACD', 'BollingerHigh', 'BollingerLow', 'Stochastic', 'CCI', 'WilliamsR', 'ATR']].values
    y = df['FutureReturn'].apply(lambda x: 1 if x > 0 else 0).values  # Binary classification: Buy (1), Sell (0)
    tickers = df['Ticker'].values
    return X, y, tickers

X, y, tickers_array = prepare_training_data(processed_data)

# Analyze the distribution of classes
unique, counts = np.unique(y, return_counts=True)
plt.bar(unique, counts)
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.title('Class Distribution')
plt.show()

# Scale features
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

# Compute class weights to handle class imbalance
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weights = {i: class_weights[i] for i in range(len(class_weights))}

# Cross-validation setup
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accuracies = []

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True, mode='max', verbose=1)

# Build a simpler classification model
def build_simple_classification_model(hp):
    model = Sequential()
    model.add(Dense(units=hp.Int('units_1', min_value=32, max_value=128, step=32), activation='relu', input_shape=(X_scaled.shape[1],)))
    model.add(Dropout(rate=hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(Dense(units=hp.Int('units_2', min_value=16, max_value=64, step=16), activation='relu'))
    model.add(Dropout(rate=hp.Float('dropout_2', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification: Buy/Sell
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Hyperparameter tuning for classification model
tuner_class = BayesianOptimization(
    build_simple_classification_model,
    objective=Objective('val_accuracy', direction='max'),
    max_trials=10,  # Increased number of trials for a more thorough search
    executions_per_trial=1,
    directory='tuner',
    project_name='stock_classification',
    overwrite=True  # Overwrite existing project
)

# Cross-validation training
for train_index, test_index in kfold.split(X_scaled, y):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    tuner_class.search(X_train_resampled, y_train_resampled, epochs=10, validation_data=(X_test, y_test), callbacks=[early_stopping], class_weight=class_weights)
    best_class_model = tuner_class.get_best_models(num_models=1)[0]
    
    # Evaluate model
    scores = best_class_model.evaluate(X_test, y_test, verbose=0)
    accuracies.append(scores[1])

print(f"Cross-validation accuracies: {accuracies}")
print(f"Mean cross-validation accuracy: {np.mean(accuracies)}")

# Make predictions with confidence scores for classification model
class_predictions = best_class_model.predict(X_test).flatten()
fpr, tpr, thresholds = roc_curve(y_test, class_predictions)

# Find the optimal threshold
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

predicted_labels = (class_predictions > optimal_threshold).astype(int)
confidence_scores = class_predictions

results = pd.DataFrame({
    'Ticker': tickers_array[test_index],
    'BuyHoldSell': predicted_labels,
    'Confidence': confidence_scores
})

# Map the buy hold sell integer values to strings
results['BuyHoldSell'] = results['BuyHoldSell'].map({0: 'Sell', 1: 'Buy'})

# Keep only the highest confidence prediction for each ticker
results = results.loc[results.groupby('Ticker')['Confidence'].idxmax()]

# Ensure all tickers are in the results
expected_tickers = set(tickers)
missing_tickers = expected_tickers - set(results['Ticker'])
if missing_tickers:
    print(f"Missing tickers in results: {missing_tickers}")

print(results)

df = pd.DataFrame(results, columns=["Ticker", "BuyHoldSell", "Confidence"])
df.to_csv("results.csv", index=False)  # Save as CSV instead of text
