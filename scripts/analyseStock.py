import pandas as pd
import numpy as np
import analyseStockPrices

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler

analyseStockPrices

# Load the data from the three different analyses
analyst_data = pd.read_csv('./data_files/analyst_ratings.csv')
news_data = pd.read_csv('./data_files/news_ratings.csv')
price_data = pd.read_csv('./data_files/results.csv')

# Merge the data on Ticker
merged_data = pd.merge(analyst_data, news_data, on='Ticker', suffixes=('_analyst', '_news'))
merged_data = pd.merge(merged_data, price_data, on='Ticker')

# Prepare the data for the model
features = merged_data[['SentimentScore_analyst', 'SentimentScore_news', 'Confidence']]
targets = merged_data['BuyHoldSell']  # This should be a column that combines Buy/Hold/Sell into numerical values

# Convert targets to numerical values
target_mapping = {'Buy': 2, 'Hold': 1, 'Sell': 0}
targets = targets.map(target_mapping)

# Convert targets to categorical
targets_categorical = to_categorical(targets, num_classes=3)

# Scale the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Build the model
model = Sequential()
model.add(Dense(64, input_dim=features_scaled.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))  # Softmax activation for multi-class classification

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(features_scaled, targets, epochs=50, validation_split=0.2)

# Make predictions
predictions = model.predict(features_scaled)

# Convert predictions to labels
predictions_labels = ['Buy' if np.argmax(pred) == 2 else 'Hold' if np.argmax(pred) == 1 else 'Sell' for pred in predictions]

# Add predictions and confidence scores to the merged data
merged_data['final_prediction'] = predictions_labels
merged_data['prediction_confidence'] = predictions.max(axis=1)

# Output the final predictions
print("\nFinal Predictions:")
print(merged_data[['Ticker', 'final_prediction', 'prediction_confidence']])

# Save the final predictions to a CSV file
merged_data[['Ticker', 'final_prediction', 'prediction_confidence']].to_csv('./data_files/final_stock_ratings_new.csv', index=False)

# Filter top 10 best and worst stocks
top_10_best = merged_data[merged_data['final_prediction'] == 'Buy'].nlargest(10, 'prediction_confidence')
top_10_worst = merged_data[merged_data['final_prediction'] == 'Sell'].nlargest(10, 'prediction_confidence')

# Output the top 10 best and worst stocks
print("\nTop 10 Best Stocks:")
print(top_10_best[['Ticker', 'final_prediction', 'prediction_confidence']])

print("\nTop 10 Worst Stocks:")
print(top_10_worst[['Ticker', 'final_prediction', 'prediction_confidence']])

# Save the top 10 best and worst stocks to CSV files
top_10_best[['Ticker', 'final_prediction', 'prediction_confidence']].to_csv('./data_files/top_10_best_stocks.csv', index=False)
top_10_worst[['Ticker', 'final_prediction', 'prediction_confidence']].to_csv('./data_files/top_10_worst_stocks.csv', index=False)