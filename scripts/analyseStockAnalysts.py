import pandas as pd
import yfinance as yf
import os
import getStockTickers

getStockTickers
# Function to fetch analyst recommendations using yfinance
def fetch_analyst_recommendations(ticker):
    try:
        stock = yf.Ticker(ticker)
        recommendations = stock.recommendations_summary
        if recommendations is not None:
            latest_recommendations = recommendations.tail(20)  # Get the latest 20 recommendations
            return latest_recommendations
        else:
            return None
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

# Function to calculate the sentiment score based on numerical recommendations
def calculate_sentiment_score(recommendations):
    weights = {
        'strongBuy': 2.0,
        'buy': 1.0,
        'hold': 0.0,
        'sell': -1.0,
        'strongSell': -2.0
    }
    score = 0
    total_recommendations = 0
    
    for column, weight in weights.items():
        if column in recommendations:
            count = recommendations[column].sum()
            score += count * weight
            total_recommendations += count
    
    if total_recommendations == 0:
        return 0  # No recommendations available to calculate score
    
    return score / total_recommendations

# Function to generate a buy, hold, or sell rating based on sentiment score
def generate_rating(sentiment_score):
    if sentiment_score > 0.5:
        return 'Buy'
    elif sentiment_score < -0.5:
        return 'Sell'
    else:
        return 'Hold'

# Main function to process a single ticker
def main(ticker):
    print(f"\nProcessing {ticker}...")
    recommendations = fetch_analyst_recommendations(ticker)
    
    if recommendations is not None and not recommendations.empty:
        print(f"Recommendations DataFrame for {ticker}:\n{recommendations.head()}\n")
        
        sentiment_score = calculate_sentiment_score(recommendations)
        rating = generate_rating(sentiment_score)
        
        print(f'Sentiment score for {ticker}: {sentiment_score}')
        print(f'Generated rating for {ticker}: {rating}')
        
        return {"Ticker": ticker, "BuyHoldSell": rating, "SentimentScore": sentiment_score}
    else:
        print(f"No analyst recommendations found for {ticker}.")
        return None

if __name__ == '__main__':
    # Load the stock list from your file
    file_path = './data_files/tickers_sorted_by_market_cap.txt'
    df_tickers = pd.read_csv(file_path)
    tickers = df_tickers['Ticker'].tolist()   # Only take the first 2 tickers for now
    
    results = []
    for ticker in tickers:
        result = main(ticker)
        if result:
            results.append(result)
    output_dir = "./data_files"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    results_df = pd.DataFrame(results)
    results_df.to_csv('./data_files/analyst_ratings.csv', index=False)
