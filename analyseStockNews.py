import feedparser
import pandas as pd
from textblob import TextBlob
import aiohttp
import asyncio
from tqdm.asyncio import tqdm_asyncio

# Function to fetch Google News RSS feed using feedparser
def fetch_news_data_rss(ticker):
    rss_url = f'https://news.google.com/rss/search?q={ticker}&hl=en-US&gl=US&ceid=US:en'
    feed = feedparser.parse(rss_url)
    headlines = []
    summaries = []
    for entry in feed.entries:
        headlines.append(entry.title)
        summaries.append(entry.summary)
    return headlines, summaries

# Function to fetch Reddit data using aiohttp
async def fetch_reddit_data(session, ticker):
    url = f'https://www.reddit.com/r/stocks/search/?q={ticker}&restrict_sr=1&sort=new'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    retries = 3
    for attempt in range(retries):
        try:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    posts = []
                    for item in soup.find_all('div', {'class': 'search-result'}):
                        title = item.find('h3').get_text()
                        content = item.find('div', {'class': 'search-result-text'}).get_text()
                        posts.append(title + " " + content)
                    return posts[:20]
        except Exception as e:
            print(f"Error fetching Reddit data for {ticker}, attempt {attempt + 1}: {e}")
            await asyncio.sleep(1)
    return []

# Function to analyze sentiment using TextBlob
def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

# Function to calculate the sentiment score
def get_sentiment_score(data):
    if not data:
        return 0
    scores = [analyze_sentiment(text) for text in data]
    return sum(scores) / len(scores)

# Function to generate a buy, hold, or sell rating based on sentiment score
def generate_rating(sentiment_score):
    if sentiment_score > 0.1:
        return 'Buy'
    elif sentiment_score < -0.1:
        return 'Sell'
    else:
        return 'Hold'

# Main function to combine everything
async def main(ticker, session):
    headlines, summaries = fetch_news_data_rss(ticker)
    reddit_data = await fetch_reddit_data(session, ticker)
    
    all_data = headlines + summaries + reddit_data
    
    if not all_data:
        print(f"No data available for {ticker}.")
        return None
    
    sentiment_score = get_sentiment_score(all_data)
    rating = generate_rating(sentiment_score)
    
    print(f'Sentiment score for {ticker}: {sentiment_score}')
    print(f'Generated rating for {ticker}: {rating}')
    
    return {"Ticker": ticker, "BuyHoldSell": rating, "SentimentScore": sentiment_score}

# Function to handle all tickers
async def process_all_tickers(tickers):
    results = []
    async with aiohttp.ClientSession() as session:
        tasks = [main(ticker, session) for ticker in tickers]
        for task in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Processing tickers"):
            result = await task
            if result:
                results.append(result)
    return results

if __name__ == '__main__':
    # Load the stock list from your file
    file_path = 'top_1000_tickers_by_market_cap.txt'
    df_tickers = pd.read_csv(file_path)
    tickers = df_tickers['Ticker'].tolist()
    
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(process_all_tickers(tickers))
    
    results_df = pd.DataFrame(results)
    results_df.to_csv('news_ratings.csv', index=False)
