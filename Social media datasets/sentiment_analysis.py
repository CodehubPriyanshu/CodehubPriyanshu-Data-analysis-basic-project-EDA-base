import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import numpy as np

# Download necessary NLTK data
nltk.download('vader_lexicon')

# Read the CSV file
df = pd.read_csv('Social-Media.csv')

# Display basic information about the dataset
print(df.info())

# Display the first few rows
print(df.head())

# Perform sentiment analysis
sia = SentimentIntensityAnalyzer()

# Function to get sentiment scores
def get_sentiment(text):
    if isinstance(text, str):
        return sia.polarity_scores(text)['compound']
    else:
        return np.nan

# Apply sentiment analysis to the 'text' column
df['sentiment_score'] = df['text'].apply(get_sentiment)

# Remove rows with NaN sentiment scores
df = df.dropna(subset=['sentiment_score'])

# Categorize sentiment
df['sentiment_category'] = pd.cut(df['sentiment_score'],
                                  bins=[-1, -0.1, 0.1, 1],
                                  labels=['Negative', 'Neutral', 'Positive'])

# Display sentiment distribution
print(df['sentiment_category'].value_counts(normalize=True))

# Visualize sentiment distribution
plt.figure(figsize=(10, 6))
df['sentiment_category'].value_counts().plot(kind='bar')
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

# Analyze sentiment by subreddit
subreddit_sentiment = df.groupby('subreddit')['sentiment_score'].mean().sort_values(ascending=False)
print("\nAverage Sentiment Score by Subreddit:")
print(subreddit_sentiment)

# Visualize sentiment by subreddit
plt.figure(figsize=(12, 6))
subreddit_sentiment.plot(kind='bar')
plt.title('Average Sentiment Score by Subreddit')
plt.xlabel('Subreddit')
plt.ylabel('Average Sentiment Score')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Analyze correlation between sentiment and other numerical features
numerical_columns = ['social_karma', 'social_num_comments', 'lex_liwc_WC', 'lex_liwc_Analytic', 'lex_liwc_Clout', 'lex_liwc_Authentic', 'lex_liwc_Tone']
correlation = df[numerical_columns + ['sentiment_score']].corr()['sentiment_score'].sort_values(ascending=False)
print("\nCorrelation between Sentiment Score and Other Features:")
print(correlation)

# Visualize correlation
plt.figure(figsize=(10, 6))
correlation.plot(kind='bar')
plt.title('Correlation between Sentiment Score and Other Features')
plt.xlabel('Features')
plt.ylabel('Correlation Coefficient')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

print("\nAnalysis Complete!")