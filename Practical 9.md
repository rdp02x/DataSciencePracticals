> [!NOTE] Question:
> Sentiment analysis, also known as opinion mining, is the process of determining the emotional tone behind a series of words. It is a common NLP task used to identify the sentiment of texts like reviews, social media posts, or news articles. Below is a simple Python program that performs sentiment analysis using the NLTK library and the VADER sentiment analysis tool.

```python
# Import necessary libraries
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download necessary NLTK resources
nltk.download('vader_lexicon')

# Initialize the SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Example texts for sentiment analysis
texts = [
    "I love this product! It's absolutely amazing and worth every penny.",
    "This is the worst experience I've ever had. I will never come back.",
    "The movie was okay, not great but not terrible either.",
    "I am very happy with the service provided, totally satisfied.",
    "I'm disappointed. The quality is not what I expected.",
    "The book is interesting, but it has some dull moments.",
    "The support team is helpful and resolved my issue quickly."
]

# Function to analyze sentiment
def analyze_sentiment(text):
    sentiment = sia.polarity_scores(text)
    overall_sentiment = 'Positive' if sentiment['compound'] > 0 else 'Negative' if sentiment['compound'] < 0 else 'Neutral'
    return sentiment, overall_sentiment

# Perform sentiment analysis on each text
for text in texts:
    print(f"Text: {text}")
    sentiment_scores, overall_sentiment = analyze_sentiment(text)
    print(f"Sentiment Scores: {sentiment_scores}")
    print(f"Overall Sentiment: {overall_sentiment}\n")
```
