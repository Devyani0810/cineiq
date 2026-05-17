import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# VADER is a ready-made sentiment tool — no training needed
# It reads text and returns a score from -1 (very negative) to +1 (very positive)
analyzer = SentimentIntensityAnalyzer()

def get_sentiment_score(text):
    # compound is the overall score: +1 = very positive, -1 = very negative
    score = analyzer.polarity_scores(str(text))["compound"]
    return score

# Since we don't have review text in MovieLens, we'll simulate sentiment
# using the average rating per movie as a stand-in
# (high average rating = positive sentiment)
def build_sentiment_scores():
    ratings = pd.read_csv("data/ratings.csv")
    movies = pd.read_csv("data/movies.csv")
    
    # Calculate average rating per movie
    avg_ratings = ratings.groupby("movieId")["rating"].mean()
    
    # Normalize to -1 to +1 scale (ratings are 0.5 to 5)
    # Formula: (rating - 2.75) / 2.25 maps the range to approx -1 to +1
    sentiment_scores = ((avg_ratings - 2.75) / 2.25).clip(-1, 1)
    
    # Merge with movie titles
    scores_df = movies[["movieId", "title"]].merge(
        sentiment_scores.rename("sentiment"),
        on="movieId"
    )
    
    print("Sample sentiment scores:")
    print(scores_df.head(10))
    
    return scores_df

# ── TEST IT ───────────────────────────────────────────────
scores = build_sentiment_scores()