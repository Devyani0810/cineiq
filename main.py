from fastapi import FastAPI
from recommend import get_similar_movies, get_recommendations_for_user, movies
from sentiment import build_sentiment_scores

app = FastAPI()

# Load sentiment scores once when the server starts
sentiment_df = build_sentiment_scores()

# Convert to a dict for fast lookup: movieId -> sentiment score
sentiment_dict = dict(zip(sentiment_df["movieId"], sentiment_df["sentiment"]))

# ── HELPER: blend SVD score with sentiment ─────────────────
def get_movie_id(title):
    match = movies[movies["title"] == title]
    if match.empty:
        return None
    return match.iloc[0]["movieId"]

# ── ROUTE 1: Homepage ──────────────────────────────────────
# When someone visits the API, show a welcome message
@app.get("/")
def home():
    return {"message": "Welcome to CineIQ API!"}

# ── ROUTE 2: Recommend movies for a user ──────────────────
# Visit /recommend?user_id=1 to get recommendations for user 1
@app.get("/recommend")
def recommend(user_id: int):
    titles = get_recommendations_for_user(user_id)
    
    if isinstance(titles, str):  # error message
        return {"error": titles}
    
    # Add sentiment score and reason to each recommendation
    results = []
    for title in titles:
        movie_id = get_movie_id(title)
        sentiment = sentiment_dict.get(movie_id, 0)
        
        # Human readable reason
        if sentiment > 0.3:
            reason = "Highly rated by audiences with similar taste"
        elif sentiment > 0:
            reason = "Recommended based on your viewing patterns"
        else:
            reason = "Matches your taste profile"
        
        results.append({
            "title": title,
            "sentiment_score": round(sentiment, 2),
            "reason": reason
        })
    
    # Sort by sentiment score so best movies come first
    results = sorted(results, key=lambda x: x["sentiment_score"], reverse=True)
    return {"user_id": user_id, "recommendations": results}

# ── ROUTE 3: Find similar movies ──────────────────────────
# Visit /similar?title=Toy Story (1995) to find similar movies
@app.get("/similar")
def similar(title: str):
    results = get_similar_movies(title)
    
    if isinstance(results, str):  # error message
        return {"error": results}
    
    output = []
    for t in results:
        movie_id = get_movie_id(t)
        sentiment = sentiment_dict.get(movie_id, 0)
        output.append({
            "title": t,
            "sentiment_score": round(sentiment, 2)
        })
    
    return {"similar_to": title, "results": output}