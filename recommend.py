import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import numpy as np

# ── LOAD DATA ──────────────────────────────────────────────
# pd.read_csv reads a CSV file and turns it into a table (called a DataFrame)
ratings = pd.read_csv("data/ratings.csv")
movies = pd.read_csv("data/movies.csv")

print("Ratings loaded:", ratings.shape)   # shape = (rows, columns)
print("Movies loaded:", movies.shape)
print(movies.head())                       # head() shows first 5 rows

# ── CONTENT BASED FILTER ───────────────────────────────────
# TF-IDF turns the genres text into numbers the computer can compare
# For example "Comedy|Romance" becomes a vector of numbers
tfidf = TfidfVectorizer(token_pattern=r"[^|]+")
tfidf_matrix = tfidf.fit_transform(movies["genres"])

# cosine_similarity compares every movie with every other movie
# Result is a 9742 x 9742 matrix of similarity scores (0 to 1)
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# This creates a quick lookup: movie title → index number
indices = pd.Series(movies.index, index=movies["title"]).drop_duplicates()

def get_similar_movies(title, n=10):
    # Find the index of the movie the user typed
    if title not in indices:
        return f"Movie '{title}' not found."
    
    idx = indices[title]
    
    # Get similarity scores for this movie vs all others
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort by similarity score, highest first
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Take top n results (skip index 0 because that's the movie itself)
    sim_scores = sim_scores[1:n+1]
    
    # Get the movie titles
    movie_indices = [i[0] for i in sim_scores]
    return movies["title"].iloc[movie_indices].tolist()

# ── TEST IT ────────────────────────────────────────────────
print("\nMovies similar to Toy Story:")
print(get_similar_movies("Toy Story (1995)"))

# ── COLLABORATIVE FILTER (SVD) ─────────────────────────────
# First we create a user-movie matrix
# Rows = users, Columns = movies, Values = ratings (0 means not rated)
user_movie_matrix = ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
).fillna(0)  # fillna(0) replaces empty ratings with 0

# TruncatedSVD breaks this giant matrix into patterns
# n_components=50 means we find 50 hidden taste patterns
svd = TruncatedSVD(n_components=50, random_state=42)
matrix_reduced = svd.fit_transform(user_movie_matrix)

# Rebuild the full predicted ratings matrix
predicted_ratings = np.dot(matrix_reduced, svd.components_)

# Turn it back into a DataFrame with proper labels
predicted_df = pd.DataFrame(
    predicted_ratings,
    index=user_movie_matrix.index,
    columns=user_movie_matrix.columns
)

def get_recommendations_for_user(user_id, n=10):
    if user_id not in predicted_df.index:
        return f"User {user_id} not found."
    
    # Get this user's predicted ratings for all movies
    user_ratings = predicted_df.loc[user_id]
    
    # Find movies this user has ALREADY rated (we don't want to recommend those)
    already_rated = ratings[ratings["userId"] == user_id]["movieId"].tolist()
    
    # Remove already rated movies and sort by predicted score
    recommendations = user_ratings.drop(index=already_rated, errors="ignore")
    recommendations = recommendations.sort_values(ascending=False).head(n)
    
    # Convert movieIds to titles
    recommended_ids = recommendations.index.tolist()
    recommended_titles = movies[movies["movieId"].isin(recommended_ids)]["title"].tolist()
    return recommended_titles

# ── TEST IT ────────────────────────────────────────────────
print("\nRecommendations for User 1:")
print(get_recommendations_for_user(1))