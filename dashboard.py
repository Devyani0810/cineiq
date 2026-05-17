import streamlit as st
import requests
import plotly.express as px
import pandas as pd

# ── PAGE CONFIG ────────────────────────────────────────────
st.set_page_config(
    page_title="CineIQ",
    page_icon="🎬",
    layout="wide"
)

# ── TITLE ──────────────────────────────────────────────────
st.title("🎬 CineIQ — Movie Recommendation Engine")
st.markdown("*Where ambition meets cinema*")
st.divider()

# ── LOAD DATA FOR CHARTS ───────────────────────────────────
@st.cache_data  # cache_data means it loads once and remembers — saves time
def load_data():
    ratings = pd.read_csv("data/ratings.csv")
    movies = pd.read_csv("data/movies.csv")
    return ratings, movies

ratings, movies = load_data()

# ── TWO TABS ───────────────────────────────────────────────
# Tab 1 = recommend for a user, Tab 2 = find similar movies
tab1, tab2 = st.tabs(["👤 Recommend for User", "🎥 Find Similar Movies"])

# ══════════════════════════════════════════════════════════
# TAB 1 — USER RECOMMENDATIONS
# ══════════════════════════════════════════════════════════
with tab1:
    st.subheader("Get personalised recommendations")
    
    # Number input for user ID
    user_id = st.number_input(
        "Enter User ID (1 to 610):",
        min_value=1,
        max_value=610,
        value=1
    )
    
    if st.button("Get Recommendations", type="primary"):
        # Call our FastAPI backend
        response = requests.get(f"http://127.0.0.1:8000/recommend?user_id={user_id}")
        data = response.json()
        
        if "error" in data:
            st.error(data["error"])
        else:
            recs = data["recommendations"]
            
            st.success(f"Top {len(recs)} recommendations for User {user_id}")
            
            # Show recommendations as cards
            for i, movie in enumerate(recs):
                col1, col2, col3 = st.columns([3, 1, 3])
                with col1:
                    st.write(f"**{i+1}. {movie['title']}**")
                with col2:
                    score = movie['sentiment_score']
                    if score > 0.3:
                        st.success(f"⭐ {score}")
                    else:
                        st.info(f"⭐ {score}")
                with col3:
                    st.caption(movie['reason'])
            
            st.divider()
            
            # ── GENRE CHART ────────────────────────────────
            st.subheader("Your taste profile")
            
            # Find what this user has already rated
            user_ratings = ratings[ratings["userId"] == user_id]
            user_movies = user_ratings.merge(movies, on="movieId")
            
            # Split genres and count them
            all_genres = user_movies["genres"].str.split("|").explode()
            genre_counts = all_genres.value_counts().reset_index()
            genre_counts.columns = ["Genre", "Count"]
            
            # Draw a bar chart
            fig = px.bar(
                genre_counts,
                x="Genre",
                y="Count",
                title=f"Genres User {user_id} watches most",
                color="Count",
                color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════
# TAB 2 — SIMILAR MOVIES
# ══════════════════════════════════════════════════════════
with tab2:
    st.subheader("Find movies similar to one you love")
    
    # Get all movie titles for the dropdown
    all_titles = movies["title"].sort_values().tolist()
    
    selected_movie = st.selectbox("Pick a movie:", all_titles)
    
    if st.button("Find Similar Movies", type="primary"):
        response = requests.get(
            f"http://127.0.0.1:8000/similar",
            params={"title": selected_movie}
        )
        data = response.json()
        
        if "error" in data:
            st.error(data["error"])
        else:
            results = data["results"]
            st.success(f"Movies similar to **{selected_movie}**")
            
            # Show as a table
            df = pd.DataFrame(results)
            df.columns = ["Title", "Sentiment Score"]
            df.index = range(1, len(df)+1)
            st.dataframe(df, use_container_width=True)
            
            # Sentiment bar chart
            fig2 = px.bar(
                df,
                x="Title",
                y="Sentiment Score",
                title="Audience sentiment for similar movies",
                color="Sentiment Score",
                color_continuous_scale="RdYlGn"
            )
            fig2.update_xaxes(tickangle=45)
            st.plotly_chart(fig2, use_container_width=True)