# CineIQ 🎬
An explainable movie recommendation engine built with ML.

## Features
- Content-based filtering using TF-IDF + cosine similarity
- Collaborative filtering using SVD matrix factorization
- Sentiment-aware re-ranking using VADER
- FastAPI backend with /recommend and /similar endpoints
- Streamlit dashboard with genre taste profile charts

## How to run
```bash
pip3 install -r requirements.txt
uvicorn main:app --reload        # Terminal 1
streamlit run dashboard.py       # Terminal 2
```
