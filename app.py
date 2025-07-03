import streamlit as st
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import requests

# -------------------------
# Caching Functions
# -------------------------
@st.cache_data
def load_books():
    df = pd.read_csv("books.csv")
    df = df[df["language_code"] == "eng"]
    df["combined"] = df["title"] + " by " + df["authors"]
    return df.reset_index(drop=True)

@st.cache_resource
def load_models():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    sentiment = pipeline("sentiment-analysis")
    return embed_model, sentiment

# -------------------------
# Fetch Google Book Info
# -------------------------
def fetch_summary_and_image(title, author):
    query = f"{title} {author}"
    url = f"https://www.googleapis.com/books/v1/volumes?q={query}"
    try:
        r = requests.get(url)
        data = r.json()
        info = data["items"][0]["volumeInfo"]
        return info.get("description", "No summary found."), info.get("imageLinks", {}).get("thumbnail", "")
    except:
        return "No summary found.", ""

# -------------------------
# Smart Search Logic
# -------------------------
def smartbook_search(query, top_k=5):
    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, book_embeddings)[0]
    top_results = torch.topk(scores, k=top_k)

    results = []
    for score, idx in zip(top_results[0], top_results[1]):
        idx = int(idx)
        row = books.iloc[idx]
        title, author = row["title"], row["authors"]
        rating = row["average_rating"]
        summary, image = fetch_summary_and_image(title, author)
        sentiment, confidence = sentiment_analyzer(summary)[0].values()

        results.append({
            "title": title,
            "author": author,
            "rating": rating,
            "score": float(score),
            "summary": summary,
            "image": image,
            "sentiment": sentiment,
            "confidence": confidence
        })
    return results

# -------------------------
# App Execution
# -------------------------
st.set_page_config(page_title="📚 SmartBook AI", layout="wide")
st.title("📚 SmartBook AI – Intelligent Book Recommender")
st.markdown("Semantic search + NLP-powered summaries and sentiment from Google Books and Transformers.")

books = load_books()
model, sentiment_analyzer = load_models()
book_embeddings = model.encode(books["combined"].tolist(), convert_to_tensor=True)

query = st.text_input("🔍 Enter the type of book you're looking for (e.g. 'mystery with strong female lead'):")

if st.button("Recommend") and query:
    with st.spinner("Finding the best matches for you..."):
        recommendations = smartbook_search(query)

    for book in recommendations:
        col1, col2 = st.columns([1, 4])
        with col1:
            if book["image"]:
                st.image(book["image"], width=110)
        with col2:
            st.subheader(book["title"])
            st.markdown(f"**Author:** {book['author']}")
            st.markdown(f"**Rating:** ⭐ {book['rating']}")
            st.markdown(f"**Similarity Score:** {book['score']:.4f}")
            st.markdown(f"**Sentiment:** {book['sentiment']} ({book['confidence']:.2f})")
            st.markdown(f"**Summary:** {book['summary']}")
            st.markdown("---")
