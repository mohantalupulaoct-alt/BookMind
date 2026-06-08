"""
search.py — Book Recommender Pipeline (ChromaDB-free)
Uses numpy cosine similarity — works on any Python version.
"""
import streamlit as st
import os
import json
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq

BASE        = os.path.dirname(__file__)
DATA_FILE   = os.path.join(BASE, "books_data.json")
EMBED_FILE  = os.path.join(BASE, "books_embeddings.npy")
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL   = "llama-3.1-8b-instant"
N_RESULTS   = 8

print("Loading book data...")
with open(DATA_FILE, "r", encoding="utf-8") as f:
    BOOKS = json.load(f)

print("Loading embeddings...")
EMBEDDINGS = np.load(EMBED_FILE).astype("float32")
norms = np.linalg.norm(EMBEDDINGS, axis=1, keepdims=True)
EMBEDDINGS_NORMED = EMBEDDINGS / (norms + 1e-9)
print(f"Ready — {len(BOOKS)} books loaded")

embedder = SentenceTransformer(EMBED_MODEL)
llm = Groq(api_key="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")

CATEGORY_MAP = {
    "biography":"Biography & Autobiography","autobiography":"Biography & Autobiography",
    "fiction":"Fiction","religion":"Religion","self-help":"Self-Help","self help":"Self-Help",
    "horror":"Horror","history":"History","science":"Science","romance":"Romance",
    "mystery":"Detective","thriller":"Thriller","fantasy":"Fantasy",
    "children":"Juvenile Fiction","juvenile":"Juvenile Fiction","cooking":"Cooking",
    "travel":"Travel","business":"Business & Economics","economics":"Business & Economics",
    "psychology":"Psychology","philosophy":"Philosophy","poetry":"Poetry",
    "comics":"Comics & Graphic Novels","graphic novel":"Comics & Graphic Novels",
    "health":"Health & Fitness","fitness":"Health & Fitness","nature":"Nature",
    "mathematics":"Mathematics","math":"Mathematics","music":"Music","art":"Art",
    "political":"Political Science","politics":"Political Science",
    "social science":"Social Science","true crime":"True Crime",
    "young adult":"Young Adult Fiction","humor":"Humor","sports":"Sports & Recreation",
    "technology":"Technology & Engineering","computers":"Computers","law":"Law",
    "medical":"Medical","education":"Education","family":"Family & Relationships",
    "relationships":"Family & Relationships","crafts":"Crafts & Hobbies",
    "hobbies":"Crafts & Hobbies","games":"Games & Activities",
    "body mind spirit":"Body, Mind & Spirit","spirituality":"Body, Mind & Spirit",
}

EXTRACT_PROMPT = """You are a search filter extractor for a book recommendation system.
Return ONLY a valid JSON object. No explanation. No markdown. Just JSON.

Format:
{
  "author": "exact author name or null",
  "category": "genre/category or null",
  "min_rating": number or null,
  "semantic_query": "8 to 10 specific descriptive words about content, themes, emotions of books the user wants",
  "user_query": "the original user query"
}

Rules for semantic_query:
- Write 8-10 words describing actual CONTENT and FEEL of books
- Never write "mood and vibe for X genre"
- Be specific to the user's actual request

Examples:
User: "biography of a historical leader"
Output: {"author":null,"category":"biography","min_rating":null,"semantic_query":"inspiring leadership historical figure political journey legacy courage perseverance","user_query":"biography of a historical leader"}

User: "feeling anxious want something calming"
Output: {"author":null,"category":null,"min_rating":null,"semantic_query":"peaceful mindfulness inner calm healing anxiety relief perspective self discovery","user_query":"feeling anxious want something calming"}
"""

def extract_filters(user_query: str) -> dict:
    response = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": EXTRACT_PROMPT},
            {"role": "user",   "content": f"Extract filters from: {user_query}"}
        ],
        temperature=0,
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"```json|```", "", raw, flags=re.IGNORECASE).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end != -1:
        raw = raw[start:end+1]
    try:
        return json.loads(raw)
    except:
        return {"author":None,"category":None,"min_rating":None,
                "semantic_query":user_query,"user_query":user_query}

def _normalize_category(raw):
    if not raw: return None
    return CATEGORY_MAP.get(raw.lower().strip(), raw)

def hybrid_search(filters: dict) -> list[dict]:
    semantic_query = filters.get("semantic_query") or filters.get("user_query") or "good book"
    q_vec = embedder.encode([semantic_query]).astype("float32")
    q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-9)
    scores = (EMBEDDINGS_NORMED @ q_norm.T).flatten()

    author_f   = filters.get("author")
    category_f = _normalize_category(filters.get("category"))
    rating_f   = float(filters["min_rating"]) if filters.get("min_rating") else None

    candidates = []
    for i, book in enumerate(BOOKS):
        if author_f and author_f.lower() not in book["authors"].lower(): continue
        if category_f and category_f not in book["categories"]: continue
        if rating_f and (book["rating"] or 0) < rating_f: continue
        candidates.append((i, float(scores[i])))

    if not candidates:
        candidates = [(i, float(scores[i])) for i in range(len(BOOKS))]

    candidates.sort(key=lambda x: x[1], reverse=True)
    return [{**BOOKS[i], "similarity": round(score, 3)} for i, score in candidates[:N_RESULTS]]

RECOMMEND_PROMPT = """You are a warm, knowledgeable book recommender like a well-read friend.
Pick the TOP 3 books from the list and write 2-3 sentences for each explaining personally why it fits.
Be conversational, not robotic. Format each with the book title bolded.
CRITICAL: Only recommend books from the list. Never invent books."""

def generate_recommendation(user_query: str, books: list[dict]) -> str:
    books_text = "\n".join([
        f"{i+1}. \"{b['title']}\" by {b['authors']} | Category: {b['categories']} | Rating: {b['rating']}"
        for i, b in enumerate(books)
    ])
    response = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": RECOMMEND_PROMPT},
            {"role": "user", "content": f"I'm looking for: \"{user_query}\"\n\nRetrieved books:\n{books_text}\n\nWrite a warm personal recommendation for the top 3."}
        ],
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()

def recommend(user_query: str) -> dict:
    filters        = extract_filters(user_query)
    books          = hybrid_search(filters)
    recommendation = generate_recommendation(user_query, books)
    return {"filters": filters, "books": books, "recommendation": recommendation}
