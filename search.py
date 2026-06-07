"""
search.py — Book Recommender Pipeline
All search logic lives here. app.py only imports from this file.
"""
import os
import json
import re
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq

CHROMA_PATH = os.path.join(os.path.dirname(__file__), "bukemd_db")
COLLECTION  = "bukemd"
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL   = "llama-3.1-8b-instant"
N_RESULTS   = 8


GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
embedder   = SentenceTransformer(EMBED_MODEL)
chroma     = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma.get_collection(COLLECTION)
llm = Groq(api_key=GROQ_API_KEY)

# ── Category map ───────────────────────────────────────────────────────────────
# Maps what Llama returns → actual substring present in your DB categories

CATEGORY_MAP = {
    "biography":          "Biography & Autobiography",
    "autobiography":      "Biography & Autobiography",
    "fiction":            "Fiction",
    "religion":           "Religion",
    "self-help":          "Self-Help",
    "self help":          "Self-Help",
    "horror":             "Horror",
    "history":            "History",
    "science":            "Science",
    "romance":            "Romance",
    "mystery":            "Detective",
    "thriller":           "Thriller",
    "fantasy":            "Fantasy",
    "children":           "Juvenile Fiction",
    "juvenile":           "Juvenile Fiction",
    "cooking":            "Cooking",
    "travel":             "Travel",
    "business":           "Business & Economics",
    "economics":          "Business & Economics",
    "psychology":         "Psychology",
    "philosophy":         "Philosophy",
    "poetry":             "Poetry",
    "comics":             "Comics & Graphic Novels",
    "graphic novel":      "Comics & Graphic Novels",
    "health":             "Health & Fitness",
    "fitness":            "Health & Fitness",
    "nature":             "Nature",
    "mathematics":        "Mathematics",
    "math":               "Mathematics",
    "music":              "Music",
    "art":                "Art",
    "political":          "Political Science",
    "politics":           "Political Science",
    "social science":     "Social Science",
    "true crime":         "True Crime",
    "young adult":        "Young Adult Fiction",
    "humor":              "Humor",
    "sports":             "Sports & Recreation",
    "technology":         "Technology & Engineering",
    "computers":          "Computers",
    "law":                "Law",
    "medical":            "Medical",
    "education":          "Education",
    "family":             "Family & Relationships",
    "relationships":      "Family & Relationships",
    "crafts":             "Crafts & Hobbies",
    "hobbies":            "Crafts & Hobbies",
    "games":              "Games & Activities",
    "body mind spirit":   "Body, Mind & Spirit",
    "spirituality":       "Body, Mind & Spirit",
}

# ── Step 1: Extract filters ────────────────────────────────────────────────────

EXTRACT_PROMPT = """You are a search filter extractor for a book recommendation system.
Return ONLY a valid JSON object with exactly these keys: author, category, min_rating, semantic_query, user_query.
No explanation. No markdown. No additional text.

Format:
{
  "author": "exact author name or null",
  "category": "genre/category or null",
  "min_rating": number or null,
  "semantic_query": "8 to 10 specific descriptive words about the content, themes, emotions, and subject matter of books the user wants",
  "user_query": "the original user query"
}

Rules for semantic_query:
- Write 8 to 10 words describing actual CONTENT and FEEL of the books
- Think: what words appear in the description of a perfect book for this user?
- Never write meta-phrases like "mood and vibe for X genre"
- Always make it specific to the user's actual request

Examples:
User: "suggest a biography of a historical leader"
Output: {"author": null, "category": "biography", "min_rating": null, "semantic_query": "inspiring leadership historical figure political journey legacy courage perseverance nation", "user_query": "suggest a biography of a historical leader"}
"""

def _extract_json_object(raw: str) -> str:
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        return raw[start:end+1]
    return raw


def extract_filters(user_query: str) -> dict:
    """Step 1 — LLM parses natural language into structured filters."""
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
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        candidate = _extract_json_object(raw)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            raise ValueError(
                "Failed to parse filter JSON from LLM response.\n"
                f"Raw response:\n{raw}\n\n"
                f"Candidate JSON:\n{candidate}\n\n"
                f"Original error: {exc}"
            ) from exc


# ── Step 2: Build where clause ────────────────────────────────────────────────

def _normalize_category(raw: str) -> str | None:
    if not raw:
        return None
    return CATEGORY_MAP.get(raw.lower().strip(), raw)

def _build_where(filters: dict) -> dict | None:
    conditions = []

    if filters.get("author"):
        conditions.append({"authors": {"$contains": filters["author"]}})

    if filters.get("category"):
        norm = _normalize_category(filters["category"])
        if norm:
            conditions.append({"categories": {"$contains": norm}})

    if filters.get("min_rating"):
        conditions.append({"rating": {"$gte": float(filters["min_rating"])}})

    if not conditions:    return None
    if len(conditions) == 1: return conditions[0]
    return {"$and": conditions}


# ── Step 3: Hybrid search ─────────────────────────────────────────────────────

def _encode_text(text: str) -> list[float]:
    raw_embedding = embedder.encode([text])
    if hasattr(raw_embedding, "tolist"):
        raw_embedding = raw_embedding.tolist()
    return [float(x) for x in raw_embedding[0]]


def _combine_query_embeddings(query_a: str, query_b: str) -> list[float]:
    if not query_a:
        return _encode_text(query_b)
    if not query_b:
        return _encode_text(query_a)

    embeddings = embedder.encode([query_a, query_b])
    if hasattr(embeddings, "tolist"):
        embeddings = embeddings.tolist()

    emb_a, emb_b = embeddings[0], embeddings[1]
    avg_embedding = [float((float(a) + float(b)) / 2) for a, b in zip(emb_a, emb_b)]
    return avg_embedding


def hybrid_search(filters: dict) -> list[dict]:
    """Step 2 — Vector search + metadata filters in one ChromaDB call."""

    semantic_query = filters.get("semantic_query")
    user_query = filters.get("user_query")
    if semantic_query and user_query:
        query_embedding = [_combine_query_embeddings(semantic_query, user_query)]
    else:
        query_embedding = [_encode_text(semantic_query or user_query or "good book")]

    where_clause = _build_where(filters)

    kwargs = {
        "query_embeddings": query_embedding,
        "n_results":        N_RESULTS,
        "include":          ["metadatas", "distances"],
    }
    if where_clause:
        kwargs["where"] = where_clause

    results = collection.query(**kwargs)
    if results['ids']!=[[]]:
        pass
    else:
        del kwargs["where"]
        results = collection.query(**kwargs)

    books = []
    for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
        books.append({
            "title":      meta.get("title",      "Unknown"),
            "authors":    meta.get("authors",    "Unknown"),
            "categories": meta.get("categories", "Unknown"),
            "rating":     meta.get("rating",     None),
            "image":      meta.get("image",      ""),
            "similarity": round(max(0.0, 1 - dist), 3),
        })
    return books


# ── Step 4: Generate recommendation ──────────────────────────────────────────

RECOMMEND_PROMPT = """You are a warm, knowledgeable book recommender like a well-read friend.

You will receive the user's request and a list of retrieved books.
Pick the TOP 3 books that best match and write 2-3 sentences for each explaining
personally why it fits — using the user's own words and mood.
Be conversational, not robotic. Format each with the book title bolded.
Examples:
"American Women in Mission: The Modern Mission Era 1792-1992" by Dana Lee Robert could be a great companion to your search. While it's not specifically focused on the Missionary Sisters of St. Columban in China, it might provide some broader context and insights into the history of women's missions in America.
CRITICAL RULE: You may ONLY recommend books from the list provided to you.
Never recommend, mention, or invent any book not in the list.
If the list has poor matches, be honest and pick the closest ones anyway."""

def generate_recommendation(user_query: str, books: list[dict]) -> str:
    """Step 3 — LLM writes personalized recommendation from retrieved books."""

    books_text = "\n".join([
        f"{i+1}. \"{b['title']}\" by {b['authors']} | "
        f"Category: {b['categories']} | Rating: {b['rating']} | Match: {b['similarity']}"
        for i, b in enumerate(books)
    ])

    response = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": RECOMMEND_PROMPT},
            {"role": "user",   "content": (
                f"I'm looking for: \"{user_query}\"\n\n"
                f"Retrieved books:\n{books_text}\n\n"
                f"Write a warm personal recommendation for the top 3."
            )}
        ],
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()


# ── Full pipeline (called by app.py) ──────────────────────────────────────────

def recommend(user_query: str) -> dict:
    """
    Single entry point for app.py.
    Returns: { books: list, recommendation: str, filters: dict }
    """
    filters        = extract_filters(user_query)
    books          = hybrid_search(filters)
    recommendation = generate_recommendation(user_query, books)
    return {
        "filters":        filters,
        "books":          books,
        "recommendation": recommendation,
    }

 