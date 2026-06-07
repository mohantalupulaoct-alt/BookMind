"""
app.py — BookMind UI v3
"""
import streamlit.components.v1 as components
import streamlit as st
from search import recommend, hybrid_search, generate_recommendation
import json, re

st.set_page_config(
    page_title="BookMind",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:wght@300;400;500&display=swap');
* { box-sizing: border-box; }
html, body, [data-testid="stApp"] {
    background: #0a0a0f !important;
    font-family: 'DM Sans', sans-serif;
}
.block-container { padding: 1.5rem 2rem !important; }
.bm-title { font-family: 'Playfair Display', serif; font-size: 1.8rem; color: #fff; margin: 0; }
.bm-sub { font-size: 0.8rem; color: #666; margin: 0 0 1.2rem; }
.stTextInput > div > div > input {
    background: #13131f !important; border: 1.5px solid #2a2a3e !important;
    border-radius: 10px !important; color: #fff !important;
    font-size: 1rem !important; padding: 0.7rem 1rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: #f59e0b !important;
    box-shadow: 0 0 0 2px rgba(245,158,11,0.15) !important;
}
[data-testid="stSidebar"] { background: #0d0d18 !important; border-right: 1px solid #1e1e2e; }
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem !important; }
.filter-title {
    font-size: 0.78rem; color: #f59e0b; letter-spacing: 0.1em;
    text-transform: uppercase; margin: 1.2rem 0 0.5rem;
    font-family: 'Playfair Display', serif;
}
.book-card {
    background: #13131f; border-radius: 12px; border: 1px solid #1e1e2e;
    overflow: hidden; transition: transform 0.2s, border-color 0.2s; margin-bottom: 1rem;
}
.book-card:hover { transform: translateY(-3px); border-color: #f59e0b44; }
.book-img-wrap {
    width: 100%; height: 175px; overflow: hidden; background: #1a1a2e;
    display: flex; align-items: center; justify-content: center;
}
.book-img-wrap img { width: 100%; height: 100%; object-fit: cover; }
.book-info { padding: 0.75rem; }
.book-title {
    font-weight: 600; font-size: 0.85rem; color: #fff; line-height: 1.3; margin-bottom: 0.25rem;
    display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;
}
.book-author { font-size: 0.73rem; color: #888; margin-bottom: 0.35rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.book-badge {
    display: inline-block; background: rgba(245,158,11,0.1); color: #f59e0b;
    border: 1px solid rgba(245,158,11,0.3); border-radius: 20px;
    padding: 1px 8px; font-size: 0.63rem; margin-bottom: 0.3rem; max-width: 100%;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.book-rating { color: #f59e0b; font-size: 0.72rem; }
.match-score {
    float: right; background: rgba(74,222,128,0.1); color: #4ade80;
    border: 1px solid rgba(74,222,128,0.3); border-radius: 20px; padding: 1px 7px; font-size: 0.63rem;
}
.filter-pill {
    display: inline-block; background: #1a1a2e; color: #818cf8;
    border: 1px solid #2e3a6e; border-radius: 20px;
    padding: 2px 10px; font-size: 0.72rem; margin-right: 4px; margin-bottom: 6px;
}
.results-meta { font-size: 0.82rem; color: #666; margin-bottom: 1rem; }
.results-meta span { color: #f59e0b; }
.empty-state { text-align: center; padding: 5rem 2rem; }
.stButton > button {
    background: linear-gradient(135deg, #f59e0b, #ef4444) !important;
    color: white !important; border: none !important; border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important; font-weight: 500 !important;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def stars(rating) -> str:
    if rating is None: return "No rating"
    filled = int(round(float(rating)))
    return "★" * filled + "☆" * (5 - filled) + f" {float(rating):.1f}"

def clean_cat(raw: str) -> str:
    if not raw: return "Unknown"
    return raw.strip("[]'\"").split("'")[0].strip()

def book_card_html(book: dict, match_pct: int) -> str:
    title  = book['title'].replace("'","&#39;").replace('"','&quot;')
    author = book['authors'].strip("[]'\"").split("'")[0].strip()
    cat    = clean_cat(book['categories'])[:30]
    img    = (f'<img src="{book["image"]}" onerror="this.style.display=\'none\'">'
              if book.get("image") else '<span style="font-size:2rem;color:#333">📖</span>')
    return f"""<div class="book-card">
      <div class="book-img-wrap">{img}</div>
      <div class="book-info">
        <div class="book-title">{title}</div>
        <div class="book-author">{author}</div>
        <span class="book-badge">{cat}</span><br>
        <span class="book-rating">{stars(book['rating'])}</span>
        <span class="match-score">🎯 {match_pct}%</span>
      </div></div>"""

def normalize_match(books):
    if not books: return books
    dists = [1 - b.get("similarity", 0) for b in books]
    mn, mx = min(dists), max(dists)
    span = mx - mn if mx != mn else 1
    for b, d in zip(books, dists):
        b["match_pct"] = int((1 - (d - mn) / span) * 100)
    return books


# ── Session state ─────────────────────────────────────────────────────────────
for key, val in {
    "books": [], "recommendation": "", "rec_books": [],
    "filters": {}, "last_query": "", "panel_open": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ── Sidebar ───────────────────────────────────────────────────────────────────
GENRES = [
    "Fiction","History","Religion","Biography & Autobiography",
    "Business & Economics","Juvenile Fiction","Computers","Science",
    "Psychology","Philosophy","Music","Health & Fitness",
    "Sports & Recreation","Social Science","Family & Relationships",
    "Education","Body, Mind & Spirit"
]

with st.sidebar:
    st.markdown('<p style="font-family:Playfair Display,serif;font-size:1.3rem;color:#f59e0b;margin:0">📚 BookMind</p>', unsafe_allow_html=True)
    st.markdown('<p style="color:#555;font-size:0.75rem;margin-bottom:1rem">Filters</p>', unsafe_allow_html=True)

    st.markdown('<div class="filter-title">Genre</div>', unsafe_allow_html=True)
    selected_genre = st.selectbox("genre", ["All Genres"] + GENRES,
                                  label_visibility="collapsed", key="genre_select")

    st.markdown('<div class="filter-title">Author</div>', unsafe_allow_html=True)
    author_input = st.text_input("author", placeholder="e.g. Stephen King",
                                 label_visibility="collapsed", key="author_filter")

    st.markdown('<div class="filter-title">Min Rating</div>', unsafe_allow_html=True)
    min_rating = st.select_slider("rating", options=[1.0,2.0,3.0,4.0,5.0],
                                  value=1.0, label_visibility="collapsed", key="rating_slider")
    st.markdown(f'<span style="color:#f59e0b;font-size:0.78rem">{"★"*int(min_rating)} and above</span>',
                unsafe_allow_html=True)

    st.divider()

    if st.button("Apply Filters", use_container_width=True, key="apply_filters"):
        with st.spinner("Filtering..."):
            try:
                fd = {
                    "author":         author_input.strip() or None,
                    "category":       selected_genre if selected_genre != "All Genres" else None,
                    "min_rating":     min_rating if min_rating > 1.0 else None,
                    "semantic_query": selected_genre if selected_genre != "All Genres" else "popular books",
                    "user_query":     f"{selected_genre} books"
                }
                books = normalize_match(hybrid_search(fd))
                st.session_state.books          = books
                st.session_state.filters        = fd
                st.session_state.recommendation = ""
                st.session_state.rec_books      = []
                st.session_state.panel_open     = False
                if books:
                    rec = generate_recommendation(fd["user_query"], books[:5])
                    st.session_state.recommendation = rec
                    st.session_state.rec_books      = books[:3]
                    st.session_state.panel_open     = True
            except Exception as e:
                st.error(f"Filter error: {e}")

    st.divider()
    st.markdown('<p style="color:#444;font-size:0.7rem">Llama 3 · ChromaDB · Sentence Transformers</p>',
                unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────────────────────

# Header
st.markdown('<h1 class="bm-title">BookMind</h1>', unsafe_allow_html=True)
st.markdown('<p class="bm-sub">Describe what you want — AI finds the perfect book</p>', unsafe_allow_html=True)

# Search bar + button
col_s, col_b = st.columns([6, 1])
with col_s:
    query = st.text_input("search",
        placeholder="Describe your mood or what you want... AI will find and recommend books",
        label_visibility="collapsed", key="main_search")
with col_b:
    search_clicked = st.button("🤖 Ask AI", key="search_btn", use_container_width=True)

# Run search
if search_clicked and query.strip() and query != st.session_state.last_query:
    with st.spinner("🤖 Finding your books..."):
        try:
            result = recommend(query)
            books  = normalize_match(result["books"])
            st.session_state.books          = books
            st.session_state.recommendation = result["recommendation"]
            st.session_state.rec_books      = books[:3]
            st.session_state.filters        = result["filters"]
            st.session_state.last_query     = query
            st.session_state.panel_open     = True
        except Exception as e:
            st.error(f"Error: {e}")
            st.info("Check your GROQ_API_KEY is set correctly.")

# ── AI Recommendation expander (shows after search or filter) ─────────────────
if st.session_state.recommendation:
    with st.expander("🤖 AI Recommendation — top 3 picks", expanded=st.session_state.panel_open):
        rec_books = st.session_state.rec_books
        rec_text  = st.session_state.recommendation

        if rec_books:
            cols3 = st.columns(3)
            for col, book in zip(cols3, rec_books[:3]):
                with col:
                    author = book['authors'].strip("[]'\"").split("'")[0].strip()
                    cat    = clean_cat(book['categories'])
                    if book.get("image"):
                        st.image(book["image"], use_container_width=True)
                    else:
                        st.markdown('<div style="height:160px;background:#1a1a2e;border-radius:8px;'
                                    'display:flex;align-items:center;justify-content:center;font-size:2rem">📖</div>',
                                    unsafe_allow_html=True)
                    st.markdown(f"""
                    <div style="padding:0.4rem 0">
                      <div style="font-weight:600;font-size:0.85rem;color:#fff;line-height:1.3">{book['title']}</div>
                      <div style="font-size:0.73rem;color:#888;margin:0.2rem 0">{author}</div>
                      <span style="background:rgba(245,158,11,0.1);color:#f59e0b;border:1px solid
                        rgba(245,158,11,0.3);border-radius:20px;padding:1px 8px;font-size:0.63rem">{cat[:25]}</span>
                      <div style="color:#f59e0b;font-size:0.72rem;margin-top:0.3rem">{stars(book['rating'])}</div>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background:#0d0d18;border-left:3px solid #f59e0b;border-radius:0 8px 8px 0;
                    padding:1rem 1.2rem;margin-top:0.8rem;color:#ccc;font-size:0.85rem;line-height:1.75">
          {rec_text.replace('<','&lt;').replace('>','&gt;')}
        </div>
        """, unsafe_allow_html=True)









        

# ── Results grid ──────────────────────────────────────────────────────────────
books = st.session_state.books

if books:
    f = st.session_state.filters
    tags = []
    if f.get("author"):   tags.append(f"👤 {f['author']}")
    if f.get("category"): tags.append(f"🏷 {f['category']}")
    if f.get("min_rating") and float(f["min_rating"]) > 1.0:
        tags.append(f"⭐ ≥ {f['min_rating']}")
    if f.get("semantic_query"):
        tags.append(f"🔎 {f['semantic_query']}")
    if tags:
        st.markdown(" ".join(f'<span class="filter-pill">{t}</span>' for t in tags),
                    unsafe_allow_html=True)

    st.markdown(f'<div class="results-meta"><span>{len(books)}</span> books found</div>',
                unsafe_allow_html=True)

    cols = st.columns(4)
    for i, book in enumerate(books):
        with cols[i % 4]:
            st.markdown(book_card_html(book, book.get("match_pct", 0)), unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="empty-state">
      <div style="font-size:3.5rem">📚</div>
      <p style="color:#555;font-size:1rem;margin-top:1rem">Describe what you want above or use filters on the left</p>
      <p style="color:#444;font-size:0.82rem">Try: "calming philosophical read" or "funny weekend book"</p>
    </div>""", unsafe_allow_html=True)