# 📚 BookMind — AI Book Recommender

> An AI-powered book recommendation engine that understands what you **FEEL**, not just what you type. Built as a first-year Computer Science project.

[![Streamlit App](https://static.streamlit.io/badge_gradient.svg)](https://bookmind-ai-recommender.streamlit.app/)

Instead of forcing users to rely on rigid filters, exact keywords, or specific genres, **BookMind** allows you to describe your exact mood or current state of mind. 

* **Example Query:** *"I'm feeling anxious and overwhelmed, I want something calming, short, and philosophical."*
* **Result:** The system extracts your underlying intent, runs a mathematical vector search across a dataset of **2,000+ books**, and returns highly personalized recommendations alongside an AI-generated explanation of *why* they fit your mood.

---

## 🚀 Live Demo

Experience the live app here: **[BookMind Web App](https://bookmind-ai-recommender.streamlit.app/)**

---

## 🛠️ Tech Stack & Architecture

This project implements a **Hybrid Semantic Search** pipeline—the exact core architecture used by large-scale production recommendation systems.

* **Frontend & UI:** [Streamlit](https://streamlit.io/) (Deployed via Streamlit Cloud)
* **LLM Orchestration:** Llama 3 via **Groq API** (for lightning-fast natural language understanding and intent extraction)
* **Embedding Model:** `sentence-transformers` (converts book data and user queries into dense numerical vectors)
* **Vector Search Engine:** Custom **NumPy** implementation executing fast mathematical cosine similarity calculations
* **Search Strategy:** **Hybrid Search** (combining advanced AI semantic matching with metadata filtering for genre, author, and ratings)

---

## 📁 Repository Structure

* `app.py` — The core Streamlit web application interface and user session layout.
* `search.py` — The mathematical engine driving the semantic search and NumPy vector matching.
* `books_data.json` — Pre-processed catalog containing metadata for over 2,000 books.
* `books_embeddings.npy` — Pre-computed multi-dimensional vector embeddings of the book descriptions.

---
