# 🎬 RAG to Riches

> *Bollywood has a dialogue for every situation.*

RAG to Riches is a local AI app that takes any real-life situation you describe — heartbreak, workplace drama, existential dread — and responds with the most dramatically fitting Bollywood dialogue. It uses a full **Retrieval-Augmented Generation (RAG)** pipeline with a locally running LLM, so everything runs on your machine with zero API costs.

---

## 📸 Demo

![RAG to Riches UI](https://i.imgur.com/placeholder.png)

---

## 🧠 How It Works — The RAG Pipeline

```
User Input (situation)
        │
        ▼
 HuggingFace Embeddings
 (paraphrase-multilingual-MiniLM-L12-v2)
        │
        ▼
 ChromaDB Vector Store ◄── built from 12,000+ Bollywood dialogues
        │
        ▼
 Top-5 Similar Dialogues retrieved (cosine similarity)
        │
        ▼
 LLM Prompt (qwen3:14b via Ollama)
 "Pick the most dramatically fitting dialogue..."
        │
        ▼
 Best Dialogue + Movie Name + Context line
        │
        ▼
 Gradio UI (typewriter effect ✨)
```

**Why RAG?**  
Instead of asking the LLM to generate a Bollywood dialogue from scratch (hallucination-prone), we first *retrieve* real dialogues from a curated dataset using semantic similarity, then ask the LLM to *pick* the best one. This gives accurate, real dialogues grounded in actual films.

---

## 🗂️ Project Structure

```
RAG-TO-RICHES/
├── backend/
│   ├── data/
│   │   ├── movies-names.txt        # movie_id~movie_name
│   │   ├── movies-quotes.txt       # movie_id~dialogue_number~dialogue
│   │   └── movies-dialogues.csv    # merged & cleaned (generated)
│   ├── dataCleanup.py              # merges + cleans the raw data → CSV
│   ├── ragDialogues.py             # core RAG function
│   ├── api.py                      # FastAPI server wrapping the RAG function
│   ├── embeddingTest.py            # standalone embedding test script
│   └── requirements.txt
├── frontend/
│   └── app.py                      # Gradio UI
├── start.sh                        # one-command launcher
└── README.md
```

---

## 🏗️ Build From Scratch

### 1. Prerequisites

- Python 3.9+
- [Homebrew](https://brew.sh/) (macOS)
- [Ollama](https://ollama.com/)

### 2. Clone the repo

```bash
git clone https://github.com/your-username/RAG-TO-RICHES.git
cd RAG-TO-RICHES
```

### 3. Create and activate virtual environment

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install dependencies

```bash
.venv/bin/python3 -m pip install --upgrade pip
.venv/bin/python3 -m pip install -r requirements.txt
```

> ⚠️ On macOS, Homebrew may override `python3` in PATH. Always use the explicit `.venv/bin/python3` path when installing packages.

### 5. Generate the cleaned dialogue dataset

```bash
python3 dataCleanup.py
```

This merges `movies-names.txt` and `movies-quotes.txt` on `movie_id`, removes nulls, removes dialogues under 6 words, removes duplicates, and writes `data/movies-dialogues.csv`.

### 6. Install and start Ollama

```bash
# install
brew install ollama

# start the server (keep this terminal open)
ollama serve

# in a new terminal — pull the model (one-time, ~9.3 GB)
ollama pull qwen3:14b

# optional: test the model directly
ollama run qwen3:14b
```

### 7. Build the vector store (first run only)

On the first run of the app, `ragDialogues.py` automatically embeds all dialogues into ChromaDB and saves them to `backend/chroma_db/`. This takes a few minutes. Subsequent runs load from disk instantly.

### 8. Run the app

```bash
cd ..   # back to project root
./start.sh
```

Open **http://127.0.0.1:7860** in your browser.

---

## ⚡ Clone and Run Locally (Quick Start)

```bash
# 1. Clone
git clone https://github.com/your-username/RAG-TO-RICHES.git
cd RAG-TO-RICHES

# 2. Set up environment
cd backend && python3 -m venv .venv && source .venv/bin/activate
.venv/bin/python3 -m pip install -r requirements.txt

# 3. Generate dataset
python3 dataCleanup.py

# 4. Start Ollama (separate terminal)
ollama serve
# and in another terminal:
ollama pull qwen3:14b   # only needed once

# 5. Launch the app
cd .. && ./start.sh
```

Then open **http://127.0.0.1:7860** 🎬

---

## ✨ Features

- 🔍 **Semantic search** across 12,000+ real Bollywood dialogues
- 🤖 **Local LLM** (qwen3:14b via Ollama) — no API keys, no costs
- 🌐 **Multilingual embeddings** — works with Hindi and English situations
- ⚠️ **Low-confidence fallback** — if no strong match is found (score < 0.35), returns a curated Bollywood classic with a warning
- ⌨️ **Typewriter effect** — dialogue streams word-by-word for dramatic flair
- 💬 **Context line** — LLM explains why the dialogue fits your situation
- 🧩 **Example chips** — 5 quick-start situations to try instantly

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Embeddings | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Vector Store | ChromaDB |
| LLM | qwen3:14b (via Ollama, local) |
| LLM Orchestration | LangChain |
| Backend API | FastAPI + Uvicorn |
| Frontend | Gradio |
| Data Processing | Pandas |

---

## 👩‍💻 Built by

**Sonal Kumari** — ML Engineer & Content Creator

Follow along for more ML projects and tutorials:

- 📸 Instagram: [@themlguppy](https://www.instagram.com/themlguppy?igsh=MWFxMDBvaTFqMHA5cQ==)
- 💼 LinkedIn: [sonalsh250](https://www.linkedin.com/in/sonalsh250)

---

*"Mere paas maa hai." — and also, a pretty solid RAG pipeline.*
