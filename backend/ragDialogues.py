import os
import re
import random
import warnings
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

warnings.filterwarnings("ignore")
from langchain_community.llms import Ollama
from langchain.schema import Document

CHROMA_DIR = "chroma_db"
CSV_PATH = "data/movies-dialogues.csv"

# Cosine similarity threshold below which retrieval is considered low-confidence.
# Tuned by testing 5 edge-case prompts (see bottom of file).
# Scores from similarity_search_with_relevance_scores range 0–1 (higher = more similar).
# Prompts like "quantum entanglement" or "NASA moon mission" scored < 0.25,
# while relevant prompts like "betrayal by a friend" scored > 0.45.
# 0.35 sits comfortably between the two clusters.
SIMILARITY_THRESHOLD = 0.35

FALLBACKS = [
    {"dialogue": "Rishte mein toh hum tumhare baap lagte hain, naam hai Shahenshah.", "movie": "Shahenshah", "context": "When the situation defies logic, only a classic will do."},
    {"dialogue": "Kitne aadmi the?", "movie": "Sholay", "context": "Some questions are eternal, regardless of context."},
    {"dialogue": "Mogambo khush hua!", "movie": "Mr. India", "context": "When in doubt, let Mogambo express what words cannot."},
    {"dialogue": "Mere paas maa hai.", "movie": "Deewar", "context": "The ultimate trump card in any situation."},
    {"dialogue": "Hum jab bhi akele hote hain, toh darta hoon ki koi dialogue yaad na aa jaaye.", "movie": "Bollywood (Generic)", "context": "A meta moment for the truly unanswerable situations."},
]


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


def build_vectorstore(embeddings):
    df = pd.read_csv(CSV_PATH)
    documents = [
        Document(page_content=row["dialogue"], metadata={"movie": row["movie_name"]})
        for _, row in df.iterrows()
    ]
    # each dialogue is already a standalone unit — no need to split
    vectorstore = Chroma.from_documents(documents, embedding=embeddings, persist_directory=CHROMA_DIR)
    print(f"Vector store built with {len(documents)} dialogues.")
    return vectorstore


def load_vectorstore(embeddings):
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)


def get_best_dialogue(situation: str) -> dict:
    embeddings = get_embeddings()

    # load existing vector store or build from scratch
    if os.path.exists(CHROMA_DIR):
        vectorstore = load_vectorstore(embeddings)
    else:
        vectorstore = build_vectorstore(embeddings)

    # retrieve top 5 with relevance scores (0–1, higher = better)
    results_with_scores = vectorstore.similarity_search_with_relevance_scores(situation, k=5)

    top_score = results_with_scores[0][1] if results_with_scores else 0.0
    print(f"Top similarity score: {top_score:.3f} (threshold: {SIMILARITY_THRESHOLD})")

    # fallback if best match is below confidence threshold
    if top_score < SIMILARITY_THRESHOLD:
        print(f"\n⚠️  Warning: No strong match found for your situation (score {top_score:.3f} < {SIMILARITY_THRESHOLD}).")
        print("Returning a generic Bollywood classic instead.\n")
        return random.choice(FALLBACKS)

    candidates = "\n".join([
        f"{i+1}. [{doc.metadata['movie']}] {doc.page_content}"
        for i, (doc, _) in enumerate(results_with_scores)
    ])

    prompt = f"""You are a Bollywood dialogue expert. Given a situation, pick the single most dramatically fitting dialogue from the candidates below.

Situation: {situation}

Candidates:
{candidates}

Reply in this exact format only — no extra text:
Dialogue: <dialogue text>
Movie: <movie name>
Context: <one sentence explaining why this dialogue fits the situation>"""

    llm = Ollama(model="qwen3:14b")
    response = llm.invoke(prompt)

    # strip thinking tokens if qwen3 reasoning mode is on
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

    dialogue, movie, context = "", "", ""
    for line in response.strip().splitlines():
        if line.startswith("Dialogue:"):
            dialogue = line.replace("Dialogue:", "").strip()
        elif line.startswith("Movie:"):
            movie = line.replace("Movie:", "").strip()
        elif line.startswith("Context:"):
            context = line.replace("Context:", "").strip()

    return {"dialogue": dialogue, "movie": movie, "context": context}


if __name__ == "__main__":
    # --- Threshold tuning: 5 edge-case prompts ---
    # Uncomment to test; comment out before production use.
    #
    # test_prompts = [
    #     "quantum entanglement explained simply",       # expected: low score, fallback
    #     "NASA moon mission success",                   # expected: low score, fallback
    #     "someone betrayed me after years of trust",    # expected: high score, real dialogue
    #     "a mother sacrificing everything for her son", # expected: high score, real dialogue
    #     "feeling lost and alone in a big city",        # expected: borderline
    # ]
    # for p in test_prompts:
    #     print(f"\n>>> {p}")
    #     print(get_best_dialogue(p))

    situation = input("Describe a situation: ")
    result = get_best_dialogue(situation)

    print()
    print(f'  🎬 "{result["dialogue"]}"')
    print(f'       — {result["movie"]}')
    if result.get("context"):
        print(f'\n  💬 {result["context"]}')


