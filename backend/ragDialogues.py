import os
import re
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.schema import Document

CHROMA_DIR = "chroma_db"
CSV_PATH = "data/movies-dialogues.csv"


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

    # retrieve top 5 semantically similar dialogues
    results = vectorstore.similarity_search(situation, k=5)

    candidates = "\n".join([
        f"{i+1}. [{doc.metadata['movie']}] {doc.page_content}"
        for i, doc in enumerate(results)
    ])

    prompt = f"""You are a Bollywood dialogue expert. Given a situation, pick the single most dramatically fitting dialogue from the candidates below.

Situation: {situation}

Candidates:
{candidates}

Reply in this exact format only — no explanation:
Dialogue: <dialogue text>
Movie: <movie name>"""

    llm = Ollama(model="qwen3:14b")
    response = llm.invoke(prompt)

    # strip thinking tokens if qwen3 reasoning mode is on
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

    dialogue, movie = "", ""
    for line in response.strip().splitlines():
        if line.startswith("Dialogue:"):
            dialogue = line.replace("Dialogue:", "").strip()
        elif line.startswith("Movie:"):
            movie = line.replace("Movie:", "").strip()

    return {"dialogue": dialogue, "movie": movie}


if __name__ == "__main__":
    situation = input("Describe a situation: ")
    result = get_best_dialogue(situation)
    print(f"\nDialogue : {result['dialogue']}")
    print(f"Movie    : {result['movie']}")

