import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

# load cleaned dialogues
df = pd.read_csv("data/movies-dialogues.csv")

# each dialogue becomes a Document; movie name stored as metadata
documents = [
    Document(page_content=row["dialogue"], metadata={"movie": row["movie_name"]})
    for _, row in df.iterrows()
]

# split long dialogues into smaller chunks if needed
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = splitter.split_documents(documents)

# multilingual model — works well for Hindi/English dialogues
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# build ChromaDB vector store from chunks
vectorstore = Chroma.from_documents(chunks, embedding=embeddings, persist_directory="chroma_db")

print(f"Vector store built with {len(chunks)} chunks.")

# --- test similarity search ---
query = "sad life"
results = vectorstore.similarity_search(query, k=5)

print(f"\nTop 5 dialogues for: '{query}'\n")
for i, doc in enumerate(results, 1):
    print(f"{i}. [{doc.metadata['movie']}] {doc.page_content}")

