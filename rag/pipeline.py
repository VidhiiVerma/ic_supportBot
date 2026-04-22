import os
from openai import OpenAI

from .loader import load_directory
from .chunker import chunk_documents
from .embedder import embed_texts
from .indexer import VectorStore
from .retriever import Retriever


# ✅ Robust path (works on Render)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "app", "data")
INDEX_DIR = os.path.join(DATA_DIR, "vector_store")

# ✅ OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class RAGSystem:
    def __init__(self, top_k=5, min_score=0.2):
        self.top_k = top_k
        self.min_score = min_score
        self.store = None
        self.retriever = None

    def build(self):
        # ❌ If folder missing → fail fast
        if not os.path.exists(DATA_DIR):
            print(f"[RAG] Data folder missing: {DATA_DIR}")
            return

        # ✅ Load existing index
        if VectorStore.exists(INDEX_DIR):
            print("[RAG] Loading existing index...")
            self.store = VectorStore.load(INDEX_DIR)
        else:
            print("[RAG] Building new index...")

            docs = load_directory(DATA_DIR)
            if not docs:
                print("[RAG] No documents found")
                return

            chunks = chunk_documents(docs)
            texts = [c["text"] for c in chunks]

            embeddings = embed_texts(texts)

            self.store = VectorStore(dimension=embeddings.shape[1])
            self.store.add(embeddings, chunks)
            self.store.save(INDEX_DIR)

        # ✅ Retriever ready
        self.retriever = Retriever(
            self.store,
            top_k=self.top_k,
            min_score=self.min_score
        )

        print(f"[RAG] Ready with {self.store.total} vectors")

    def ask(self, query: str):
        if not self.retriever:
            return "System not initialized."

        results = self.retriever.retrieve(query)

        if not results:
            return "Not found in document."

        context = "\n\n".join([r["text"] for r in results])

        prompt = f"""
Answer ONLY from the context below.
If the answer is not present, reply: Not found in document.

Context:
{context}

Question:
{query}
"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Answer strictly from context."},
                    {"role": "user", "content": prompt}
                ]
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print("[RAG ERROR]", e)
            return "LLM failed."