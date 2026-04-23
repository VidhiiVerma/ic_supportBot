import os
from openai import AzureOpenAI

from .loader import load_directory
from .chunker import chunk_documents
from .embedder import embed_texts
from .indexer import VectorStore
from .retriever import Retriever

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
INDEX_DIR = os.path.join(DATA_DIR, "vector_store")

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-12-01-preview"
)


class RAGSystem:
    def __init__(self, top_k=5, min_score=0.2):
        self.top_k = top_k
        self.min_score = min_score
        self.store = None
        self.retriever = None

    def build(self):
        if not os.path.exists(DATA_DIR):
            print(f"[RAG] Data folder missing: {DATA_DIR}")
            return

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

        self.retriever = Retriever(
            self.store,
            top_k=self.top_k,
            min_score=self.min_score
        )
        print(f"[RAG] Ready with {self.store.total} vectors")

    def ask(self, query: str) -> dict:
        if not self.retriever:
            return {"answer": "System not initialized.", "context": None}

        results = self.retriever.retrieve(query)

        if not results or not isinstance(results, list):
            print("[RAG ERROR] Invalid results:", results)
            return {"answer": "Not found in document.", "context": None}

        # Safely extract text whether results are dicts or plain strings
        context_parts = []
        for r in results:
            if isinstance(r, dict):
                context_parts.append(r.get("text", ""))
            elif isinstance(r, str):
                context_parts.append(r)

        context = "\n\n".join(filter(None, context_parts))

        if not context.strip():
            return {"answer": "Not found in document.", "context": None}

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
                model="gpt-5-chat",
                messages=[
                    {"role": "system", "content": "Answer strictly from context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=400
            )
            answer = response.choices[0].message.content.strip()
            return {"answer": answer, "context": context}  # ✅ always a dict

        except Exception as e:
            print("[RAG ERROR]", e)
            return {"answer": "LLM failed.", "context": None}