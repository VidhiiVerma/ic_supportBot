import os
import shutil
import requests

from .loader import load_directory
from .chunker import chunk_documents
from .embedder import embed_texts
from .indexer import VectorStore
from .retriever import Retriever


# ✅ FIXED: correct base path (works on Render)
BASE_DIR = os.getcwd()

_DEFAULT_DATA_DIR = os.path.join(BASE_DIR, "data")
_DEFAULT_INDEX_DIR = os.path.join(_DEFAULT_DATA_DIR, "vector_store")


class RAGSystem:
    def __init__(
        self,
        data_dir: str = _DEFAULT_DATA_DIR,
        index_dir: str = _DEFAULT_INDEX_DIR,
        model_name: str = "phi",
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        top_k: int = 5,
        min_score: float = 0.20,
    ):
        self.data_dir = os.path.abspath(data_dir)
        self.index_dir = os.path.abspath(index_dir)
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.min_score = min_score

        self.store = None
        self.retriever = None
        self.all_chunks = []
        self.loaded_files = []

    def build(self):
        # ✅ Prevent crash if folder missing
        if not os.path.exists(self.data_dir):
            print(f"[RAG] Data directory NOT FOUND: {self.data_dir}")
            return

        # Load existing index if exists
        if VectorStore.exists(self.index_dir):
            print(f"[RAG] Loading existing index from: {self.index_dir}")
            self.store = VectorStore.load(self.index_dir)
            self.retriever = Retriever(
                self.store, top_k=self.top_k, min_score=self.min_score
            )
            print(f"[RAG] Loaded {self.store.total} vectors")
            return

        print(f"[RAG] Scanning: {self.data_dir}")
        documents = load_directory(self.data_dir)

        if not documents:
            print("[RAG] No supported files found.")
            return

        # Chunk
        self.all_chunks = chunk_documents(
            documents, self.chunk_size, self.chunk_overlap
        )

        if not self.all_chunks:
            print("[RAG] No chunks generated.")
            return

        # Embed
        texts = [c["text"] for c in self.all_chunks]
        embeddings = embed_texts(texts)

        # Index
        self.store = VectorStore(dimension=embeddings.shape[1])
        self.store.add(embeddings, self.all_chunks)
        self.store.save(self.index_dir)

        # Retriever
        self.retriever = Retriever(
            self.store, top_k=self.top_k, min_score=self.min_score
        )

        print(f"[RAG] Ready with {self.store.total} vectors")

    def retrieve(self, query: str):
        if not self.retriever:
            return []
        return self.retriever.retrieve(query)

    def ask(self, query: str):
        results = self.retrieve(query)

        if not results:
            return "Not found in document."

        context = "\n\n".join([r["text"] for r in results])

        prompt = f"""
Answer ONLY from context.
If not found, say: Not found in document.

Context:
{context}

Question:
{query}
"""

        try:
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": self.model_name, "prompt": prompt, "stream": False},
                timeout=30,
            )
            return resp.json().get("response", "").strip()

        except:
            return "LLM not available."

    generate = ask