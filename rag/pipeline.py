import os
import shutil

import requests

from .loader import load_directory
from .chunker import chunk_documents
from .embedder import embed_texts
from .indexer import VectorStore
from .retriever import Retriever

# Default directories (relative to this file)
_DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
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

        self.store: VectorStore | None = None
        self.retriever: Retriever | None = None
        self.all_chunks: list[dict] = []
        self.loaded_files: list[str] = []

    def build(self):
        # If a persisted index exists, load it directly (fast)
        if VectorStore.exists(self.index_dir):
            print(f"[RAG] Loading existing index from: {self.index_dir}")
            self.store = VectorStore.load(self.index_dir)
            self.retriever = Retriever(
                self.store, top_k=self.top_k, min_score=self.min_score
            )
            print(
                f"[RAG] Loaded existing index — {self.store.total} vectors, "
                f"dimension={self.store.dimension}"
            )
            return

        # Otherwise: full pipeline → load → chunk → embed → save
        print(f"[RAG] Scanning: {self.data_dir}")
        documents = load_directory(self.data_dir)

        if not documents:
            print("[RAG] No supported files found.")
            return

        # Show what was loaded
        seen = set()
        for doc in documents:
            label = doc["filename"]
            if doc["sheet_name"]:
                label += f" [sheet: {doc['sheet_name']}]"
            if label not in seen:
                print(f"  Loaded: {label}")
                seen.add(label)

        # Chunk
        self.all_chunks = chunk_documents(
            documents, self.chunk_size, self.chunk_overlap
        )
        print(f"[RAG] {len(self.all_chunks)} chunks from {len(seen)} source(s)")

        if not self.all_chunks:
            print("[RAG] No chunks generated.")
            return

        # Track loaded file names
        self.loaded_files = list(seen)

        # Embed
        texts = [c["text"] for c in self.all_chunks]
        print(f"[RAG] Embedding {len(texts)} chunks …")
        embeddings = embed_texts(texts)

        # Index
        self.store = VectorStore(dimension=embeddings.shape[1])
        self.store.add(embeddings, self.all_chunks)

        # Persist to disk
        self.store.save(self.index_dir)
        print(f"[RAG] Index saved to: {self.index_dir}")

        # Retriever
        self.retriever = Retriever(
            self.store, top_k=self.top_k, min_score=self.min_score
        )

        print(
            f"[RAG] Ready — {self.store.total} vectors "
            f"from {len(self.loaded_files)} source(s)."
        )

    # REBUILD (for when new files are added) 
    def rebuild(self):
        """Delete saved index and rebuild from scratch."""
        if os.path.isdir(self.index_dir):
            shutil.rmtree(self.index_dir)
            print(f"[RAG] Deleted old index at: {self.index_dir}")

        if self.store:
            self.store.clear()
        self.all_chunks.clear()
        self.loaded_files.clear()
        self.retriever = None
        self.build()

    # RETRIEVE
    def retrieve(self, query: str) -> list[dict]:
        """Embed the query and return the most relevant chunks with metadata."""
        if not self.retriever:
            return []
        return self.retriever.retrieve(query)

    # GENERATE (answer from retrieved context only) 
    def ask(self, query: str) -> str:
        """Retrieve context, pass to LLM, return answer. No hallucination."""
        results = self.retrieve(query)

        if not results:
            return "Not found in document."

        # Build context block with source attribution
        context_parts = []
        for r in results:
            source = r["filename"]
            if r.get("sheet_name"):
                source += f" (sheet: {r['sheet_name']})"
            context_parts.append(f"[Source: {source}]\n{r['text']}")

        context = "\n\n".join(context_parts)

        prompt = (
            "You are a strict document-answering assistant.\n"
            "Answer ONLY from the context below.\n"
            "If the answer is NOT in the context, reply exactly: "
            '"Not found in document."\n'
            "Do NOT fabricate or assume any information.\n"
            "Be professional and concise.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{query}\n\n"
            "Answer:"
        )

        try:
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 300},
                },
                timeout=120,
            )
            if resp.status_code != 200:
                return f"LLM error (HTTP {resp.status_code})."
            return resp.json()["response"].strip()

        except requests.exceptions.ConnectionError:
            return "Error: Ollama is not running on localhost:11434."
        except requests.exceptions.Timeout:
            return "Error: LLM request timed out."

    # alias
    generate = ask
