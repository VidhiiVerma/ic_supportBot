
from .embedder import embed_texts
from .indexer import VectorStore


class Retriever:
    """
    Wraps VectorStore to provide a clean query → results interface.
    """

    def __init__(
        self,
        store: VectorStore,
        top_k: int = 5,
        min_score: float = 0.20,
    ):
        self.store = store
        self.top_k = top_k
        self.min_score = min_score

    def retrieve(self, query: str) -> list[dict]:
        """
        Embed the query and return the most relevant chunks with metadata.

        Each returned dict contains:
            filename, file_type, sheet_name, chunk_index, text, score
        """
        if not self.store or self.store.total == 0:
            return []

        q_vec = embed_texts([query])
        return self.store.search(q_vec, top_k=self.top_k, min_score=self.min_score)
