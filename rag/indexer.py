import os
import pickle

import faiss
import numpy as np


class VectorStore:
    def __init__(self, dimension: int):
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata: list[dict] = []

    def add(self, embeddings: np.ndarray, metadata_list: list[dict]):
        self.index.add(np.array(embeddings).astype("float32"))
        self.metadata.extend(metadata_list)

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 5,
        min_score: float = 0.20,
    ) -> list[dict]:
        
        if self.index.ntotal == 0:
            return []

        q = np.array(query_vec).astype("float32")
        if q.ndim == 1:
            q = q.reshape(1, -1)

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(q, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and score >= min_score:
                entry = dict(self.metadata[idx])  # copy
                entry["score"] = float(score)
                results.append(entry)

        return results

    def clear(self):
        self.index.reset()
        self.metadata.clear()

    @property
    def total(self) -> int:
        return self.index.ntotal

    @property
    def dimension(self) -> int:
        return self.index.d

    def save(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        faiss.write_index(self.index, os.path.join(directory, "faiss.index"))
        with open(os.path.join(directory, "metadata.pkl"), "wb") as f:
            pickle.dump(self.metadata, f)

    @classmethod
    def load(cls, directory: str) -> "VectorStore":
        index_path = os.path.join(directory, "faiss.index")
        meta_path = os.path.join(directory, "metadata.pkl")

        if not os.path.isfile(index_path) or not os.path.isfile(meta_path):
            raise FileNotFoundError(
                f"No persisted index found in '{directory}'. "
                "Run the indexing pipeline first."
            )

        index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)

        store = cls.__new__(cls)
        store.index = index
        store.metadata = metadata
        return store

    @staticmethod
    def exists(directory: str) -> bool:
        return (
            os.path.isfile(os.path.join(directory, "faiss.index"))
            and os.path.isfile(os.path.join(directory, "metadata.pkl"))
        )
