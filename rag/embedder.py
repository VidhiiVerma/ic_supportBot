import numpy as np
import requests

OLLAMA_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"


def embed_texts(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.array([], dtype="float32")

    embeddings = []

    for text in texts:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": EMBED_MODEL,
                "prompt": text
            },
            timeout=120,
        )

        resp.raise_for_status()
        vector = resp.json()["embedding"]
        embeddings.append(vector)

    embeddings = np.array(embeddings, dtype="float32")

    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = embeddings / norms

    return embeddings