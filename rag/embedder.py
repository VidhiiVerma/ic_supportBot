import numpy as np
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def embed_texts(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.array([], dtype="float32")

    response = client.embeddings.create(
        input=texts,
        model="text-embedding-3-small"
    )

    embeddings = np.array(
        [d.embedding for d in response.data],
        dtype="float32"
    )

    # normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return embeddings / norms