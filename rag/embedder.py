import numpy as np
import os
from openai import AzureOpenAI

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-12-01-preview"
)

def embed_texts(texts: list[str]) -> np.ndarray:
    if not texts:
        raise ValueError("[EMBED ERROR] No input texts provided")

    try:
        response = client.embeddings.create(
            input=texts,
            model="text-embedding-3-small"  # must match your deployment name
        )

        embeddings = np.array(
            [d.embedding for d in response.data],
            dtype="float32"
        )

        if embeddings.size == 0:
            raise ValueError("[EMBED ERROR] Empty embeddings returned")

        # normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1

        return embeddings / norms

    except Exception as e:
        raise RuntimeError(f"[EMBED ERROR] {str(e)}")