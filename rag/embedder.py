import numpy as np
import os
from openai import AzureOpenAI

# Azure client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version = "2024-12-01-preview"
)

def embed_texts(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.array([], dtype="float32")

    try:
        response = client.embeddings.create(
            input=texts,
            model="text-embedding-3-small"  
        )

        embeddings = np.array(
            [d.embedding for d in response.data],
            dtype="float32"
        )

        # normalize (cosine similarity prep)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1

        return embeddings / norms

    except Exception as e:
        print(f"[EMBED ERROR] {str(e)}")
        return np.array([], dtype="float32")