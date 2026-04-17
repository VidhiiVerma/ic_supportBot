import re

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    sentences = re.split(r'(?<=[.!?;])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return [text] if text.strip() else []

    chunks = []
    current: list[str] = []
    current_len = 0

    for sent in sentences:
        if current_len + len(sent) > chunk_size and current:
            chunk_str = " ".join(current)
            chunks.append(chunk_str)

            overlap_sents: list[str] = []
            overlap_len = 0
            for s in reversed(current):
                if overlap_len + len(s) > overlap:
                    break
                overlap_sents.insert(0, s)
                overlap_len += len(s) + 1

            current = overlap_sents
            current_len = overlap_len

        current.append(sent)
        current_len += len(sent) + 1

    if current:
        chunks.append(" ".join(current))

    return chunks


def chunk_documents(
    docs: list[dict],
    chunk_size: int = 600,
    overlap: int = 150,
) -> list[dict]:
    """
    Takes loader output (list of dicts with filename, file_type, sheet_name, text)
    and produces chunked records with metadata:

        {
            "filename": str,
            "file_type": str,
            "sheet_name": str | None,
            "chunk_index": int,
            "text": str,
        }
    """
    chunked: list[dict] = []

    for doc in docs:
        text_chunks = chunk_text(doc["text"], chunk_size, overlap)

        for idx, chunk in enumerate(text_chunks):
            chunked.append({
                "filename": doc["filename"],
                "file_type": doc["file_type"],
                "sheet_name": doc.get("sheet_name"),
                "chunk_index": idx,
                "text": chunk,
            })

    return chunked
