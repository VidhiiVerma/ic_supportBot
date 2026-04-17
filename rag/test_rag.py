"""
Interactive RAG test.
Run from ic_chatbot_backend/:  python -m rag.test_rag
"""
from rag.pipeline import RAGSystem

rag = RAGSystem()
rag.build()

print("\nReady! Ask a question (or type 'quit').\n")

while True:
    q = input("Ask: ").strip()
    if not q or q.lower() in ("quit", "exit", "q"):
        break

    # Show retrieval results with metadata
    hits = rag.retrieve(q)
    print(f"\n── {len(hits)} chunks retrieved ──")
    for i, hit in enumerate(hits, 1):
        source = hit["filename"]
        if hit.get("sheet_name"):
            source += f" [sheet: {hit['sheet_name']}]"
        preview = hit["text"][:120].replace("\n", " ")
        print(f"  [{i}] score={hit['score']:.3f}  src={source}  {preview} …")

    # Generate answer
    answer = rag.ask(q)
    print(f"\nAnswer:\n{answer}\n")
