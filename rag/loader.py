import os
from .parser import parse_docx, parse_txt, parse_xlsx

_SUPPORTED = {".docx": "docx", ".txt": "txt", ".xlsx": "xlsx"}


def load_directory(data_dir: str):
    data_dir = os.path.abspath(data_dir)

    # ✅ Prevent crash
    if not os.path.exists(data_dir):
        print(f"[RAG] Directory not found: {data_dir}")
        return []

    results = []

    for fname in sorted(os.listdir(data_dir)):
        ext = os.path.splitext(fname)[1].lower()

        if ext not in _SUPPORTED:
            continue

        fpath = os.path.join(data_dir, fname)

        try:
            if ext == ".docx":
                text = parse_docx(fpath)
                results.append({
                    "filename": fname,
                    "file_type": "docx",
                    "sheet_name": None,
                    "text": text,
                })

            elif ext == ".txt":
                text = parse_txt(fpath)
                results.append({
                    "filename": fname,
                    "file_type": "txt",
                    "sheet_name": None,
                    "text": text,
                })

            elif ext == ".xlsx":
                sheets = parse_xlsx(fpath)
                for sheet in sheets:
                    results.append({
                        "filename": fname,
                        "file_type": "xlsx",
                        "sheet_name": sheet["sheet_name"],
                        "text": sheet["text"],
                    })

        except Exception as e:
            print(f"[RAG] Failed to load {fname}: {e}")

    return results