import os
from .parser import parse_docx, parse_txt, parse_xlsx

_SUPPORTED = {".docx": "docx", ".txt": "txt", ".xlsx": "xlsx"}


def load_directory(data_dir: str):
    data_dir = os.path.abspath(data_dir)

    if not os.path.exists(data_dir):
        print(f"[RAG] Directory not found: {data_dir}")
        return []

    results = []

    for fname in sorted(os.listdir(data_dir)):
        fpath = os.path.join(data_dir, fname)

        # ✅ skip folders
        if os.path.isdir(fpath):
            continue

        ext = os.path.splitext(fname)[1].lower()

        if ext not in _SUPPORTED:
            continue

        try:
            print(f"[RAG] Loading {fname}...")

            if ext == ".docx":
                text = parse_docx(fpath)

                if not text.strip():
                    print(f"[RAG] Empty docx skipped: {fname}")
                    continue

                results.append({
                    "filename": fname,
                    "file_type": "docx",
                    "sheet_name": None,
                    "text": text.strip(),
                })

            elif ext == ".txt":
                text = parse_txt(fpath)

                if not text.strip():
                    print(f"[RAG] Empty txt skipped: {fname}")
                    continue

                results.append({
                    "filename": fname,
                    "file_type": "txt",
                    "sheet_name": None,
                    "text": text.strip(),
                })

            elif ext == ".xlsx":
                sheets = parse_xlsx(fpath)

                for sheet in sheets:
                    text = sheet.get("text", "").strip()

                    if not text:
                        continue

                    results.append({
                        "filename": fname,
                        "file_type": "xlsx",
                        "sheet_name": sheet.get("sheet_name"),
                        "text": text,
                    })

        except Exception as e:
            print(f"[RAG] Failed to load {fname}: {e}")

    print(f"[RAG] Total docs loaded: {len(results)}")
    return results