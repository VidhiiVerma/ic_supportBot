import os
from .parser import parse_docx, parse_txt, parse_xlsx

_SUPPORTED = {".docx": "docx", ".txt": "txt", ".xlsx": "xlsx"}


def load_directory(data_dir: str) -> list[dict]:
    data_dir = os.path.abspath(data_dir)
    results: list[dict] = []

    for fname in sorted(os.listdir(data_dir)):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in _SUPPORTED:
            continue

        fpath = os.path.join(data_dir, fname)
        file_type = _SUPPORTED[ext]

        if file_type == "docx":
            text = parse_docx(fpath)
            results.append({
                "filename": fname,
                "file_type": file_type,
                "sheet_name": None,
                "text": text,
            })

        elif file_type == "txt":
            text = parse_txt(fpath)
            results.append({
                "filename": fname,
                "file_type": file_type,
                "sheet_name": None,
                "text": text,
            })

        elif file_type == "xlsx":
            sheets = parse_xlsx(fpath)
            for sheet in sheets:
                results.append({
                    "filename": fname,
                    "file_type": file_type,
                    "sheet_name": sheet["sheet_name"],
                    "text": sheet["text"],
                })

    return results
