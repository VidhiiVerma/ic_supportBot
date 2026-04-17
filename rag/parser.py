import os
import pandas as pd
from docx import Document


def parse_docx(file_path: str) -> str:
    doc = Document(file_path)
    parts = []

    for element in doc.element.body:
        tag = element.tag.split("}")[-1]

        if tag == "p":
            text = element.text
            if text and text.strip():
                parts.append(text.strip())

        elif tag == "tbl":
            for table in doc.tables:
                if table._element is element:
                    header = [c.text.strip() for c in table.rows[0].cells]
                    parts.append(" | ".join(header))
                    parts.append("-" * 50)
                    for row in table.rows[1:]:
                        cells = [c.text.strip() for c in row.cells]
                        if any(cells):
                            parts.append(" | ".join(cells))
                    parts.append("")
                    break

    return "\n".join(parts)


def parse_txt(file_path: str) -> str:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            with open(file_path, "r", encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    with open(file_path, "rb") as f:
        return f.read().decode("utf-8", errors="replace")


def parse_xlsx(file_path: str) -> list[dict]:
    sheets = pd.read_excel(file_path, sheet_name=None, engine="openpyxl")
    results = []

    for sheet_name, df in sheets.items():
        rows_text = []
        columns = [str(c).strip() for c in df.columns]

        for _, row in df.iterrows():
            parts = []
            for col in columns:
                val = row[col]
                if pd.notna(val):
                    parts.append(f"{col} is {val}")
            if parts:
                rows_text.append(", ".join(parts) + ".")

        if rows_text:
            results.append({
                "sheet_name": sheet_name,
                "text": "\n".join(rows_text),
            })

    return results
