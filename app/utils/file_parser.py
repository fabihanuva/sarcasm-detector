"""
app/utils/file_parser.py
------------------------
Extract plain text from .txt, .docx, and .csv uploads.
"""

import csv
import io
from typing import List


def parse_uploaded_file(file_storage) -> List[str]:
    """
    Accept a Werkzeug FileStorage object, return a list of non-empty text lines.
    Raises ValueError for unsupported file types.
    """
    filename = file_storage.filename.lower()

    if filename.endswith(".txt"):
        return _parse_txt(file_storage)
    elif filename.endswith(".docx"):
        return _parse_docx(file_storage)
    elif filename.endswith(".csv"):
        return _parse_csv(file_storage)
    else:
        raise ValueError(f"Unsupported file type: {filename}")


def _parse_txt(fs) -> List[str]:
    content = fs.read().decode("utf-8", errors="ignore")
    return [line.strip() for line in content.splitlines() if line.strip()]


def _parse_docx(fs) -> List[str]:
    try:
        from docx import Document
        doc = Document(fs)
        return [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    except Exception as e:
        raise ValueError(f"Could not parse .docx file: {e}")


def _parse_csv(fs) -> List[str]:
    content = fs.read().decode("utf-8", errors="ignore")
    reader  = csv.reader(io.StringIO(content))
    lines   = []
    
    # Simple header detection: skip if first row contains common header keywords
    header_keywords = {"text", "headline", "content", "message", "input", "label", "is_sarcastic"}
    
    for i, row in enumerate(reader):
        if i == 0 and any(cell.lower() in header_keywords for cell in row):
            continue
        
        line = " ".join(cell.strip() for cell in row if cell.strip())
        if line:
            lines.append(line)
    return lines
