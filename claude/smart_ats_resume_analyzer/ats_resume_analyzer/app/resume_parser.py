"""
resume_parser.py
----------------
PDF text extraction using pdfplumber (preferred) with PyPDF2 as fallback.
"""

import io

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Accepts raw PDF bytes (from Streamlit st.file_uploader)
    and returns extracted text string.
    """
    text = ""

    # --- Try pdfplumber first (better layout handling) ---
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        if text.strip():
            return text
    except Exception as e:
        print(f"[pdfplumber]  failed: {e}")

    # --- Fallback: PyPDF2 ---
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        for page in reader.pages:
            text += page.extract_text() or ""
    except Exception as e:
        print(f"[PyPDF2]  failed: {e}")

    return text


def extract_text_from_file(filepath: str) -> str:
    """Helper to read a local PDF file and return text."""
    with open(filepath, "rb") as f:
        return extract_text_from_pdf(f.read())
