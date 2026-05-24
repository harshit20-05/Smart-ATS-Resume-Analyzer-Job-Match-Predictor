"""
preprocessor.py
---------------
Phase 4: Data Cleaning & NLP Preprocessing
All text-cleaning functions live here.

Pipeline:
  Raw Text
    → lowercase
    → remove URLs / emails / special characters
    → remove stopwords
    → lemmatize
    → tokenize
    → return clean text string (for TF-IDF) and token list
"""

import re
import string
import pandas as pd
import numpy as np

# NLTK downloads (run once)
import nltk
nltk.download("stopwords",    quiet=True)
nltk.download("punkt",        quiet=True)
nltk.download("wordnet",      quiet=True)
nltk.download("punkt_tabular", quiet=True)

from nltk.corpus import stopwords
from nltk.stem   import WordNetLemmatizer
from nltk.tokenize import word_tokenize

STOP_WORDS  = set(stopwords.words("english"))
LEMMATIZER  = WordNetLemmatizer()


# ──────────────────────────────────────────────────────────
# 4.1  Individual cleaning helpers
# ──────────────────────────────────────────────────────────

def to_lowercase(text: str) -> str:
    """Convert all characters to lowercase."""
    return text.lower()


def remove_urls(text: str) -> str:
    """Remove http/https URLs and www links."""
    return re.sub(r"http\S+|www\S+|https\S+", " ", text)


def remove_emails(text: str) -> str:
    """Remove email addresses."""
    return re.sub(r"\S+@\S+", " ", text)


def remove_phone_numbers(text: str) -> str:
    """Remove common phone number patterns."""
    return re.sub(r"[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]", " ", text)


def remove_special_characters(text: str) -> str:
    """Remove punctuation and non-alphanumeric characters."""
    # Keep spaces; remove everything else that's not a letter/digit
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stopwords(text: str) -> str:
    """Remove English stopwords."""
    tokens = text.split()
    filtered = [w for w in tokens if w not in STOP_WORDS and len(w) > 2]
    return " ".join(filtered)


def lemmatize_text(text: str) -> str:
    """Lemmatize each token (e.g. 'running' → 'run')."""
    tokens = text.split()
    lemmas = [LEMMATIZER.lemmatize(w) for w in tokens]
    return " ".join(lemmas)


def tokenize(text: str) -> list:
    """Return list of word tokens from clean text."""
    return word_tokenize(text)


# ──────────────────────────────────────────────────────────
# 4.2  Full pipeline in one function
# ──────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Master cleaning function — apply all steps in order.
    Returns a single clean string ready for TF-IDF.
    """
    if not isinstance(text, str):
        return ""

    text = to_lowercase(text)
    text = remove_urls(text)
    text = remove_emails(text)
    text = remove_phone_numbers(text)
    text = remove_special_characters(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    return text


# ──────────────────────────────────────────────────────────
# 4.3  Apply to full DataFrame
# ──────────────────────────────────────────────────────────

def preprocess_dataframe(df: pd.DataFrame,
                          text_col: str = "Resume") -> pd.DataFrame:
    """
    Apply clean_text() to the resume text column.
    Adds a new column 'Cleaned_Resume'.
    Also handles:
      - Null/NaN text values   → replaced with empty string
      - Duplicate rows         → dropped
      - Whitespace-only rows   → dropped
    """
    df = df.copy()

    # --- handle nulls ---
    null_count = df[text_col].isnull().sum()
    print(f"[preprocess]  Null values in '{text_col}': {null_count}")
    df[text_col].fillna("", inplace=True)

    # --- drop duplicates ---
    before = len(df)
    df.drop_duplicates(subset=[text_col], inplace=True)
    print(f"[preprocess]  Duplicates removed: {before - len(df)}")

    # --- clean text ---
    print("[preprocess]  Cleaning text … (this may take a moment)")
    df["Cleaned_Resume"] = df[text_col].apply(clean_text)

    # --- drop empty rows after cleaning ---
    df = df[df["Cleaned_Resume"].str.strip() != ""].reset_index(drop=True)
    print(f"[preprocess]  Final shape: {df.shape}")
    return df


# ──────────────────────────────────────────────────────────
# 4.4  Encode target labels
# ──────────────────────────────────────────────────────────

from sklearn.preprocessing import LabelEncoder

def encode_labels(df: pd.DataFrame,
                  label_col: str = "Category") -> tuple:
    """
    Label-encode the job category column.
    Returns (df_with_encoded_col, fitted_LabelEncoder).
    """
    le = LabelEncoder()
    df = df.copy()
    df["Category_Encoded"] = le.fit_transform(df[label_col])
    print(f"[encode]  Classes ({len(le.classes_)}):\n  {list(le.classes_)}")
    return df, le


# ──────────────────────────────────────────────────────────
# Quick self-test
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample = ("Experienced DATA SCIENTIST with 5 years. "
              "Contact: john@gmail.com | +91-9876543210 | www.portfolio.com "
              "Skills: Python, Machine Learning, TensorFlow, SQL. "
              "Led end-to-end ML pipelines. Excellent communication skills.")

    print("Original :", sample)
    print("Cleaned  :", clean_text(sample))
