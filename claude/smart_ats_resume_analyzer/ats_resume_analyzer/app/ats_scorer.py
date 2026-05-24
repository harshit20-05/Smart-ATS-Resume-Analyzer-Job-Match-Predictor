"""
ats_scorer.py
-------------
Phase 9: ATS Score Calculation

ATS Score formula (weighted composite):

  ATS_Score = (W1 × Skill_Match) +
              (W2 × Keyword_Density) +
              (W3 × Cosine_Similarity) +
              (W4 × Experience_Score)

  Where W1+W2+W3+W4 = 1.0  (weights sum to 1)

Default weights:
  Skill match      : 40%
  Cosine similarity: 35%
  Keyword density  : 15%
  Experience       : 10%
"""

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise        import cosine_similarity

from app.skill_extractor import extract_skills, skill_match_score


# ──────────────────────────────────────────────────────────
# Component 1: Keyword Density Score
# ──────────────────────────────────────────────────────────
def keyword_density_score(resume_text: str, jd_text: str) -> float:
    """
    Fraction of JD keywords (non-stopword tokens) found in resume.
    Returns value in [0, 1].
    """
    jd_words = set(re.findall(r"\b[a-z]{4,}\b", jd_text.lower()))
    resume_words = set(re.findall(r"\b[a-z]{4,}\b", resume_text.lower()))
    if not jd_words:
        return 0.0
    matched = jd_words & resume_words
    return len(matched) / len(jd_words)


# ──────────────────────────────────────────────────────────
# Component 2: Cosine Similarity
# ──────────────────────────────────────────────────────────
def text_cosine_similarity(resume_text: str, jd_text: str) -> float:
    """
    TF-IDF cosine similarity between resume and job description.

    cos(θ) = (Resume_vec · JD_vec) / (||Resume_vec|| × ||JD_vec||)

    Measures overall content overlap beyond simple keyword matching.
    """
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    try:
        matrix = vectorizer.fit_transform([resume_text, jd_text])
        sim = cosine_similarity(matrix[0:1], matrix[1:2])
        return float(sim[0][0])
    except Exception:
        return 0.0


# ──────────────────────────────────────────────────────────
# Component 3: Experience Score
# ──────────────────────────────────────────────────────────
def experience_score(resume_text: str, required_years: int = 0) -> float:
    """
    Extracts years of experience from resume text.
    Returns normalised score in [0, 1].
    """
    pattern = r"(\d+)\s*(?:\+)?\s*(?:years?|yrs?)\s*(?:of)?\s*experience"
    matches = re.findall(pattern, resume_text.lower())
    if not matches:
        return 0.3   # baseline if not stated

    years_found = max(int(m) for m in matches)

    if required_years <= 0:
        # No requirement → normalise against 10 years
        return min(years_found / 10.0, 1.0)
    else:
        return min(years_found / required_years, 1.0)


# ──────────────────────────────────────────────────────────
# Master ATS Score
# ──────────────────────────────────────────────────────────
def calculate_ats_score(resume_text: str,
                         jd_text: str,
                         required_years: int = 0,
                         weights: dict = None) -> dict:
    """
    Calculate the full ATS compatibility score.

    Returns a dict with:
      - ats_score      : final composite score (0–100)
      - skill_match    : skill overlap fraction
      - cosine_sim     : TF-IDF cosine similarity
      - keyword_density: JD keyword coverage
      - experience     : experience component
      - matched_skills : list of matched skills
      - missing_skills : list of JD skills not in resume
    """
    if weights is None:
        weights = {
            "skill_match"     : 0.40,
            "cosine_sim"      : 0.35,
            "keyword_density" : 0.15,
            "experience"      : 0.10,
        }

    # --- compute each component ---
    sm   = skill_match_score(resume_text, jd_text)
    cs   = text_cosine_similarity(resume_text, jd_text)
    kd   = keyword_density_score(resume_text, jd_text)
    exp  = experience_score(resume_text, required_years)

    # --- weighted composite ---
    raw_score = (
        weights["skill_match"]      * sm   +
        weights["cosine_sim"]       * cs   +
        weights["keyword_density"]  * kd   +
        weights["experience"]       * exp
    )
    ats_score = round(raw_score * 100, 1)

    # --- matched / missing skills ---
    resume_skills = set(extract_skills(resume_text))
    jd_skills     = set(extract_skills(jd_text))
    matched = sorted(resume_skills & jd_skills)
    missing = sorted(jd_skills - resume_skills)

    return {
        "ats_score"       : ats_score,
        "skill_match"     : round(sm * 100, 1),
        "cosine_sim"      : round(cs * 100, 1),
        "keyword_density" : round(kd * 100, 1),
        "experience"      : round(exp * 100, 1),
        "matched_skills"  : matched,
        "missing_skills"  : missing,
    }


# ──────────────────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    resume = ("5 years experience Python developer. Proficient in "
              "Python, Django, REST API, Docker, PostgreSQL, AWS. "
              "Led backend microservices team.")
    jd = ("Looking for Python developer 3+ years experience. "
          "Must know Django, REST API, Docker, AWS, CI/CD.")

    result = calculate_ats_score(resume, jd, required_years=3)
    for k, v in result.items():
        print(f"  {k:20s}: {v}")
