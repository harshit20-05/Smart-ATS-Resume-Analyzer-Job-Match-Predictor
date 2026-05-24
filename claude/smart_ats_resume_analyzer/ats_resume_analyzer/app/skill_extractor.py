"""
skill_extractor.py
------------------
Extracts skills from resume text by matching against a curated skill list.
Used in both EDA (skill frequency analysis) and the Streamlit dashboard.
"""

import re

# ──────────────────────────────────────────────────────────
# Master skill vocabulary  (extend as needed)
# ──────────────────────────────────────────────────────────
TECH_SKILLS = [
    # Programming languages
    "python", "java", "javascript", "c++", "c#", "ruby", "php", "swift",
    "kotlin", "golang", "scala", "r", "matlab", "typescript",
    # Web
    "html", "css", "react", "angular", "vue", "node.js", "django",
    "flask", "spring boot", "rest api", "graphql",
    # Data / ML
    "machine learning", "deep learning", "tensorflow", "pytorch", "keras",
    "pandas", "numpy", "scikit-learn", "nlp", "computer vision",
    "data analysis", "big data", "hadoop", "spark", "tableau", "power bi",
    # Databases
    "sql", "mysql", "postgresql", "mongodb", "redis", "oracle",
    "elasticsearch", "cassandra",
    # Cloud / DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", "jenkins",
    "ci/cd", "terraform", "ansible", "linux",
    # Soft skills
    "leadership", "communication", "teamwork", "problem solving",
    "project management", "agile", "scrum",
    # Domain-specific
    "autocad", "solidworks", "plc", "scada", "selenium", "sap",
    "salesforce", "excel", "powerpoint",
]

# compile once for speed
_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(s) for s in TECH_SKILLS) + r")\b",
    flags=re.IGNORECASE
)


def extract_skills(text: str) -> list:
    """Return a deduplicated list of matched skills (lowercase)."""
    if not isinstance(text, str):
        return []
    found = _PATTERN.findall(text.lower())
    return list(dict.fromkeys(found))        # preserve order, remove dups


def skill_match_score(resume_text: str, jd_text: str) -> float:
    """
    Returns the fraction of JD skills found in the resume.
    Score range: 0.0 – 1.0
    """
    resume_skills = set(extract_skills(resume_text))
    jd_skills     = set(extract_skills(jd_text))
    if not jd_skills:
        return 0.0
    matched = resume_skills & jd_skills
    return len(matched) / len(jd_skills)


if __name__ == "__main__":
    test = "Proficient in Python, TensorFlow, Docker, AWS, and SQL. Good communication skills."
    print("Skills found:", extract_skills(test))
