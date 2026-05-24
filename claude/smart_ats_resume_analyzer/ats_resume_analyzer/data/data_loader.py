"""
data_loader.py
--------------
Phase 3: Data Collection & Loading
Loads the Kaggle Resume Dataset + Job Description Dataset,
merges them, and performs initial exploration.

Dataset Links:
  - Resume Dataset   : https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset
  - Job Descriptions : https://www.kaggle.com/datasets/ravindrasinghrana/job-description-dataset

Column Descriptions:
  Resume Dataset  → 'Category' (job role), 'Resume' (raw resume text)
  Job Description → 'Job Title', 'Job Description', 'skills', 'Experience'
"""

import pandas as pd
import numpy as np
import os

# ──────────────────────────────────────────────────────────
# 3.1  Load datasets
# ──────────────────────────────────────────────────────────
def load_resume_dataset(path: str = "data/raw/resume_dataset.csv") -> pd.DataFrame:
    """Load the Kaggle resume CSV. Returns a DataFrame."""
    df = pd.read_csv(path)
    print(f"[resume_dataset]  Shape : {df.shape}")
    print(f"[resume_dataset]  Columns: {list(df.columns)}")
    print(f"[resume_dataset]  Categories ({df['Category'].nunique()}):\n"
          f"  {df['Category'].value_counts().to_dict()}\n")
    return df


def load_job_description_dataset(path: str = "data/raw/job_descriptions.csv") -> pd.DataFrame:
    """Load the Kaggle job description CSV. Returns a DataFrame."""
    df = pd.read_csv(path)
    print(f"[job_dataset]  Shape : {df.shape}")
    print(f"[job_dataset]  Columns: {list(df.columns)}\n")
    return df


# ──────────────────────────────────────────────────────────
# 3.2  Generate a synthetic dataset when real CSVs are absent
#      (useful for demo / viva purposes)
# ──────────────────────────────────────────────────────────
def generate_synthetic_dataset(n_samples: int = 500) -> pd.DataFrame:
    """
    Creates a realistic synthetic resume dataset with:
      - Category  (job role label)
      - Resume    (simulated resume text)
      - Skills    (comma-separated skill tags)
      - Experience (years as int)
      - Selected  (binary target: 1 = hired, 0 = rejected)
    """
    np.random.seed(42)

    categories = [
        "Data Science", "Web Developer", "HR", "Advocate",
        "Arts", "Mechanical Engineer", "Sales", "Health and Fitness",
        "Civil Engineer", "Java Developer", "Business Analyst",
        "SAP Developer", "Automation Testing", "Electrical Engineering",
        "Operations Manager", "Python Developer", "DevOps Engineer",
        "Network Security", "PMO", "Database Administrator"
    ]

    skill_map = {
        "Data Science"        : ["python", "pandas", "numpy", "machine learning", "sklearn", "tensorflow"],
        "Web Developer"       : ["html", "css", "javascript", "react", "node.js", "mongodb"],
        "Java Developer"      : ["java", "spring boot", "hibernate", "maven", "sql", "microservices"],
        "Python Developer"    : ["python", "django", "flask", "rest api", "postgresql", "docker"],
        "DevOps Engineer"     : ["docker", "kubernetes", "ci/cd", "jenkins", "aws", "terraform"],
        "Network Security"    : ["firewall", "vpn", "ids/ips", "penetration testing", "siem", "wireshark"],
        "Database Administrator": ["sql", "oracle", "mysql", "postgresql", "backup", "performance tuning"],
        "Business Analyst"    : ["requirement gathering", "uml", "agile", "stakeholder management", "sql"],
        "HR"                  : ["recruitment", "onboarding", "payroll", "hrms", "employee engagement"],
        "Sales"               : ["crm", "lead generation", "negotiation", "cold calling", "salesforce"],
        "Civil Engineer"      : ["autocad", "staad pro", "project management", "soil testing", "estimation"],
        "Mechanical Engineer" : ["solidworks", "catia", "ansys", "cnc machining", "thermodynamics"],
        "Electrical Engineering": ["plc", "scada", "circuit design", "matlab", "power systems"],
        "SAP Developer"       : ["sap abap", "sap mm", "sap fi", "bapi", "idoc"],
        "Automation Testing"  : ["selenium", "testng", "jira", "postman", "api testing"],
        "Operations Manager"  : ["supply chain", "lean", "six sigma", "erp", "kpi"],
        "PMO"                 : ["ms project", "pmp", "risk management", "budget planning", "agile"],
        "Arts"                : ["photoshop", "illustrator", "video editing", "content creation"],
        "Advocate"            : ["legal research", "contract drafting", "litigation", "ipc"],
        "Health and Fitness"  : ["nutrition", "fitness coaching", "cpr", "anatomy", "physiotherapy"],
    }

    rows = []
    for _ in range(n_samples):
        cat  = np.random.choice(categories)
        skills = skill_map.get(cat, ["communication", "teamwork"])
        chosen = np.random.choice(skills, size=min(4, len(skills)), replace=False)
        exp  = np.random.randint(0, 12)

        resume_text = (
            f"Experienced {cat} professional with {exp} years of experience. "
            f"Proficient in {', '.join(chosen)}. "
            f"Worked on multiple projects involving {chosen[0]} and {chosen[-1]}. "
            f"Strong analytical and problem-solving skills."
        )
        # Hiring probability: more experience + more keywords → higher chance
        prob = 0.3 + 0.04 * exp + 0.05 * len(chosen)
        selected = int(np.random.rand() < min(prob, 0.9))

        rows.append({
            "Category"   : cat,
            "Resume"     : resume_text,
            "Skills"     : ", ".join(chosen),
            "Experience" : exp,
            "Selected"   : selected
        })

    df = pd.DataFrame(rows)
    print(f"[synthetic]  Shape : {df.shape}")
    print(f"[synthetic]  Selection rate : {df['Selected'].mean():.2%}")
    return df


# ──────────────────────────────────────────────────────────
# 3.3  Basic data understanding
# ──────────────────────────────────────────────────────────
def explore_dataset(df: pd.DataFrame) -> None:
    """Prints key statistics for viva / notebook exploration."""
    print("=" * 50)
    print("DATASET OVERVIEW")
    print("=" * 50)
    print(f"Shape        : {df.shape}")
    print(f"Columns      : {list(df.columns)}")
    print(f"\ndtypes:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nDuplicates   : {df.duplicated().sum()}")
    print(f"\ndescribe():\n{df.describe(include='all')}")


# ──────────────────────────────────────────────────────────
# 3.4  Merge resume + job description datasets
# ──────────────────────────────────────────────────────────
def merge_datasets(resume_df: pd.DataFrame,
                   job_df: pd.DataFrame,
                   on: str = "Category") -> pd.DataFrame:
    """
    Left-joins resume dataset with job descriptions on the job category.
    job_df must have a 'Job Title' column that maps to resume 'Category'.
    """
    job_df_renamed = job_df.rename(columns={"Job Title": "Category"})
    merged = pd.merge(resume_df, job_df_renamed, on=on, how="left")
    print(f"[merged]  Shape after join: {merged.shape}")
    return merged


# ──────────────────────────────────────────────────────────
# 3.5  Save processed data
# ──────────────────────────────────────────────────────────
def save_processed(df: pd.DataFrame,
                   path: str = "data/processed/resume_processed.csv") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[saved]  → {path}")


# ──────────────────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = generate_synthetic_dataset(n_samples=500)
    explore_dataset(df)
    save_processed(df)
