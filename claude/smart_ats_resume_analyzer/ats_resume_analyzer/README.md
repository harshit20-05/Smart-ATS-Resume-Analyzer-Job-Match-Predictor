# Smart ATS Resume Analyzer & Job Match Predictor

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31-red)

> An AI-powered Applicant Tracking System built from scratch using classical Machine Learning.

---

## Project Overview

This system automates resume screening by:
- Extracting text from PDF resumes
- Calculating ATS compatibility scores using cosine similarity
- Classifying resumes into job categories (Naive Bayes)
- Predicting candidate selection probability (Logistic Regression)
- Clustering similar resumes (K-Means + Hierarchical)
- Reducing dimensionality for visualisation (PCA)

---

## Folder Structure

```
ats_resume_analyzer/
├── data/
│   ├── raw/                  # Kaggle CSV files go here
│   ├── processed/            # Cleaned data after preprocessing
│   ├── data_loader.py        # Dataset loading + synthetic generator
│   ├── preprocessor.py       # NLP text cleaning pipeline
│   └── feature_engineering.py # TF-IDF, RFE, PCA, scaling
│
├── models/
│   ├── ml_models.py          # All ML model code
│   ├── logistic_regression.pkl
│   ├── naive_bayes.pkl
│   ├── kmeans.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── label_encoder.pkl
│   └── linear_regression.pkl
│
├── app/
│   ├── streamlit_app.py      # Streamlit dashboard
│   ├── ats_scorer.py         # ATS score calculator
│   ├── resume_parser.py      # PDF text extraction
│   └── skill_extractor.py    # Skill keyword matching
│
├── screenshots/              # Saved chart PNGs
├── reports/                  # EDA and evaluation reports
├── main.py                   # Master training pipeline
├── requirements.txt
└── README.md
```

---

## ML Algorithms

| Algorithm | Task | Key Metric |
|---|---|---|
| Logistic Regression | Selection prediction | F1 = 0.83, AUC = 0.53 |
| Naive Bayes (Multinomial) | Category classification | Acc = 1.00 |
| K-Means Clustering | Resume segmentation | Inertia minimised |
| Hierarchical Clustering | Skill grouping | Ward linkage |
| Linear Regression | ATS score | R² = 0.44 |
| PCA | Dim reduction (3000→2) | 2 components |
| RFE | Feature selection | Top 100 features |

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train all models
```bash
python main.py
```

### 3. Launch the dashboard
```bash
streamlit run app/streamlit_app.py
```

---

## Dataset Download

1. Resume Dataset → https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset
   - Place as `data/raw/resume_dataset.csv`
   - Columns: `Category`, `Resume`

2. Job Description Dataset → https://www.kaggle.com/datasets/ravindrasinghrana/job-description-dataset
   - Place as `data/raw/job_descriptions.csv`
   - Columns: `Job Title`, `Job Description`, `skills`, `Experience`

---

## ATS Score Formula

```
ATS_Score = 0.40 × Skill_Match
          + 0.35 × Cosine_Similarity
          + 0.15 × Keyword_Density
          + 0.10 × Experience_Score
```

Score range: 0–100
- ≥ 75 → Excellent Match ✅
- 50–74 → Moderate Match ⚠️
- < 50  → Poor Match ❌

---

## Key Formulas Covered in Project

**TF-IDF:**  `TF-IDF(t,d) = TF(t,d) × log(N/df(t))`

**Logistic Regression:** `P(y=1) = 1 / (1 + e^{-z})` where `z = β₀ + Σβᵢxᵢ`

**Bayes Theorem:** `P(Class|words) = P(words|Class) × P(Class) / P(words)`

**K-Means Cost:** `J = Σᵢ Σ_{x∈Cᵢ} ||x − μᵢ||²`

**PCA:** `Z = X · W`  (W = top eigenvectors of covariance matrix)

**Cosine Similarity:** `cos(θ) = (A·B) / (||A|| × ||B||)`

---

## Viva Questions (with Answers)

**Q: Why Naive Bayes for text classification?**
A: Naive Bayes is computationally efficient, works well with high-dimensional text (TF-IDF), and leverages Bayes' theorem with the independence assumption for fast training.

**Q: What is the difference between Precision and Recall?**
A: Precision = TP/(TP+FP) — of all positive predictions, how many are correct. Recall = TP/(TP+FN) — of all actual positives, how many did we catch. F1 balances both.

**Q: Why PCA before K-Means?**
A: TF-IDF vectors are 3000+ dimensions. K-Means uses Euclidean distance which degrades in high dimensions (curse of dimensionality). PCA reduces to 2D, making clustering more meaningful.

**Q: What is cosine similarity?**
A: Measures the angle between two vectors regardless of magnitude. cos(θ)=0 → orthogonal (no overlap), cos(θ)=1 → identical. Used to compare resume and JD text vectors.

**Q: Why not use accuracy alone?**
A: For imbalanced datasets (e.g. more rejections than selections), a model predicting "always reject" has high accuracy but is useless. F1, Precision, Recall, and AUC are better holistic measures.

**Q: What is the elbow method in K-Means?**
A: Plots WCSS (within-cluster sum of squares) vs K. The "elbow" — where the decrease in WCSS starts to flatten — indicates the optimal K beyond which additional clusters give diminishing returns.

---

## Future Scope
- Deep Learning resume parsing (BERT embeddings)
- Named Entity Recognition for skills/education/experience extraction
- Job recommendation engine
- Real-time API integration with LinkedIn/Naukri
- Multi-language resume support

---

## Tech Stack
Python 3.10+ · Pandas · NumPy · Scikit-learn · NLTK · Streamlit · Matplotlib · Seaborn · WordCloud · PDFPlumber

---

*B.Tech CSE Machine Learning Project*
