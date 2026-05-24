"""
feature_engineering.py
-----------------------
Phase 6: Feature Engineering

Covers:
  - TF-IDF Vectorization
  - CountVectorizer
  - Feature scaling (StandardScaler)
  - PCA
  - RFE (Recursive Feature Elimination)
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing           import StandardScaler, LabelEncoder
from sklearn.decomposition           import PCA
from sklearn.feature_selection       import RFE
from sklearn.linear_model            import LogisticRegression


# ──────────────────────────────────────────────────────────
# 6.1  TF-IDF Vectorization
# ──────────────────────────────────────────────────────────
def build_tfidf(corpus: list,
                max_features: int = 3000,
                ngram_range: tuple = (1, 2)) -> tuple:
    """
    Fit a TF-IDF vectorizer on corpus.

    TF-IDF formula:
      TF(t,d)  = count(t in d) / total_terms(d)
      IDF(t)   = log(N / df(t))     [N = total docs, df = docs containing t]
      TF-IDF   = TF * IDF

    Why: Rewards rare but important words; penalises common filler words.

    Returns:
      (X_tfidf sparse matrix,  fitted TfidfVectorizer)
    """
    vectorizer = TfidfVectorizer(
        max_features = max_features,
        ngram_range  = ngram_range,    # unigrams + bigrams
        sublinear_tf = True,           # log(1+TF) instead of raw TF
        min_df       = 2,              # ignore very rare terms
        max_df       = 0.95,           # ignore near-universal terms
    )
    X = vectorizer.fit_transform(corpus)
    print(f"[TF-IDF]  Matrix shape: {X.shape}")
    return X, vectorizer


# ──────────────────────────────────────────────────────────
# 6.2  CountVectorizer  (used for Naive Bayes)
# ──────────────────────────────────────────────────────────
def build_count_vectorizer(corpus: list,
                            max_features: int = 3000) -> tuple:
    """
    Bag-of-Words representation — raw term counts.
    Required by MultinomialNB (which needs non-negative integers).
    """
    vectorizer = CountVectorizer(max_features=max_features, min_df=2)
    X = vectorizer.fit_transform(corpus)
    print(f"[CountVec]  Matrix shape: {X.shape}")
    return X, vectorizer


# ──────────────────────────────────────────────────────────
# 6.3  Feature Scaling
# ──────────────────────────────────────────────────────────
def scale_features(X: np.ndarray) -> tuple:
    """
    Standardize numeric features: z = (x - μ) / σ
    Required for algorithms sensitive to feature magnitude
    (Logistic Regression, PCA, K-Means).
    """
    scaler = StandardScaler(with_mean=False)  # sparse-safe
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


# ──────────────────────────────────────────────────────────
# 6.4  PCA — Dimensionality Reduction
# ──────────────────────────────────────────────────────────
def apply_pca(X, n_components: int = 2) -> tuple:
    """
    PCA projects data onto the directions of maximum variance.

    Formula:  Z = X · W
      where W = eigenvectors of the covariance matrix XᵀX.
      The top-k eigenvectors capture the most variance.

    Used here to:
      1. Reduce TF-IDF (3000-dim) to 2D for scatter-plot visualisation.
      2. Feed to K-Means (lower dim → faster, less noise).

    Returns:
      (X_pca array,  fitted PCA object)
    """
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X.toarray() if hasattr(X, "toarray") else X)
    explained = pca.explained_variance_ratio_.sum() * 100
    print(f"[PCA]  {n_components} components explain {explained:.1f}% variance")
    return X_pca, pca


# ──────────────────────────────────────────────────────────
# 6.5  Recursive Feature Elimination (RFE)
# ──────────────────────────────────────────────────────────
def apply_rfe(X, y, n_features: int = 100) -> tuple:
    """
    RFE iteratively removes the weakest features based on model weights.

    Steps:
      1. Fit Logistic Regression on all features.
      2. Rank features by |coefficient|.
      3. Remove bottom-ranked features.
      4. Repeat until n_features remain.

    Returns:
      (X_rfe,  fitted RFE selector,  selected feature mask)
    """
    # Use small LR as estimator — fast, interpretable weights
    estimator = LogisticRegression(max_iter=200, solver="lbfgs",
                                   multi_class="auto", C=1.0)

    # For sparse high-dimensional input, convert dense slice for RFE
    if hasattr(X, "toarray"):
        X_dense = X.toarray()
    else:
        X_dense = np.array(X)

    selector = RFE(estimator=estimator, n_features_to_select=n_features, step=50)
    X_rfe    = selector.fit_transform(X_dense, y)
    print(f"[RFE]  Features before: {X_dense.shape[1]}  →  after: {X_rfe.shape[1]}")
    return X_rfe, selector


# ──────────────────────────────────────────────────────────
# 6.6  Cosine Similarity  (for ATS score)
# ──────────────────────────────────────────────────────────
from sklearn.metrics.pairwise import cosine_similarity

def compute_cosine_similarity(vec1, vec2) -> float:
    """
    Cosine similarity between two TF-IDF vectors.

    cos(θ) = (A · B) / (||A|| × ||B||)

    Returns a value in [0, 1].
    Used in ATS score calculation: how similar is resume to JD?
    """
    sim = cosine_similarity(vec1, vec2)
    return float(sim[0][0])


if __name__ == "__main__":
    # Quick smoke test
    corpus = [
        "python machine learning data science pandas numpy",
        "java spring boot microservices sql hibernate",
        "python django rest api docker kubernetes",
    ]
    X, vec = build_tfidf(corpus)
    print("Feature names sample:", vec.get_feature_names_out()[:10])
