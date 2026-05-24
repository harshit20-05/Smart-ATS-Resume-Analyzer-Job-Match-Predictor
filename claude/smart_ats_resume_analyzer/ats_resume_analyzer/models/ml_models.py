"""
ml_models.py
------------
Phase 7: All Machine Learning Models

Implements:
  1. Logistic Regression   → selection prediction
  2. Naive Bayes           → resume category classification
  3. K-Means Clustering    → resume segmentation
  4. Hierarchical Clustering
  5. Linear / Multiple Regression → ATS score prediction
  6. PCA (see feature_engineering.py for standalone PCA)

Each model section includes:
  - Training
  - Evaluation metrics
  - Confusion matrix
  - ROC curve
  - Save/Load with pickle
"""

import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection  import train_test_split
from sklearn.metrics          import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, auc, precision_score, recall_score, f1_score
)


# ══════════════════════════════════════════════════════════
# 1.  LOGISTIC REGRESSION
# ══════════════════════════════════════════════════════════
from sklearn.linear_model import LogisticRegression

def train_logistic_regression(X, y, test_size: float = 0.2):
    """
    Logistic Regression for binary classification.

    The model predicts P(selected=1 | features):
      P(y=1) = σ(β₀ + β₁x₁ + ... + βₙxₙ)
      where σ(z) = 1 / (1 + e^{-z})   ← sigmoid function

    A candidate is predicted selected when P(y=1) > 0.5.

    Returns dict with model, split data, and all metrics.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    model = LogisticRegression(
        max_iter   = 1000,
        solver     = "lbfgs",
        C          = 1.0,            # regularisation strength
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred     = model.predict(X_test)
    y_prob     = model.predict_proba(X_test)[:, 1]  # prob of class=1

    accuracy   = accuracy_score(y_test, y_pred)
    precision  = precision_score(y_test, y_pred, zero_division=0)
    recall     = recall_score(y_test, y_pred, zero_division=0)
    f1         = f1_score(y_test, y_pred, zero_division=0)
    cm         = confusion_matrix(y_test, y_pred)
    report     = classification_report(y_test, y_pred)

    print("=" * 50)
    print("LOGISTIC REGRESSION RESULTS")
    print("=" * 50)
    print(f"  Accuracy  : {accuracy:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"\nClassification Report:\n{report}")

    return {
        "model"    : model,
        "X_train"  : X_train, "X_test" : X_test,
        "y_train"  : y_train, "y_test" : y_test,
        "y_pred"   : y_pred,  "y_prob" : y_prob,
        "accuracy" : accuracy, "precision": precision,
        "recall"   : recall,  "f1"     : f1,
        "cm"       : cm,      "report" : report,
    }


def plot_confusion_matrix(cm: np.ndarray,
                           class_names=None,
                           title: str = "Confusion Matrix",
                           save_path: str = None) -> None:
    """
    Confusion Matrix explained:
      TP | FP
      --------
      FN | TN

      Precision = TP / (TP + FP)  → accuracy of positive predictions
      Recall    = TP / (TP + FN)  → fraction of positives caught
      F1        = 2 * P * R / (P + R) → harmonic mean
    """
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names or ["Not Selected", "Selected"],
                yticklabels=class_names or ["Not Selected", "Selected"])
    plt.title(title)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_roc_curve(y_test, y_prob,
                   title: str = "ROC Curve",
                   save_path: str = None) -> float:
    """
    ROC Curve: plots TPR vs FPR at every decision threshold.

      TPR (Recall)   = TP / (TP + FN)
      FPR (Fall-out) = FP / (FP + TN)

    AUC = Area Under Curve.
      AUC = 0.5 → random guessing
      AUC = 1.0 → perfect model

    A good model has AUC > 0.8.
    """
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc     = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="steelblue", lw=2,
             label=f"ROC Curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return roc_auc


# ══════════════════════════════════════════════════════════
# 2.  NAIVE BAYES
# ══════════════════════════════════════════════════════════
from sklearn.naive_bayes import MultinomialNB, BernoulliNB

def train_naive_bayes(X, y, variant: str = "multinomial") -> dict:
    """
    Naive Bayes Text Classifier

    Bayes' Theorem:
      P(Class | words) = P(words | Class) × P(Class)
                         ────────────────────────────
                                P(words)

    "Naive" assumption: each word is conditionally independent given the class.
    This is not strictly true but works very well for text classification.

    MultinomialNB → best for TF-IDF / raw count features (resume text)
    BernoulliNB   → best for binary presence/absence features

    Example:
      Resume contains "python", "pandas" → P(Data Science | these words) ≈ 0.68

    variant: "multinomial" or "bernoulli"
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if variant == "bernoulli":
        model = BernoulliNB()
    else:
        model = MultinomialNB(alpha=1.0)   # alpha=1 → Laplace smoothing

    model.fit(X_train, y_train)
    y_pred    = model.predict(X_test)
    accuracy  = accuracy_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n[{variant.upper()} NB]  Accuracy: {accuracy:.4f}  |  F1: {f1:.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))

    return {
        "model": model, "y_test": y_test, "y_pred": y_pred,
        "accuracy": accuracy, "f1": f1,
        "X_train": X_train, "X_test": X_test,
    }


def compare_naive_bayes(X, y) -> dict:
    """Train both variants and return the better one."""
    print("=== Naive Bayes Comparison ===")
    mnb = train_naive_bayes(X, y, variant="multinomial")
    bnb = train_naive_bayes(X, y, variant="bernoulli")

    best = mnb if mnb["accuracy"] >= bnb["accuracy"] else bnb
    print(f"\nBest variant: {'Multinomial' if best is mnb else 'Bernoulli'}"
          f"  (acc={best['accuracy']:.4f})")
    return best


# ══════════════════════════════════════════════════════════
# 3.  K-MEANS CLUSTERING
# ══════════════════════════════════════════════════════════
from sklearn.cluster import KMeans

def find_optimal_k(X, k_range=range(2, 11),
                   save_path: str = None) -> int:
    """
    Elbow Method — finds optimal K by plotting within-cluster
    sum of squares (WCSS / inertia) vs. K.

    Cost function:
      J = Σ_i Σ_{x ∈ Cᵢ} ||x − μᵢ||²
      where μᵢ = centroid of cluster i

    The "elbow" is where adding more clusters gives diminishing returns.
    """
    wcss = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        wcss.append(km.inertia_)

    plt.figure(figsize=(7, 4))
    plt.plot(list(k_range), wcss, "bo-", linewidth=2, markersize=8)
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("WCSS (Inertia)")
    plt.title("Elbow Method — Optimal K")
    plt.xticks(list(k_range))
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    # Find elbow point automatically using max curvature
    diffs  = np.diff(wcss)
    diffs2 = np.diff(diffs)
    optimal_k = list(k_range)[np.argmax(diffs2) + 2]
    print(f"[K-Means]  Suggested optimal K = {optimal_k}")
    return optimal_k


def train_kmeans(X, n_clusters: int = 5) -> dict:
    """
    K-Means clustering:
      1. Initialise K centroids randomly (k-means++ strategy)
      2. Assign each point to nearest centroid
      3. Recompute centroids as cluster mean
      4. Repeat until convergence

    Euclidean distance: d(x, μ) = √Σ(xᵢ − μᵢ)²
    """
    model = KMeans(n_clusters=n_clusters, random_state=42,
                   n_init=10, max_iter=300)
    labels = model.fit_predict(X)
    inertia = model.inertia_

    print(f"[K-Means]  K={n_clusters}  |  Inertia={inertia:.2f}")
    for i in range(n_clusters):
        print(f"  Cluster {i}: {np.sum(labels == i)} samples")

    return {"model": model, "labels": labels, "inertia": inertia}


def visualise_clusters(X_pca: np.ndarray, labels: np.ndarray,
                        save_path: str = None) -> None:
    """Plot 2D PCA projection coloured by cluster label."""
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1],
                          c=labels, cmap="Set2", s=40, alpha=0.7)
    plt.colorbar(scatter, label="Cluster")
    plt.title("K-Means Clusters (PCA 2D Projection)")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ══════════════════════════════════════════════════════════
# 4.  HIERARCHICAL CLUSTERING
# ══════════════════════════════════════════════════════════
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster         import AgglomerativeClustering

def plot_dendrogram(X, method: str = "ward",
                    sample_size: int = 100,
                    save_path: str = None) -> None:
    """
    Hierarchical Agglomerative Clustering dendrogram.

    Linkage methods:
      ward   → minimises variance within merged clusters (default, best)
      single → minimum distance between any two points
      complete → maximum distance between any two points
      average → average of all pairwise distances

    The dendrogram is cut at the longest vertical line to get clusters.
    """
    # Use a subsample for readability
    if X.shape[0] > sample_size:
        idx = np.random.choice(X.shape[0], size=sample_size, replace=False)
        X_sample = X[idx] if not hasattr(X, "toarray") else X.toarray()[idx]
    else:
        X_sample = X.toarray() if hasattr(X, "toarray") else X

    Z = linkage(X_sample, method=method)

    plt.figure(figsize=(14, 5))
    dendrogram(Z, no_labels=True, color_threshold=0.7 * max(Z[:, 2]))
    plt.title(f"Hierarchical Clustering Dendrogram (method={method})")
    plt.xlabel("Resume samples")
    plt.ylabel("Distance")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def train_hierarchical(X, n_clusters: int = 5) -> np.ndarray:
    """Fit AgglomerativeClustering and return labels."""
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    X_dense = X.toarray() if hasattr(X, "toarray") else X
    labels = model.fit_predict(X_dense)
    print(f"[Hierarchical]  K={n_clusters}  |  Unique labels: {np.unique(labels)}")
    return labels


# ══════════════════════════════════════════════════════════
# 5.  LINEAR / MULTIPLE REGRESSION  (ATS Score Prediction)
# ══════════════════════════════════════════════════════════
from sklearn.linear_model import LinearRegression
from sklearn.metrics      import mean_squared_error, r2_score

def train_linear_regression(X, y) -> dict:
    """
    Multiple Linear Regression to predict continuous ATS score.

    Model:  ŷ = β₀ + β₁x₁ + β₂x₂ + … + βₙxₙ
    Fit by minimising RSS = Σ(yᵢ − ŷᵢ)²

    Features (x): skill count, experience years, keyword density, ...
    Target  (y):  ATS compatibility score [0–100]
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f"[LinearReg]  R²={r2:.4f}  |  RMSE={rmse:.4f}")

    plt.figure(figsize=(6, 5))
    plt.scatter(y_test, y_pred, alpha=0.5, color="steelblue")
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()], "r--", lw=2)
    plt.xlabel("Actual ATS Score")
    plt.ylabel("Predicted ATS Score")
    plt.title(f"Linear Regression — Actual vs Predicted (R²={r2:.3f})")
    plt.tight_layout()
    plt.show()

    return {"model": model, "r2": r2, "rmse": rmse,
            "y_test": y_test, "y_pred": y_pred}


# ══════════════════════════════════════════════════════════
# PICKLE — Save / Load
# ══════════════════════════════════════════════════════════
def save_model(obj, filename: str, folder: str = "models") -> str:
    """Persist any sklearn model or vectorizer to disk."""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"[saved]  -> {path}")
    return path


def load_model(filename: str, folder: str = "models"):
    """Load a pickled model from disk."""
    path = os.path.join(folder, filename)
    with open(path, "rb") as f:
        obj = pickle.load(f)
    print(f"[loaded]  <- {path}")
    return obj
