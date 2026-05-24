"""
main.py
-------
Master training pipeline — runs all phases end to end.

Usage:
  python main.py

What it does:
  1. Generate / load dataset
  2. Preprocess text
  3. Feature engineering (TF-IDF, RFE, PCA)
  4. Train all models
  5. Evaluate and compare models
  6. Save models to disk
  7. Print final comparison table
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure modules are importable
sys.path.insert(0, os.path.dirname(__file__))

from data.data_loader         import generate_synthetic_dataset, explore_dataset
from data.preprocessor        import preprocess_dataframe, encode_labels
from data.feature_engineering import (
    build_tfidf, build_count_vectorizer,
    scale_features, apply_pca, apply_rfe, compute_cosine_similarity
)
from models.ml_models import (
    train_logistic_regression, plot_confusion_matrix, plot_roc_curve,
    train_naive_bayes, compare_naive_bayes,
    find_optimal_k, train_kmeans, visualise_clusters,
    plot_dendrogram, train_hierarchical,
    train_linear_regression,
    save_model
)

os.makedirs("models",            exist_ok=True)
os.makedirs("screenshots",       exist_ok=True)
os.makedirs("data/processed",    exist_ok=True)


# ══════════════════════════════════════════════════════════
# STEP 1 — DATA
# ══════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 1: Data Loading")
print("="*60)
df = generate_synthetic_dataset(n_samples=600)
explore_dataset(df)

# Save raw synthetic data
df.to_csv("data/processed/resume_synthetic.csv", index=False)


# ══════════════════════════════════════════════════════════
# STEP 2 — PREPROCESSING
# ══════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 2: Preprocessing")
print("="*60)
df = preprocess_dataframe(df, text_col="Resume")
df, label_encoder = encode_labels(df, label_col="Category")


# ══════════════════════════════════════════════════════════
# STEP 3 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 3: Feature Engineering")
print("="*60)
corpus = df["Cleaned_Resume"].tolist()
y_class = df["Category_Encoded"].values     # multi-class target
y_binary = df["Selected"].values            # binary target

# TF-IDF for most models
X_tfidf, tfidf_vec = build_tfidf(corpus, max_features=1500)

# CountVec for Naive Bayes (needs non-negative counts)
X_count, count_vec = build_count_vectorizer(corpus, max_features=1500)

# PCA to 2D for visualisation
X_pca, pca_obj = apply_pca(X_tfidf, n_components=2)

# ATS Score features (numeric): skill count + experience
df["Skill_Count"]  = df["Resume"].apply(lambda t: len(t.split(",")))
df["ATS_Score_sim"] = (
    df["Experience"] * 5
    + df["Skill_Count"] * 3
    + np.random.normal(50, 10, len(df))
).clip(10, 100)

X_reg = df[["Experience", "Skill_Count"]].values
y_reg = df["ATS_Score_sim"].values


# ══════════════════════════════════════════════════════════
# STEP 4 — MODELS
# ══════════════════════════════════════════════════════════

# ── 4a. Logistic Regression ──
print("\n" + "="*60)
print("STEP 4a: Logistic Regression (binary — selection prediction)")
print("="*60)
lr_results = train_logistic_regression(X_tfidf, y_binary)
plot_confusion_matrix(lr_results["cm"],
                      class_names=["Not Selected", "Selected"],
                      save_path="screenshots/lr_confusion_matrix.png")
roc_auc = plot_roc_curve(lr_results["y_test"], lr_results["y_prob"],
                          save_path="screenshots/lr_roc_curve.png")
save_model(lr_results["model"], "logistic_regression.pkl")

# ── 4b. Naive Bayes ──
print("\n" + "="*60)
print("STEP 4b: Naive Bayes (resume category classification)")
print("="*60)
nb_best = compare_naive_bayes(X_count, y_class)
save_model(nb_best["model"], "naive_bayes.pkl")

# ── 4c. K-Means Clustering ──
print("\n" + "="*60)
print("STEP 4c: K-Means Clustering")
print("="*60)
optimal_k = find_optimal_k(X_pca, k_range=range(2, 11),
                            save_path="screenshots/kmeans_elbow.png")
km_results = train_kmeans(X_pca, n_clusters=optimal_k)
visualise_clusters(X_pca, km_results["labels"],
                   save_path="screenshots/kmeans_clusters.png")
save_model(km_results["model"], "kmeans.pkl")

# ── 4d. Hierarchical Clustering ──
print("\n" + "="*60)
print("STEP 4d: Hierarchical Clustering")
print("="*60)
plot_dendrogram(X_tfidf, method="ward",
                save_path="screenshots/dendrogram.png")
h_labels = train_hierarchical(X_tfidf, n_clusters=5)

# ── 4e. Linear Regression (ATS score) ──
print("\n" + "="*60)
print("STEP 4e: Linear Regression (ATS score prediction)")
print("="*60)
lr_reg = train_linear_regression(X_reg, y_reg)
save_model(lr_reg["model"], "linear_regression.pkl")


# ══════════════════════════════════════════════════════════
# STEP 5 — SAVE SUPPORTING OBJECTS
# ══════════════════════════════════════════════════════════
save_model(tfidf_vec,     "tfidf_vectorizer.pkl")
save_model(count_vec,     "count_vectorizer.pkl")
save_model(label_encoder, "label_encoder.pkl")
save_model(pca_obj,       "pca.pkl")


# ══════════════════════════════════════════════════════════
# STEP 6 — MODEL COMPARISON TABLE
# ══════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 6: Model Comparison Summary")
print("="*60)

comparison = pd.DataFrame([
    {
        "Model"     : "Logistic Regression",
        "Task"      : "Binary Classification",
        "Accuracy"  : lr_results["accuracy"],
        "Precision" : lr_results["precision"],
        "Recall"    : lr_results["recall"],
        "F1 Score"  : lr_results["f1"],
        "AUC"       : roc_auc,
    },
    {
        "Model"     : "Naive Bayes (best)",
        "Task"      : "Multi-class Classification",
        "Accuracy"  : nb_best["accuracy"],
        "Precision" : "-",
        "Recall"    : "-",
        "F1 Score"  : nb_best["f1"],
        "AUC"       : "-",
    },
    {
        "Model"     : "Linear Regression",
        "Task"      : "Regression (ATS Score)",
        "Accuracy"  : f"R²={lr_reg['r2']:.3f}",
        "Precision" : "-",
        "Recall"    : "-",
        "F1 Score"  : f"RMSE={lr_reg['rmse']:.2f}",
        "AUC"       : "-",
    },
])

print(comparison.to_string(index=False))

# Bar chart comparison
fig, ax = plt.subplots(figsize=(9, 4))
metrics = ["Logistic Regression", "Naive Bayes"]
accs    = [lr_results["accuracy"], nb_best["accuracy"]]
f1s     = [lr_results["f1"], nb_best["f1"]]

x = np.arange(len(metrics))
w = 0.3
ax.bar(x - w/2, [a*100 for a in accs], w, label="Accuracy %", color="steelblue")
ax.bar(x + w/2, [f*100 for f in f1s],  w, label="F1 Score %",  color="coral")
ax.set_xticks(x); ax.set_xticklabels(metrics)
ax.set_ylabel("Score (%)")
ax.set_title("Model Comparison: Accuracy vs F1 Score")
ax.legend(); ax.set_ylim(0, 110)
for i, (a, f) in enumerate(zip(accs, f1s)):
    ax.text(i - w/2, a*100 + 1, f"{a*100:.1f}", ha="center", fontsize=9)
    ax.text(i + w/2, f*100 + 1, f"{f*100:.1f}", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig("screenshots/model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n[SUCCESS] Training complete! All models saved to /models/")
print("[INFO] Run the dashboard with: streamlit run app/streamlit_app.py")
