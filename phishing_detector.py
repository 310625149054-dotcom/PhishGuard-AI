"""
PhishGuard AI - Model Training & Evaluation Pipeline
----------------------------------------------------
Loads the Phishing Email dataset, performs stratified train/test split,
extracts TF-IDF and handcrafted URL/keyword features using features.py,
compares Logistic Regression and Multinomial Naive Bayes models,
evaluates performance, generates confusion matrix visualization,
and serializes artifacts into model/.
"""

import os
import json
import time
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')  # Headless backend for server/script execution
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from scipy.sparse import hstack, csr_matrix

from features import clean_email_text, extract_batch_tabular_features


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "Phishing_Email.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
STATIC_DIR = os.path.join(BASE_DIR, "static")
CONFUSION_MATRIX_ROOT = os.path.join(BASE_DIR, "confusion_matrix.png")
CONFUSION_MATRIX_STATIC = os.path.join(STATIC_DIR, "confusion_matrix.png")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")


def load_and_clean_data(csv_path: str):
    """
    Loads dataset, handles missing/empty values, standardizes labels,
    and returns cleaned texts and binary targets (0: Safe, 1: Phishing).
    """
    print(f"[*] Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Identify text and label columns
    # In Phishing_Email.csv columns are: 'Unnamed: 0', 'Email Text', 'Email Type'
    text_col = 'Email Text' if 'Email Text' in df.columns else df.columns[1]
    label_col = 'Email Type' if 'Email Type' in df.columns else df.columns[2]
    
    initial_count = len(df)
    # Drop rows where text is null
    df = df.dropna(subset=[text_col, label_col]).copy()
    
    # Map label to binary: Safe -> 0, Phishing -> 1
    # Check distinct values
    unique_labels = df[label_col].unique()
    print(f"[*] Raw labels found: {unique_labels}")
    
    def map_label(val):
        s = str(val).lower().strip()
        if "phish" in s or "spam" in s or s == "1":
            return 1
        return 0
        
    df['target'] = df[label_col].apply(map_label)
    
    # Preprocess text
    print("[*] Preprocessing email text bodies...")
    df['cleaned_text'] = df[text_col].apply(clean_email_text)
    
    # Filter out empty texts after cleaning
    df = df[df['cleaned_text'].str.len() > 5].copy()
    
    print(f"[*] Dataset cleaned: {len(df)} valid records (dropped {initial_count - len(df)} empty/invalid rows).")
    class_counts = df['target'].value_counts().to_dict()
    print(f"[*] Class distribution: Safe (0) = {class_counts.get(0, 0)}, Phishing (1) = {class_counts.get(1, 0)}")
    
    return df['cleaned_text'].tolist(), df['target'].to_numpy(), class_counts


def plot_confusion_matrix(cm, labels, save_paths, accuracy, f1):
    """Generates an aesthetic cybersecurity-themed confusion matrix plot."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=150)
    
    # Modern dark background styling
    fig.patch.set_facecolor('#0d1322')
    ax.set_facecolor('#0d1322')
    
    cmap = plt.cm.Blues
    cax = ax.matshow(cm, cmap=cmap, alpha=0.85)
    
    # Annotate numbers
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            pct = (val / np.sum(cm)) * 100
            color = "white" if val > (cm.max() / 2) else "#94a3b8"
            ax.text(j, i, f"{val:,}\n({pct:.1f}%)", ha="center", va="center",
                    color=color, fontsize=12, fontweight="bold")
            
    # Ticks and labels
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=11, color="#cbd5e1", fontweight="600")
    ax.set_yticklabels(labels, fontsize=11, color="#cbd5e1", fontweight="600")
    
    ax.tick_params(axis='both', which='both', length=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#1e293b')
    ax.spines['left'].set_color('#1e293b')
    
    ax.set_xlabel("Predicted Label", fontsize=12, color="#06b6d4", fontweight="600", labelpad=12)
    ax.set_ylabel("True Label", fontsize=12, color="#06b6d4", fontweight="600", labelpad=12)
    
    plt.title(f"PhishGuard AI — Confusion Matrix\nAccuracy: {accuracy*100:.2f}% | Phishing F1: {f1*100:.2f}%",
              fontsize=13, color="#f8fafc", fontweight="bold", pad=20)
    
    plt.tight_layout()
    for path in save_paths:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        plt.savefig(path, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"[*] Confusion matrix saved to: {save_paths}")


def train_and_evaluate():
    """Main training, comparison, and serialization pipeline."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(STATIC_DIR, exist_ok=True)
    
    # 1. Load data
    texts, labels, class_distribution = load_and_clean_data(DATA_PATH)
    
    # 2. Stratified train/test split (80% train, 20% test)
    print("[*] Performing stratified 80/20 train/test split (random_state=42)...")
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        texts, labels, test_size=0.20, random_state=42, stratify=labels
    )
    
    print(f"[*] Training samples: {len(X_train_text)} | Test samples: {len(X_test_text)}")
    
    # 3. TF-IDF Feature Extraction
    print("[*] Fitting TF-IDF Vectorizer (unigrams + bigrams, max_features=8000)...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=8000,
        sublinear_tf=True,
        min_df=2
    )
    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    X_test_tfidf = vectorizer.transform(X_test_text)
    
    # 4. Tabular URL & Keyword Features
    print("[*] Extracting handcrafted URL and security keyword features...")
    X_train_tab = extract_batch_tabular_features(X_train_text)
    X_test_tab = extract_batch_tabular_features(X_test_text)
    
    # Scale tabular features with MaxAbsScaler (preserves sparsity and non-negativity)
    scaler = MaxAbsScaler()
    X_train_tab_scaled = scaler.fit_transform(X_train_tab)
    X_test_tab_scaled = scaler.transform(X_test_tab)
    
    # Combine sparse TF-IDF with tabular features
    X_train_combined = hstack([X_train_tfidf, csr_matrix(X_train_tab_scaled)]).tocsr()
    X_test_combined = hstack([X_test_tfidf, csr_matrix(X_test_tab_scaled)]).tocsr()
    print(f"[*] Combined feature matrix shape: {X_train_combined.shape}")
    
    # 5. Model Candidate Comparison
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=2.0, random_state=42, solver='lbfgs'),
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.1)
    }
    
    best_model_name = None
    best_model = None
    best_f1 = -1.0
    model_evaluations = {}
    
    for name, model in models.items():
        print(f"\n[+] Training {name}...")
        t0 = time.time()
        model.fit(X_train_combined, y_train)
        train_time = time.time() - t0
        
        preds = model.predict(X_test_combined)
        proba = model.predict_proba(X_test_combined)[:, 1] if hasattr(model, "predict_proba") else None
        
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, pos_label=1)
        rec = recall_score(y_test, preds, pos_label=1)
        f1 = f1_score(y_test, preds, pos_label=1)
        
        print(f"    - Accuracy : {acc*100:.2f}%")
        print(f"    - Precision: {prec*100:.2f}% (Phishing class)")
        print(f"    - Recall   : {rec*100:.2f}% (Phishing class)")
        print(f"    - F1-Score : {f1*100:.2f}% (Phishing class)")
        print(f"    - Training time: {train_time:.2f}s")
        
        model_evaluations[name] = {
            "accuracy": round(float(acc) * 100, 2),
            "precision": round(float(prec) * 100, 2),
            "recall": round(float(rec) * 100, 2),
            "f1_score": round(float(f1) * 100, 2),
        }
        
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model = model

    print(f"\n[*] Selected Champion Model: {best_model_name} (F1-score: {best_f1*100:.2f}%)")
    
    # Final evaluation on the test set
    final_preds = best_model.predict(X_test_combined)
    final_acc = accuracy_score(y_test, final_preds)
    final_prec = precision_score(y_test, final_preds, pos_label=1)
    final_rec = recall_score(y_test, final_preds, pos_label=1)
    final_f1 = f1_score(y_test, final_preds, pos_label=1)
    
    cm = confusion_matrix(y_test, final_preds)
    # cm layout: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = cm.ravel()
    
    print("\n--- Confusion Matrix ---")
    print(f"True Negatives  (Safe correctly identified)    : {tn}")
    print(f"False Positives (Safe flagged as Phishing)     : {fp}")
    print(f"False Negatives (Phishing missed - HIGH RISK!) : {fn}")
    print(f"True Positives  (Phishing caught successfully) : {tp}")
    
    clf_report = classification_report(
        y_test, final_preds, target_names=["Safe", "Phishing"], output_dict=True
    )
    
    # 6. Save Confusion Matrix Plot
    plot_confusion_matrix(
        cm, ["Safe", "Phishing"],
        [CONFUSION_MATRIX_ROOT, CONFUSION_MATRIX_STATIC],
        final_acc, final_f1
    )
    
    # 7. Save Metrics JSON
    metrics_data = {
        "model_name": best_model_name,
        "comparison": model_evaluations,
        "accuracy": round(float(final_acc) * 100, 2),
        "precision": round(float(final_prec) * 100, 2),
        "recall": round(float(final_rec) * 100, 2),
        "f1_score": round(float(final_f1) * 100, 2),
        "train_samples": int(len(X_train_text)),
        "test_samples": int(len(X_test_text)),
        "class_distribution": {
            "safe": int(class_distribution.get(0, 0)),
            "phishing": int(class_distribution.get(1, 0))
        },
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        },
        "classification_report": clf_report,
        "security_notes": {
            "false_negative_risk": "In email cybersecurity, False Negatives (FN) represent the highest risk: a malicious email is delivered to the user inbox labelled as 'Safe'. Our feature engineering deliberately balances precision and recall to minimize FN while maintaining low FP alarm fatigue."
        }
    }
    
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"[*] Metrics saved to: {METRICS_PATH}")
    
    # 8. Save Model, Vectorizer, and Scaler Artifacts
    model_save_path = os.path.join(MODEL_DIR, "phishing_model.joblib")
    vectorizer_save_path = os.path.join(MODEL_DIR, "vectorizer.joblib")
    scaler_save_path = os.path.join(MODEL_DIR, "scaler.joblib")
    
    joblib.dump(best_model, model_save_path)
    joblib.dump(vectorizer, vectorizer_save_path)
    joblib.dump(scaler, scaler_save_path)
    
    # Also save standard aliases for compatibility
    joblib.dump(best_model, os.path.join(MODEL_DIR, "model.joblib"))
    
    print(f"[*] Model artifacts successfully serialized into {MODEL_DIR}/:")
    print(f"    - {model_save_path}")
    print(f"    - {vectorizer_save_path}")
    print(f"    - {scaler_save_path}")
    print("[OK] Model training and evaluation successfully completed!")


if __name__ == "__main__":
    train_and_evaluate()
