"""
models/ml_validator.py
----------------------
ML-based data quality checks using:
- KNN: Outlier detection in numerical data
- K-Means: Distribution shift & cluster anomalies
- Decision Tree: Label inconsistency detection
- Naive Bayes: Text/categorical distribution validation
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score
from typing import Dict, Any, List


# ──────────────────────────────────────────────────────────────
# 1. KNN OUTLIER DETECTION
# ──────────────────────────────────────────────────────────────

def knn_outlier_detection(df: pd.DataFrame, contamination: float = 0.05) -> Dict[str, Any]:
    """
    Use KNN distance to detect outliers in numerical columns.
    Points with abnormally large average distance to neighbors
    are flagged as outliers.
    """
    # Select only numeric columns, exclude columns with lists/complex types
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    except Exception:
        num_cols = []

    if len(num_cols) < 1:
        return {
            "check": "knn_outlier_detection",
            "outlier_count": 0,
            "outlier_percentage": 0.0,
            "severity": "none",
            "score_deduction": 0,
            "suggestions": ["No numerical columns found for outlier detection."],
            "outlier_indices": []
        }

    # Drop rows with all-NaN in numeric cols for fitting
    try:
        numeric_df = df[num_cols].dropna()
    except Exception as e:
        return {
            "check": "knn_outlier_detection",
            "outlier_count": 0,
            "outlier_percentage": 0.0,
            "severity": "none",
            "score_deduction": 0,
            "suggestions": [f"Could not process numerical columns: {str(e)}"],
            "outlier_indices": []
        }

    if len(numeric_df) < 5:
        return {
            "check": "knn_outlier_detection",
            "outlier_count": 0,
            "outlier_percentage": 0.0,
            "severity": "none",
            "score_deduction": 0,
            "suggestions": ["Insufficient data for KNN outlier detection."],
            "outlier_indices": []
        }

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_df)

    k = min(5, len(numeric_df) - 1)
    nbrs = NearestNeighbors(n_neighbors=k)
    nbrs.fit(X_scaled)

    distances, _ = nbrs.kneighbors(X_scaled)
    avg_distances = distances.mean(axis=1)

    # Flag points beyond mean + 2*std as outliers
    threshold = avg_distances.mean() + 2 * avg_distances.std()
    outlier_mask = avg_distances > threshold
    outlier_indices = numeric_df.index[outlier_mask].tolist()
    outlier_count = int(outlier_mask.sum())
    outlier_pct = round((outlier_count / len(numeric_df)) * 100, 2)

    severity = "none"
    if outlier_pct > 10:
        severity = "high"
    elif outlier_pct > 3:
        severity = "medium"
    elif outlier_pct > 0:
        severity = "low"

    suggestions = []
    if outlier_count > 0:
        suggestions.append(f"Found {outlier_count} outlier rows. Consider IQR-based capping or removal.")
        suggestions.append("Investigate outliers: they may represent real edge cases or data errors.")
        if outlier_pct > 10:
            suggestions.append("High outlier rate suggests systemic data collection issues.")

    return {
        "check": "knn_outlier_detection",
        "numerical_columns_checked": num_cols,
        "outlier_count": outlier_count,
        "outlier_percentage": outlier_pct,
        "severity": severity,
        "score_deduction": min(outlier_pct * 0.8, 15),
        "suggestions": suggestions,
        "outlier_indices": [int(i) for i in outlier_indices[:20]]  # limit for report
    }


# ──────────────────────────────────────────────────────────────
# 2. K-MEANS CLUSTER ANOMALY & DISTRIBUTION SHIFT
# ──────────────────────────────────────────────────────────────

def kmeans_distribution_check(df: pd.DataFrame, n_clusters: int = 3) -> Dict[str, Any]:
    """
    Use K-Means to detect:
    - Very small/imbalanced clusters (anomaly concentration)
    - Distribution irregularities across feature space
    """
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    except Exception:
        num_cols = []

    if len(num_cols) < 1:
        return {
            "check": "kmeans_distribution",
            "anomaly_cluster_count": 0,
            "severity": "none",
            "score_deduction": 0,
            "cluster_info": [],
            "suggestions": ["No numerical columns found."],
        }

    try:
        numeric_df = df[num_cols].dropna()
    except Exception as e:
        return {
            "check": "kmeans_distribution",
            "anomaly_cluster_count": 0,
            "severity": "none",
            "score_deduction": 0,
            "cluster_info": [],
            "suggestions": [f"Could not process numerical columns: {str(e)}"],
        }
    
    if len(numeric_df) < n_clusters * 2:
        return {
            "check": "kmeans_distribution",
            "anomaly_cluster_count": 0,
            "severity": "none",
            "score_deduction": 0,
            "cluster_info": [],
            "suggestions": ["Insufficient rows for K-Means analysis."],
        }

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_df)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    # Analyze cluster sizes
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    cluster_info = []
    anomaly_clusters = 0

    for cluster_id, count in zip(unique, counts):
        pct = round((count / total) * 100, 2)
        is_anomaly = pct < 5  # Clusters with <5% of data = anomalous
        if is_anomaly:
            anomaly_clusters += 1
        cluster_info.append({
            "cluster_id": int(cluster_id),
            "size": int(count),
            "percentage": pct,
            "flagged_as_anomaly": is_anomaly
        })

    severity = "none"
    if anomaly_clusters >= 2:
        severity = "high"
    elif anomaly_clusters == 1:
        severity = "medium"

    suggestions = []
    if anomaly_clusters > 0:
        suggestions.append(f"{anomaly_clusters} cluster(s) contain very few points — potential anomaly concentration.")
        suggestions.append("Inspect records in small clusters; they may be erroneous or edge cases.")
    else:
        suggestions.append("Data distribution appears balanced across clusters.")

    return {
        "check": "kmeans_distribution",
        "n_clusters": n_clusters,
        "anomaly_cluster_count": anomaly_clusters,
        "cluster_info": cluster_info,
        "severity": severity,
        "score_deduction": anomaly_clusters * 5,
        "suggestions": suggestions
    }


# ──────────────────────────────────────────────────────────────
# 3. DECISION TREE — LABEL INCONSISTENCY
# ──────────────────────────────────────────────────────────────

def decision_tree_label_check(df: pd.DataFrame, label_col: str = None) -> Dict[str, Any]:
    """
    Train a Decision Tree on the dataset features to predict labels.
    Low cross-validation accuracy indicates label inconsistencies.
    """
    # Auto-detect label column
    if label_col is None:
        # Look for common label column names
        candidates = ["label", "target", "class", "y", "output", "outcome"]
        for c in candidates:
            if c in df.columns:
                label_col = c
                break

    if label_col is None or label_col not in df.columns:
        return {
            "check": "decision_tree_label_consistency",
            "label_column": None,
            "consistency_score": None,
            "severity": "none",
            "score_deduction": 0,
            "suggestions": ["No label column detected. Skipping label consistency check."]
        }

    # Prepare features
    feature_cols = [c for c in df.columns if c != label_col]
    df_clean = df.dropna(subset=[label_col])

    X = df_clean[feature_cols].copy()
    y = df_clean[label_col].copy()

    # Encode categoricals
    for col in X.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # Fill remaining NaN with median
    X = X.fillna(X.median(numeric_only=True))

    # Encode labels
    le_y = LabelEncoder()
    y_enc = le_y.fit_transform(y.astype(str))

    if len(X) < 10 or len(np.unique(y_enc)) < 2:
        return {
            "check": "decision_tree_label_consistency",
            "label_column": label_col,
            "consistency_score": None,
            "severity": "none",
            "score_deduction": 0,
            "suggestions": ["Insufficient data or single-class label for Decision Tree check."]
        }

    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    cv_scores = cross_val_score(clf, X, y_enc, cv=min(5, len(X) // 2), scoring="accuracy")
    mean_accuracy = round(float(cv_scores.mean()), 4)

    # Low accuracy with a decision tree means noisy/inconsistent labels
    severity = "none"
    deduction = 0
    if mean_accuracy < 0.60:
        severity = "high"
        deduction = 15
    elif mean_accuracy < 0.75:
        severity = "medium"
        deduction = 8
    elif mean_accuracy < 0.85:
        severity = "low"
        deduction = 3

    suggestions = []
    if mean_accuracy < 0.75:
        suggestions.append(f"Label consistency score is {mean_accuracy:.1%} — suggests noisy or conflicting labels.")
        suggestions.append("Review labeling process; consider re-labeling ambiguous samples.")
        suggestions.append("Use majority vote or crowdsourced labeling to resolve conflicts.")
    else:
        suggestions.append(f"Labels appear consistent (Decision Tree CV accuracy: {mean_accuracy:.1%}).")

    return {
        "check": "decision_tree_label_consistency",
        "label_column": label_col,
        "consistency_score": mean_accuracy,
        "consistency_percentage": round(mean_accuracy * 100, 2),
        "severity": severity,
        "score_deduction": deduction,
        "suggestions": suggestions
    }


# ──────────────────────────────────────────────────────────────
# 4. NAIVE BAYES — FEATURE DISTRIBUTION VALIDATION
# ──────────────────────────────────────────────────────────────

def naive_bayes_distribution_check(df: pd.DataFrame, label_col: str = None) -> Dict[str, Any]:
    """
    Train a Gaussian Naive Bayes on numerical features.
    Very low log-likelihood indicates poor distributional fit —
    signals distribution drift or corruption.
    """
    if label_col is None:
        candidates = ["label", "target", "class", "y", "output", "outcome"]
        for c in candidates:
            if c in df.columns:
                label_col = c
                break

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if label_col in num_cols:
        num_cols.remove(label_col)

    if len(num_cols) < 1 or label_col is None:
        return {
            "check": "naive_bayes_distribution",
            "severity": "none",
            "score_deduction": 0,
            "suggestions": ["Insufficient numerical features for Naive Bayes check."]
        }

    df_clean = df[num_cols + [label_col]].dropna()
    if len(df_clean) < 10:
        return {
            "check": "naive_bayes_distribution",
            "severity": "none",
            "score_deduction": 0,
            "suggestions": ["Insufficient clean rows for Naive Bayes check."]
        }

    X = df_clean[num_cols].values
    le = LabelEncoder()
    y = le.fit_transform(df_clean[label_col].astype(str))

    if len(np.unique(y)) < 2:
        return {
            "check": "naive_bayes_distribution",
            "severity": "none",
            "score_deduction": 0,
            "suggestions": ["Only one class present; cannot train Naive Bayes."]
        }

    gnb = GaussianNB()
    cv_scores = cross_val_score(gnb, X, y, cv=min(5, len(X) // 2), scoring="accuracy")
    nb_accuracy = round(float(cv_scores.mean()), 4)

    # NB accuracy reflects how Gaussian the feature distributions are
    severity = "none"
    deduction = 0
    drift_detected = False

    if nb_accuracy < 0.50:
        severity = "high"
        deduction = 12
        drift_detected = True
    elif nb_accuracy < 0.65:
        severity = "medium"
        deduction = 6
        drift_detected = True

    suggestions = []
    if drift_detected:
        suggestions.append("Feature distributions deviate significantly from Gaussian — possible data drift.")
        suggestions.append("Check if new data was collected under different conditions or from a different population.")
        suggestions.append("Apply normalization (e.g., Box-Cox, log transform) to numerical features.")
    else:
        suggestions.append(f"Feature distributions appear reasonable (NB accuracy: {nb_accuracy:.1%}).")

    return {
        "check": "naive_bayes_distribution",
        "nb_cv_accuracy": nb_accuracy,
        "distribution_drift_detected": drift_detected,
        "severity": severity,
        "score_deduction": deduction,
        "suggestions": suggestions
    }