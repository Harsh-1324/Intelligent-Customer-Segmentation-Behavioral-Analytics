"""
Cluster evaluation module providing Silhouette Score, Davies-Bouldin Index, 
Calinski-Harabasz Score, and algorithm comparisons.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

def compute_clustering_metrics(X_scaled: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    Computes standard unsupervised clustering evaluation metrics.
    Filters out noise points (label == -1) if DBSCAN was used.
    """
    valid_mask = labels != -1
    n_clusters = len(set(labels[valid_mask]))

    if n_clusters < 2:
        return {
            "n_clusters": n_clusters,
            "silhouette": 0.0,
            "davies_bouldin": 0.0,
            "calinski_harabasz": 0.0
        }

    X_eval = X_scaled[valid_mask]
    labels_eval = labels[valid_mask]

    sample_size = min(len(X_eval), 4000)
    indices = np.random.choice(len(X_eval), size=sample_size, replace=False)

    sil = float(silhouette_score(X_eval[indices], labels_eval[indices]))
    db = float(davies_bouldin_score(X_eval, labels_eval))
    ch = float(calinski_harabasz_score(X_eval, labels_eval))

    return {
        "n_clusters": int(n_clusters),
        "silhouette": round(sil, 4),
        "davies_bouldin": round(db, 4),
        "calinski_harabasz": round(ch, 2)
    }

def compute_elbow_and_silhouette_range(
    X_scaled: np.ndarray, 
    k_min: int = 2, 
    k_max: int = 8
) -> Dict[str, List[Any]]:
    """
    Evaluates K-Means across a range of K values to find optimal cluster count.
    """
    k_values = list(range(k_min, k_max + 1))
    inertias = []
    silhouettes = []

    sample_size = min(len(X_scaled), 3000)
    indices = np.random.choice(len(X_scaled), size=sample_size, replace=False)

    for k in k_values:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(round(float(km.inertia_), 2))
        
        score = silhouette_score(X_scaled[indices], km.labels_[indices])
        silhouettes.append(round(float(score), 4))

    return {
        "k_values": k_values,
        "inertias": inertias,
        "silhouettes": silhouettes
    }

def benchmark_clustering_algorithms(X_scaled: np.ndarray, k_optimal: int = 5) -> pd.DataFrame:
    """
    Benchmarks K-Means, DBSCAN, and Agglomerative Clustering.
    """
    km = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
    km_labels = km.fit_predict(X_scaled)
    km_metrics = compute_clustering_metrics(X_scaled, km_labels)

    db = DBSCAN(eps=1.2, min_samples=15)
    db_labels = db.fit_predict(X_scaled)
    db_metrics = compute_clustering_metrics(X_scaled, db_labels)

    sample_n = min(len(X_scaled), 3000)
    sample_idx = np.random.choice(len(X_scaled), size=sample_n, replace=False)
    agg = AgglomerativeClustering(n_clusters=k_optimal)
    agg_labels = agg.fit_predict(X_scaled[sample_idx])
    agg_metrics = compute_clustering_metrics(X_scaled[sample_idx], agg_labels)

    benchmark_data = [
        {
            "Algorithm": "K-Means",
            "Clusters": km_metrics["n_clusters"],
            "Silhouette Score": km_metrics["silhouette"],
            "Davies-Bouldin Index": km_metrics["davies_bouldin"],
            "Calinski-Harabasz Score": km_metrics["calinski_harabasz"]
        },
        {
            "Algorithm": "DBSCAN",
            "Clusters": db_metrics["n_clusters"],
            "Silhouette Score": db_metrics["silhouette"],
            "Davies-Bouldin Index": db_metrics["davies_bouldin"],
            "Calinski-Harabasz Score": db_metrics["calinski_harabasz"]
        },
        {
            "Algorithm": "Agglomerative Clustering",
            "Clusters": agg_metrics["n_clusters"],
            "Silhouette Score": agg_metrics["silhouette"],
            "Davies-Bouldin Index": agg_metrics["davies_bouldin"],
            "Calinski-Harabasz Score": agg_metrics["calinski_harabasz"]
        }
    ]

    return pd.DataFrame(benchmark_data)
