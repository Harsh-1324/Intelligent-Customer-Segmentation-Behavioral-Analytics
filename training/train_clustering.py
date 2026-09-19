"""
Main training pipeline: executes feature engineering, unsupervised clustering (K-Means, DBSCAN, Agglomerative),
evaluates metrics, fits PCA, generates cluster profiles, and exports all production artifacts.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# Add current directory to path for clean relative imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing import (
    engineer_features,
    CLUSTERING_FEATURES,
    fit_scaler,
    fit_pca
)
from evaluate_clusters import (
    compute_clustering_metrics,
    compute_elbow_and_silhouette_range,
    benchmark_clustering_algorithms
)
from cluster_profiling import generate_cluster_profiles

def run_training_pipeline():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "customer_data.csv")
    models_dir = os.path.join(base_dir, "models")
    outputs_dir = os.path.join(base_dir, "outputs")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    print(f"1. Loading dataset from: {data_path}")
    raw_df = pd.read_csv(data_path)
    print(f"   Loaded {len(raw_df)} rows and {len(raw_df.columns)} columns.")

    print("2. Engineering behavioral and RFM features...")
    featured_df = engineer_features(raw_df)
    X = featured_df[CLUSTERING_FEATURES]
    print(f"   Selected {len(CLUSTERING_FEATURES)} core clustering features: {CLUSTERING_FEATURES}")

    print("3. Fitting StandardScaler...")
    scaler = fit_scaler(X)
    X_scaled = scaler.transform(X)

    print("4. Evaluating K-Means across range K=2..8 for Elbow & Silhouette analysis...")
    eval_curves = compute_elbow_and_silhouette_range(X_scaled, k_min=2, k_max=8)
    with open(os.path.join(models_dir, "evaluation_curves.json"), "w") as f:
        json.dump(eval_curves, f, indent=2)

    optimal_k = 5
    print(f"5. Training primary K-Means model with K={optimal_k}...")
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=15)
    cluster_labels = kmeans.fit_predict(X_scaled)
    featured_df["Cluster"] = cluster_labels

    print("6. Computing distances from cluster centroids...")
    centroids = kmeans.cluster_centers_
    distances = [
        np.linalg.norm(X_scaled[i] - centroids[cluster_labels[i]])
        for i in range(len(X_scaled))
    ]
    featured_df["Distance_From_Centroid"] = np.round(distances, 4)

    print("7. Fitting Dimensionality Reduction (PCA 2D and 3D)...")
    pca_2d = fit_pca(X_scaled, n_components=2)
    pca_3d = fit_pca(X_scaled, n_components=3)

    coords_2d = pca_2d.transform(X_scaled)
    coords_3d = pca_3d.transform(X_scaled)

    featured_df["PCA_1"] = np.round(coords_2d[:, 0], 4)
    featured_df["PCA_2"] = np.round(coords_2d[:, 1], 4)
    featured_df["PCA_3D_1"] = np.round(coords_3d[:, 0], 4)
    featured_df["PCA_3D_2"] = np.round(coords_3d[:, 1], 4)
    featured_df["PCA_3D_3"] = np.round(coords_3d[:, 2], 4)

    print("8. Profiling clusters and generating business segment mappings...")
    profiles = generate_cluster_profiles(featured_df, cluster_col="Cluster")
    
    segment_map = {int(k): v["segment_name"] for k, v in profiles.items()}
    featured_df["Segment_Name"] = featured_df["Cluster"].map(segment_map)

    with open(os.path.join(models_dir, "cluster_profiles.json"), "w") as f:
        json.dump(profiles, f, indent=2)

    print("9. Running algorithm benchmarking (K-Means vs DBSCAN vs Agglomerative)...")
    benchmark_df = benchmark_clustering_algorithms(X_scaled, k_optimal=optimal_k)
    benchmark_path = os.path.join(models_dir, "algorithm_comparison.json")
    benchmark_df.to_json(benchmark_path, orient="records", indent=2)
    print(benchmark_df.to_string(index=False))

    print("10. Exporting model artifacts...")
    joblib.dump(kmeans, os.path.join(models_dir, "clustering_model.pkl"))
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
    joblib.dump(pca_2d, os.path.join(models_dir, "pca.pkl"))
    joblib.dump(pca_3d, os.path.join(models_dir, "pca_3d.pkl"))
    joblib.dump(CLUSTERING_FEATURES, os.path.join(models_dir, "feature_columns.pkl"))

    output_csv = os.path.join(outputs_dir, "segmented_customers.csv")
    featured_df.to_csv(output_csv, index=False)
    print(f"   Annotated dataset saved to: {output_csv}")
    print("\nTraining completed successfully! All artifacts ready for production.")

if __name__ == "__main__":
    run_training_pipeline()
