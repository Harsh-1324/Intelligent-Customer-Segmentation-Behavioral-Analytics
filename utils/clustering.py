"""
Clustering utilities for prediction, distance calculations, and pipeline helpers.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

def predict_single_customer(
    customer_features_df: pd.DataFrame,
    scaler,
    kmeans_model,
    pca_2d,
    feature_columns: list,
    profiles: dict
) -> Dict[str, Any]:
    """
    Transforms a single customer input, scales it, predicts cluster, 
    computes distance to centroid, and attaches profile metadata.
    """
    X = customer_features_df[feature_columns]
    X_scaled = scaler.transform(X)
    
    cluster_id = int(kmeans_model.predict(X_scaled)[0])
    centroid = kmeans_model.cluster_centers_[cluster_id]
    distance = float(np.linalg.norm(X_scaled[0] - centroid))

    pca_coords = pca_2d.transform(X_scaled)[0]
    
    profile = profiles.get(str(cluster_id), {
        "segment_name": f"Cluster {cluster_id}",
        "tier": "Standard",
        "description": "Customer cluster",
        "strategies": []
    })

    return {
        "cluster": cluster_id,
        "segment_name": profile.get("segment_name"),
        "tier": profile.get("tier"),
        "distance_from_centroid": round(distance, 4),
        "pca_coordinates": {
            "pca_1": round(float(pca_coords[0]), 4),
            "pca_2": round(float(pca_coords[1]), 4)
        },
        "description": profile.get("description"),
        "recommended_strategies": profile.get("strategies", [])
    }
