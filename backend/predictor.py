"""
Model inference manager for loading artifacts and performing predictions.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List

class ModelPredictor:
    def __init__(self, models_dir: str = None):
        if models_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            models_dir = os.path.join(base_dir, "models")
        
        self.models_dir = models_dir
        self.kmeans = None
        self.scaler = None
        self.pca_2d = None
        self.pca_3d = None
        self.feature_columns = []
        self.profiles = {}
        self.is_loaded = False

        self.load_artifacts()

    def load_artifacts(self):
        try:
            kmeans_path = os.path.join(self.models_dir, "clustering_model.pkl")
            scaler_path = os.path.join(self.models_dir, "scaler.pkl")
            pca_path = os.path.join(self.models_dir, "pca.pkl")
            pca_3d_path = os.path.join(self.models_dir, "pca_3d.pkl")
            cols_path = os.path.join(self.models_dir, "feature_columns.pkl")
            profiles_path = os.path.join(self.models_dir, "cluster_profiles.json")

            if os.path.exists(kmeans_path):
                self.kmeans = joblib.load(kmeans_path)
                self.scaler = joblib.load(scaler_path)
                self.pca_2d = joblib.load(pca_path)
                if os.path.exists(pca_3d_path):
                    self.pca_3d = joblib.load(pca_3d_path)
                self.feature_columns = joblib.load(cols_path)
                
                with open(profiles_path, "r") as f:
                    self.profiles = json.load(f)

                self.is_loaded = True
                print("Model artifacts successfully loaded.")
            else:
                print(f"Warning: Artifacts not yet trained at {self.models_dir}.")
        except Exception as e:
            print(f"Error loading model artifacts: {e}")
            self.is_loaded = False

    def predict_single(self, feature_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Scales input row, predicts cluster, computes Euclidean distance to centroid,
        and retrieves profile metadata.
        """
        if not self.is_loaded:
            raise RuntimeError("Model artifacts not loaded.")

        X = feature_df[self.feature_columns]
        X_scaled = self.scaler.transform(X)

        cluster_id = int(self.kmeans.predict(X_scaled)[0])
        centroid = self.kmeans.cluster_centers_[cluster_id]
        distance = float(np.linalg.norm(X_scaled[0] - centroid))

        pca_coords = self.pca_2d.transform(X_scaled)[0]

        profile = self.profiles.get(str(cluster_id), {
            "segment_name": f"Cluster {cluster_id}",
            "tier": "Standard",
            "description": "Behavioral cluster",
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

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes batch CSV dataframe and returns annotated results.
        """
        if not self.is_loaded:
            raise RuntimeError("Model artifacts not loaded.")

        # Ensure features are present
        data = df.copy()
        if "Recency_Days" not in data.columns and "Days_Since_Last_Inquiry" in data.columns:
            data["Recency_Days"] = data["Days_Since_Last_Inquiry"]
        if "Frequency_Per_Month" not in data.columns:
            tenure = data["Tenure_Months"] if "Tenure_Months" in data.columns else 1.0
            data["Frequency_Per_Month"] = data["Total_Bookings"] / np.maximum(tenure, 1.0)
        if "Avg_Booking_Value_INR" not in data.columns and "Total_Spend_INR" in data.columns:
            orders = data["Total_Bookings"] if "Total_Bookings" in data.columns else 1
            data["Avg_Booking_Value_INR"] = data["Total_Spend_INR"] / np.maximum(orders, 1)
        if "Engagement_Score" not in data.columns:
            data["Engagement_Score"] = (
                (data.get("Website_Visits", 10) * 0.20) +
                (data.get("App_Sessions", 5) * 0.30) +
                (data.get("WhatsApp_Inquiries", 2) * 0.35) +
                (data.get("Wishlist_Count", 2) * 0.15)
            )
        if "Discount_Dependency" not in data.columns and "Discount_Usage_Ratio" in data.columns:
            data["Discount_Dependency"] = data["Discount_Usage_Ratio"]

        X = data[self.feature_columns]
        X_scaled = self.scaler.transform(X)

        clusters = self.kmeans.predict(X_scaled)
        data["Cluster"] = clusters

        # Distance from centroid
        centroids = self.kmeans.cluster_centers_
        distances = [
            float(np.linalg.norm(X_scaled[i] - centroids[clusters[i]]))
            for i in range(len(X_scaled))
        ]
        data["Distance_From_Centroid"] = np.round(distances, 4)

        segment_map = {int(k): v["segment_name"] for k, v in self.profiles.items()}
        data["Segment_Name"] = data["Cluster"].map(segment_map)

        return data

# Global predictor instance
predictor = ModelPredictor()
