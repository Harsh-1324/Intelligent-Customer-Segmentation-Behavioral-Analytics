"""
Data preprocessing, behavioral feature engineering, and scaling pipeline for customer segmentation.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

NUMERICAL_FEATURE_COLUMNS = [
    "Age",
    "Tenure_Months",
    "Total_Bookings",
    "Total_Spend_INR",
    "Avg_Booking_Value_INR",
    "Days_Since_Last_Inquiry",
    "Discount_Usage_Ratio",
    "Website_Visits",
    "App_Sessions",
    "WhatsApp_Inquiries",
    "Cart_Shortlist_Additions",
    "Abandonment_Rate",
    "Wishlist_Count",
    "Support_Interactions",
    "Recency_Days",
    "Frequency_Per_Month",
    "Engagement_Score",
    "Discount_Dependency",
    "Shortlist_To_Booking_Ratio"
]

CLUSTERING_FEATURES = [
    "Recency_Days",
    "Frequency_Per_Month",
    "Total_Spend_INR",
    "Avg_Booking_Value_INR",
    "Engagement_Score",
    "Discount_Dependency",
    "Abandonment_Rate",
    "Cart_Shortlist_Additions"
]

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derives behavioral RFM and engagement features from raw customer interaction data.
    """
    data = df.copy()

    # 1. Recency: Days since last interaction
    data["Recency_Days"] = data["Days_Since_Last_Inquiry"]

    # 2. Frequency: Bookings normalized by tenure in months
    safe_tenure = np.maximum(data["Tenure_Months"], 1)
    data["Frequency_Per_Month"] = np.round(data["Total_Bookings"] / safe_tenure, 3)

    # 3. Monetary value & Average Booking Value
    if "Avg_Booking_Value_INR" not in data.columns:
        safe_orders = np.maximum(data["Total_Bookings"], 1)
        data["Avg_Booking_Value_INR"] = np.round(data["Total_Spend_INR"] / safe_orders, 2)

    # 4. Composite Engagement Score (Weighted combination of behavioral signals)
    data["Engagement_Score"] = np.round(
        (data["Website_Visits"] * 0.20) +
        (data["App_Sessions"] * 0.30) +
        (data["WhatsApp_Inquiries"] * 0.35) +
        (data["Wishlist_Count"] * 0.15),
        2
    )

    # 5. Discount Dependency
    data["Discount_Dependency"] = np.round(data["Discount_Usage_Ratio"], 3)

    # 6. Shortlist-to-Booking Ratio (Conversion efficacy)
    safe_shortlist = np.maximum(data["Cart_Shortlist_Additions"], 1)
    data["Shortlist_To_Booking_Ratio"] = np.round(data["Total_Bookings"] / safe_shortlist, 3)

    return data

def fit_scaler(X: pd.DataFrame) -> StandardScaler:
    """Fits StandardScaler on feature matrix."""
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler

def fit_pca(X_scaled: np.ndarray, n_components: int = 2) -> PCA:
    """Fits PCA for dimensionality reduction and 2D/3D visualization."""
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X_scaled)
    return pca
