"""
Feature preprocessing for the FastAPI backend service.
"""

import numpy as np
import pandas as pd
from backend.schemas import CustomerInput

def transform_customer_input(payload: CustomerInput) -> pd.DataFrame:
    """
    Transforms a single customer input payload into a 1-row DataFrame matching the model's feature schema.
    """
    safe_orders = max(payload.orders, 1)
    aov = payload.avg_order_value if payload.avg_order_value is not None else (payload.total_spend / safe_orders)
    safe_tenure = max(payload.tenure, 1.0)
    freq = payload.orders / safe_tenure

    # Composite engagement score matching training calculation
    eng_score = (
        (payload.website_visits * 0.20) +
        (payload.app_sessions * 0.30) +
        (payload.whatsapp_inquiries * 0.35) +
        (payload.wishlist_count * 0.15)
    )

    df_row = pd.DataFrame([{
        "Recency_Days": float(payload.days_since_last_inquiry),
        "Frequency_Per_Month": round(freq, 3),
        "Total_Spend_INR": float(payload.total_spend),
        "Avg_Booking_Value_INR": round(aov, 2),
        "Engagement_Score": round(eng_score, 2),
        "Discount_Dependency": round(payload.discount_usage, 3),
        "Abandonment_Rate": round(payload.abandonment_rate, 3),
        "Cart_Shortlist_Additions": float(payload.cart_shortlist_additions)
    }])

    return df_row
