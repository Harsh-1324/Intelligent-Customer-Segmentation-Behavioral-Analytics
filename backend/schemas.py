"""
Pydantic data schemas for FastAPI request and response validation.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class CustomerInput(BaseModel):
    age: int = Field(..., ge=18, le=100, description="Customer age")
    tenure: float = Field(..., ge=0.0, description="Tenure with platform in months")
    orders: int = Field(..., ge=1, description="Total number of bookings/orders placed")
    total_spend: float = Field(..., ge=0.0, description="Total spend in INR (₹)")
    avg_order_value: Optional[float] = Field(None, description="Average order value in INR (₹)")
    days_since_last_inquiry: int = Field(..., ge=0, description="Days since last booking or inquiry (Recency)")
    discount_usage: float = Field(..., ge=0.0, le=1.0, description="Proportion of orders using discount (0.0 to 1.0)")
    website_visits: int = Field(..., ge=0, description="Number of website visits in period")
    app_sessions: int = Field(0, ge=0, description="Number of mobile app sessions")
    whatsapp_inquiries: int = Field(0, ge=0, description="WhatsApp vendor/concierge inquiries")
    cart_shortlist_additions: int = Field(1, ge=0, description="Venues/vendors shortlisted")
    abandonment_rate: float = Field(0.3, ge=0.0, le=1.0, description="Shortlist/cart abandonment rate")
    wishlist_count: int = Field(0, ge=0, description="Total items in wishlist")
    support_interactions: int = Field(0, ge=0, description="Customer support requests")

    class Config:
        json_schema_extra = {
            "example": {
                "age": 31,
                "tenure": 24.0,
                "orders": 4,
                "total_spend": 2600000.0,
                "avg_order_value": 650000.0,
                "days_since_last_inquiry": 15,
                "discount_usage": 0.05,
                "website_visits": 58,
                "app_sessions": 45,
                "whatsapp_inquiries": 28,
                "cart_shortlist_additions": 18,
                "abandonment_rate": 0.25,
                "wishlist_count": 22,
                "support_interactions": 3
            }
        }

class SegmentationResponse(BaseModel):
    cluster: int
    segment_name: str
    tier: str
    distance_from_centroid: float
    pca_coordinates: Dict[str, float]
    description: str
    recommended_strategies: List[str]

class ClusterMetrics(BaseModel):
    avg_spend_inr: float
    avg_booking_value_inr: float
    avg_orders: float
    avg_frequency_monthly: float
    avg_recency_days: float
    avg_discount_usage: float
    avg_engagement_score: float
    avg_abandonment_rate: float

class ClusterProfileItem(BaseModel):
    cluster_id: int
    segment_name: str
    tier: str
    persona: str
    customer_count: int
    customer_percentage: float
    description: str
    metrics: ClusterMetrics
    strategies: List[str]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    total_clusters: int
    version: str
