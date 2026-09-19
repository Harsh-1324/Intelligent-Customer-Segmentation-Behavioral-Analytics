"""
Automated cluster profiling, rule-based segment labeling, and business strategy generator.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd

def generate_cluster_profiles(df: pd.DataFrame, cluster_col: str = "Cluster") -> Dict[str, Any]:
    """
    Computes statistical profiles for each cluster and assigns business labels based on behavioral attributes.
    """
    profiles = {}
    total_customers = len(df)

    median_spend = df["Total_Spend_INR"].median()
    median_freq = df["Frequency_Per_Month"].median()
    median_recency = df["Recency_Days"].median()
    median_discount = df["Discount_Dependency"].median()
    median_engagement = df["Engagement_Score"].median()

    grouped = df.groupby(cluster_col)

    for cluster_id, group in grouped:
        count = len(group)
        pct = round((count / total_customers) * 100, 1)

        mean_spend = float(group["Total_Spend_INR"].mean())
        mean_aov = float(group["Avg_Booking_Value_INR"].mean())
        mean_orders = float(group["Total_Bookings"].mean())
        mean_freq = float(group["Frequency_Per_Month"].mean())
        mean_recency = float(group["Recency_Days"].mean())
        mean_discount = float(group["Discount_Dependency"].mean())
        mean_engagement = float(group["Engagement_Score"].mean())
        mean_abandonment = float(group["Abandonment_Rate"].mean())

        # Business segment labeling based on relative feature distributions
        if mean_spend > median_spend * 2.0 and mean_aov > 300000:
            name = "High-Value Destination Celebrants"
            desc = "Premium customers planning grand celebrations, destination weddings, and luxury banquets. Insensitive to discounts, high conversion when provided bespoke concierge."
            persona = "Luxury & Destination Wedding Planners"
            recommended_strategy = [
                "Assign dedicated VIP Wedding Concierge & private venue tours in Lonavala/Alibaug",
                "Offer premium curated vendor packages (luxury decorators, celebrity photographers)",
                "Provide priority calendar hold on top-tier 5-star banquet lawns",
                "Do not offer blanket discount codes; focus on exclusive value-add amenities"
            ]
            tier = "Platinum"

        elif mean_orders > 6 and mean_freq > median_freq * 1.5:
            name = "Corporate & Recurring Event Planners"
            desc = "B2B and corporate event organizers booking recurring conferences, offsites, and team celebrations with steady volume and high lifetime value."
            persona = "Corporate & High-Frequency Organizers"
            recommended_strategy = [
                "Deploy dedicated account management and B2B corporate billing agreements",
                "Offer multi-event volume rebates and annual retainer packages",
                "Automate recurring vendor invoicing and bulk caterer coordination",
                "Deliver priority customer support SLA (< 15 min response time)"
            ]
            tier = "Gold"

        elif mean_recency > median_recency * 1.6 and mean_engagement < median_engagement * 0.7:
            name = "At-Risk / Lapsed Inactive Planners"
            desc = "Previously engaged customers with no recent inquiries or bookings for 100+ days and declining platform visits."
            persona = "Dormant & Churn-Risk Users"
            recommended_strategy = [
                "Launch automated WhatsApp re-engagement with 'Planning your next celebration?' touchpoint",
                "Offer limited-time reactivation perks (₹5,000 venue credit)",
                "Send seasonal celebration calendars (Diwali, New Year, Wedding Muhurat dates)",
                "Gather quick feedback survey to identify drop-off pain points"
            ]
            tier = "At-Risk"

        elif mean_discount > median_discount * 1.3 or mean_abandonment > 0.65:
            name = "Budget-Conscious Occasional Celebrants"
            desc = "Price-sensitive customers organizing birthdays, intimate parties, and small family gatherings. Highly responsive to flash sales and promo codes."
            persona = "Price-Sensitive & Deal Seekers"
            recommended_strategy = [
                "Target with budget banquet halls, pocket-friendly party lawns, and package deals",
                "Send cart/shortlist abandonment reminders with exclusive 10% coupon codes",
                "Promote bundled starter packages (Venue + DJ + Basic Decor under ₹75,000)",
                "Provide transparent pricing breakdown to reduce hesitation"
            ]
            tier = "Silver"

        else:
            name = "Active Intimate Celebration Organizers"
            desc = "Regular individuals planning anniversaries, engagements, and local family functions. Steady engagement and healthy booking conversion."
            persona = "Core Steady Celebrants"
            recommended_strategy = [
                "Showcase curated local verified vendors in Mumbai, Pune, and Nashik",
                "Recommend customizable catering and photography add-ons",
                "Offer flexible EMI or staged milestone payment plans",
                "Encourage reviews and referral bonuses for upcoming family celebrations"
            ]
            tier = "Bronze"

        profiles[str(cluster_id)] = {
            "cluster_id": int(cluster_id),
            "segment_name": name,
            "tier": tier,
            "persona": persona,
            "customer_count": count,
            "customer_percentage": pct,
            "description": desc,
            "metrics": {
                "avg_spend_inr": round(mean_spend, 2),
                "avg_booking_value_inr": round(mean_aov, 2),
                "avg_orders": round(mean_orders, 2),
                "avg_frequency_monthly": round(mean_freq, 3),
                "avg_recency_days": round(mean_recency, 1),
                "avg_discount_usage": round(mean_discount, 3),
                "avg_engagement_score": round(mean_engagement, 2),
                "avg_abandonment_rate": round(mean_abandonment, 3)
            },
            "strategies": recommended_strategy
        }

    return profiles
