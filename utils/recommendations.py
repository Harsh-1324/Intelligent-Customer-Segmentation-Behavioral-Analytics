"""
Business recommendation engine tailored for curated event & celebration marketplace.
Decoupled recommendation logic translating cluster profiles and individual customer signals
into tactical marketing playbooks.
"""

from typing import Dict, Any, List

def get_actionable_recommendations(
    segment_name: str,
    total_spend: float,
    discount_dependency: float,
    recency_days: float,
    event_type: str = "Destination Wedding"
) -> Dict[str, Any]:
    """
    Returns prioritized operational and marketing recommendations based on segment and behavioral signals.
    """
    channel = "Email"
    priority = "Medium"
    primary_campaign = "Standard Engagement"
    actions = []

    if "High-Value" in segment_name:
        priority = "Critical / VIP"
        channel = "Dedicated Relationship Manager / VIP Phone Concierge"
        primary_campaign = "Exclusive Luxury Destination Showcase"
        actions = [
            f"Schedule an in-person or 1-on-1 virtual venue tour for luxury venues (Lonavala villas, Alibaug sea-facing lawns, 5-star banquets).",
            f"Introduce complimentary tasting session with premier catering partners.",
            f"Assign a senior event concierge specialist to handle end-to-end vendor negotiation.",
            f"Avoid generic discount coupons; offer value enhancements such as complimentary bridal suite or drone photography."
        ]

    elif "Corporate" in segment_name:
        priority = "High"
        channel = "B2B Corporate Account Executive / Direct Email"
        primary_campaign = "Annual Corporate Event Retainer & Volume Rebates"
        actions = [
            f"Offer consolidated billing with GST-compliant multi-event invoices.",
            f"Present corporate offsite packages with team-building resort options in Pune and Lonavala.",
            f"Assign dedicated account manager with guaranteed 1-hour vendor turnaround time.",
            f"Provide bulk booking rebates (e.g. 5% credit on booking 3+ annual events)."
        ]

    elif "At-Risk" in segment_name:
        priority = "High (Win-Back)"
        channel = "WhatsApp Business Notification & Push Notification"
        primary_campaign = "Celebration Reactivation & ₹5,000 Booking Credit"
        actions = [
            f"Send automated WhatsApp nudge: 'Planning an upcoming celebration? Your exclusive ₹5,000 venue credit is waiting.'",
            f"Deliver curated list of top-trending celebration venues updated since their last visit {int(recency_days)} days ago.",
            f"Deploy an automated 2-question survey to understand drop-off reason (budget, dates, or vendor availability).",
            f"Offer flexible cancellation terms to lower re-entry barrier."
        ]

    elif "Budget" in segment_name:
        priority = "Medium"
        channel = "App Push Notification & SMS Alerts"
        primary_campaign = "Value Party Packages & Flash Venue Discounts"
        actions = [
            f"Promote curated 'All-Inclusive Celebration Packages' under ₹1,00,000 for birthdays and intimate functions.",
            f"Trigger automated 10% discount promo code upon cart/shortlist abandonment.",
            f"Highlight verified banquet halls with transparent per-plate pricing in Thane, Navi Mumbai, and Pune.",
            f"Feature interest-free 3-month milestone payment options."
        ]

    else:  # Active Intimate Celebration
        priority = "Medium"
        channel = "Email & App In-App Feed"
        primary_campaign = "Curated Local Celebrations & Verified Vendors"
        actions = [
            f"Recommend top-rated local decorators and candid photographers in their home city.",
            f"Send tailored checklist for their specific celebration: '{event_type} Planning Guide'.",
            f"Invite to join the Celebration Loyalty Club to earn credits for future family celebrations.",
            f"Highlight customer reviews and photo galleries from recently hosted events."
        ]

    return {
        "priority_level": priority,
        "recommended_channel": channel,
        "campaign_name": primary_campaign,
        "action_items": actions
    }
