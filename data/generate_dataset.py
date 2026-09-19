"""
Synthetic Customer Dataset Generator for Curated Event & Celebration Platform.
Generates realistic customer transactional and behavioral data across celebration destinations.
"""

import os
import numpy as np
import pandas as pd

def generate_customer_dataset(num_samples: int = 6000, random_seed: int = 42) -> pd.DataFrame:
    np.random.seed(random_seed)

    cities = ["Mumbai", "Pune", "Lonavala", "Alibaug", "Nashik", "Thane", "Navi Mumbai", "Goa"]
    city_weights = [0.28, 0.22, 0.12, 0.10, 0.10, 0.08, 0.06, 0.04]

    event_types = [
        "Destination Wedding", "Local Wedding", "Corporate Event", 
        "Birthday Party", "Engagement", "Anniversary Celebration", "Festival / Party"
    ]

    # Archetype proportions: 
    # 0: High-Value Destination Celebrants (~18%)
    # 1: Corporate & Recurring Planners (~15%)
    # 2: Budget-Conscious Celebrants (~28%)
    # 3: Active Intimate Celebration Organizers (~24%)
    # 4: At-Risk / Lapsed Inactive Planners (~15%)
    archetypes = np.random.choice([0, 1, 2, 3, 4], size=num_samples, p=[0.18, 0.15, 0.28, 0.24, 0.15])

    data = []

    for i in range(num_samples):
        cust_id = f"CUST-{10000 + i}"
        arch = archetypes[i]
        city = np.random.choice(cities, p=city_weights)
        
        if arch == 0:  # High-Value Destination Celebrants
            age = int(np.clip(np.random.normal(31, 4), 23, 58))
            primary_event = np.random.choice(
                ["Destination Wedding", "Local Wedding", "Engagement"], 
                p=[0.65, 0.25, 0.10]
            )
            tenure_months = int(np.clip(np.random.normal(24, 8), 4, 60))
            orders = int(np.clip(np.random.poisson(3.5), 1, 10))
            avg_order_value = np.clip(np.random.normal(650000, 150000), 250000, 2000000)
            total_spend = round(orders * avg_order_value, 2)
            days_since_last_inquiry = int(np.clip(np.random.exponential(25), 2, 120))
            discount_usage = np.clip(np.random.beta(1.5, 6.0), 0.0, 0.35)
            returns = int(np.random.choice([0, 1], p=[0.92, 0.08]))
            website_visits = int(np.clip(np.random.normal(55, 15), 15, 120))
            app_sessions = int(np.clip(np.random.normal(48, 14), 10, 110))
            whatsapp_inquiries = int(np.clip(np.random.normal(24, 8), 5, 60))
            cart_shortlist_additions = int(np.clip(np.random.normal(16, 5), 4, 40))
            abandonment_rate = np.clip(np.random.beta(2, 5), 0.1, 0.55)
            wishlist_count = int(np.clip(np.random.normal(20, 6), 5, 50))
            support_interactions = int(np.clip(np.random.poisson(4.0), 1, 12))

        elif arch == 1:  # Corporate & Recurring Event Planners
            age = int(np.clip(np.random.normal(38, 6), 26, 62))
            primary_event = np.random.choice(
                ["Corporate Event", "Festival / Party", "Anniversary Celebration"], 
                p=[0.75, 0.15, 0.10]
            )
            tenure_months = int(np.clip(np.normal(36, 10), 10, 60))
            orders = int(np.clip(np.random.poisson(11), 5, 25))
            avg_order_value = np.clip(np.random.normal(180000, 45000), 75000, 450000)
            total_spend = round(orders * avg_order_value, 2)
            days_since_last_inquiry = int(np.clip(np.random.exponential(18), 1, 75))
            discount_usage = np.clip(np.random.beta(3.0, 4.0), 0.15, 0.65)
            returns = int(np.random.choice([0, 1, 2], p=[0.85, 0.12, 0.03]))
            website_visits = int(np.clip(np.random.normal(68, 16), 25, 140))
            app_sessions = int(np.clip(np.random.normal(55, 15), 18, 120))
            whatsapp_inquiries = int(np.clip(np.random.normal(32, 10), 8, 80))
            cart_shortlist_additions = int(np.clip(np.random.normal(22, 6), 6, 55))
            abandonment_rate = np.clip(np.random.beta(2.5, 4.5), 0.15, 0.50)
            wishlist_count = int(np.clip(np.random.normal(18, 5), 4, 45))
            support_interactions = int(np.clip(np.random.poisson(5.5), 1, 15))

        elif arch == 2:  # Budget-Conscious Occasional Celebrants
            age = int(np.clip(np.random.normal(27, 5), 20, 50))
            primary_event = np.random.choice(
                ["Birthday Party", "Festival / Party", "Engagement"], 
                p=[0.55, 0.30, 0.15]
            )
            tenure_months = int(np.clip(np.random.normal(14, 7), 2, 40))
            orders = int(np.clip(np.random.poisson(1.8), 1, 5))
            avg_order_value = np.clip(np.random.normal(35000, 12000), 10000, 85000)
            total_spend = round(orders * avg_order_value, 2)
            days_since_last_inquiry = int(np.clip(np.random.exponential(50), 5, 200))
            discount_usage = np.clip(np.random.beta(6.0, 2.0), 0.45, 0.95)
            returns = int(np.random.choice([0, 1], p=[0.88, 0.12]))
            website_visits = int(np.clip(np.random.normal(28, 9), 5, 65))
            app_sessions = int(np.clip(np.random.normal(18, 7), 3, 50))
            whatsapp_inquiries = int(np.clip(np.random.normal(8, 4), 1, 25))
            cart_shortlist_additions = int(np.clip(np.random.normal(10, 4), 2, 28))
            abandonment_rate = np.clip(np.random.beta(5.0, 2.0), 0.45, 0.90)
            wishlist_count = int(np.clip(np.random.normal(9, 4), 1, 30))
            support_interactions = int(np.clip(np.random.poisson(1.8), 0, 8))

        elif arch == 3:  # Active Intimate Celebration Organizers
            age = int(np.clip(np.random.normal(32, 5), 22, 55))
            primary_event = np.random.choice(
                ["Anniversary Celebration", "Local Wedding", "Engagement", "Birthday Party"], 
                p=[0.35, 0.30, 0.20, 0.15]
            )
            tenure_months = int(np.clip(np.random.normal(22, 8), 4, 48))
            orders = int(np.clip(np.random.poisson(3.8), 2, 8))
            avg_order_value = np.clip(np.random.normal(135000, 35000), 50000, 300000)
            total_spend = round(orders * avg_order_value, 2)
            days_since_last_inquiry = int(np.clip(np.random.exponential(30), 3, 110))
            discount_usage = np.clip(np.random.beta(2.5, 4.0), 0.15, 0.55)
            returns = int(np.random.choice([0, 1], p=[0.94, 0.06]))
            website_visits = int(np.clip(np.random.normal(42, 11), 12, 85))
            app_sessions = int(np.clip(np.random.normal(36, 10), 10, 75))
            whatsapp_inquiries = int(np.clip(np.random.normal(18, 6), 4, 45))
            cart_shortlist_additions = int(np.clip(np.random.normal(14, 4), 3, 35))
            abandonment_rate = np.clip(np.random.beta(3.0, 4.0), 0.20, 0.65)
            wishlist_count = int(np.clip(np.random.normal(15, 5), 3, 38))
            support_interactions = int(np.clip(np.random.poisson(2.6), 0, 9))

        else:  # At-Risk / Lapsed Inactive Planners
            age = int(np.clip(np.random.normal(34, 6), 22, 60))
            primary_event = np.random.choice(event_types)
            tenure_months = int(np.clip(np.random.normal(26, 8), 8, 55))
            orders = int(np.clip(np.random.poisson(1.4), 1, 4))
            avg_order_value = np.clip(np.random.normal(70000, 30000), 15000, 200000)
            total_spend = round(orders * avg_order_value, 2)
            days_since_last_inquiry = int(np.clip(np.random.normal(190, 45), 100, 365))
            discount_usage = np.clip(np.random.beta(3.0, 3.0), 0.1, 0.7)
            returns = int(np.random.choice([0, 1], p=[0.82, 0.18]))
            website_visits = int(np.clip(np.random.normal(12, 5), 1, 30))
            app_sessions = int(np.clip(np.random.normal(8, 4), 0, 25))
            whatsapp_inquiries = int(np.clip(np.random.normal(3, 2), 0, 12))
            cart_shortlist_additions = int(np.clip(np.random.normal(4, 2), 0, 15))
            abandonment_rate = np.clip(np.random.beta(6.0, 1.5), 0.60, 0.98)
            wishlist_count = int(np.clip(np.random.normal(4, 2), 0, 16))
            support_interactions = int(np.clip(np.random.poisson(1.2), 0, 6))

        # Add 3% random noise across features for realistic dataset anomalies
        if np.random.rand() < 0.03:
            total_spend = round(total_spend * np.random.uniform(0.4, 2.5), 2)
            days_since_last_inquiry = int(np.clip(days_since_last_inquiry * np.random.uniform(0.3, 2.0), 1, 400))

        data.append({
            "Customer_ID": cust_id,
            "Age": age,
            "Location": city,
            "Primary_Event_Type": primary_event,
            "Tenure_Months": tenure_months,
            "Total_Bookings": orders,
            "Total_Spend_INR": total_spend,
            "Avg_Booking_Value_INR": round(total_spend / max(orders, 1), 2),
            "Days_Since_Last_Inquiry": days_since_last_inquiry,
            "Discount_Usage_Ratio": round(discount_usage, 3),
            "Cancellations": returns,
            "Website_Visits": website_visits,
            "App_Sessions": app_sessions,
            "WhatsApp_Inquiries": whatsapp_inquiries,
            "Cart_Shortlist_Additions": cart_shortlist_additions,
            "Abandonment_Rate": round(abandonment_rate, 3),
            "Wishlist_Count": wishlist_count,
            "Support_Interactions": support_interactions
        })

    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, "customer_data.csv")
    
    print(f"Generating realistic customer dataset for celebration platform...")
    df = generate_customer_dataset(num_samples=6000)
    df.to_csv(output_path, index=False)
    print(f"Dataset successfully created at: {output_path}")
    print(f"Total records: {len(df)} | Columns: {len(df.columns)}")
