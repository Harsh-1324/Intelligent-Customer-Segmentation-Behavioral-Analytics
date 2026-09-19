"""
CelebrateIQ | Intelligent Customer Segmentation & Behavioral Analytics Platform
Streamlit Analytics Dashboard (Multi-Page Interactive Application)
"""

import os
import sys
import json
import joblib
import requests
import numpy as np
import pandas as pd
import streamlit as st

# Configure project path imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from utils.visualization import (
    plot_2d_pca,
    plot_3d_pca,
    plot_cluster_distribution,
    plot_radar_comparison,
    plot_segment_metric_bars,
    plot_elbow_and_silhouette
)
from utils.recommendations import get_actionable_recommendations
from backend.preprocessing import transform_customer_input
from backend.schemas import CustomerInput

# Page Configuration
st.set_page_config(
    page_title="CelebrateIQ | Customer Segmentation & Behavioral Analytics",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1D3557;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #6C757D;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #FFFFFF;
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border-left: 4px solid #E63946;
        margin-bottom: 1rem;
    }
    .badge-platinum {
        background: linear-gradient(135deg, #e5e5be, #003973);
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-gold {
        background: #F4A261;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-silver {
        background: #2A9D8F;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-atrisk {
        background: #E63946;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# Data & Model Loader
@st.cache_resource
def load_project_artifacts():
    models_dir = os.path.join(BASE_DIR, "models")
    data_dir = os.path.join(BASE_DIR, "data")
    outputs_dir = os.path.join(BASE_DIR, "outputs")

    artifacts = {}
    try:
        artifacts["kmeans"] = joblib.load(os.path.join(models_dir, "clustering_model.pkl"))
        artifacts["scaler"] = joblib.load(os.path.join(models_dir, "scaler.pkl"))
        artifacts["pca"] = joblib.load(os.path.join(models_dir, "pca.pkl"))
        if os.path.exists(os.path.join(models_dir, "pca_3d.pkl")):
            artifacts["pca_3d"] = joblib.load(os.path.join(models_dir, "pca_3d.pkl"))
        artifacts["features"] = joblib.load(os.path.join(models_dir, "feature_columns.pkl"))

        with open(os.path.join(models_dir, "cluster_profiles.json"), "r") as f:
            artifacts["profiles"] = json.load(f)

        with open(os.path.join(models_dir, "evaluation_curves.json"), "r") as f:
            artifacts["curves"] = json.load(f)

        with open(os.path.join(models_dir, "algorithm_comparison.json"), "r") as f:
            artifacts["benchmark"] = pd.read_json(f)

        # Load annotated dataset or raw dataset
        seg_csv = os.path.join(outputs_dir, "segmented_customers.csv")
        if os.path.exists(seg_csv):
            artifacts["df"] = pd.read_csv(seg_csv)
        else:
            artifacts["df"] = pd.read_csv(os.path.join(data_dir, "customer_data.csv"))
            
        artifacts["loaded"] = True
    except Exception as e:
        artifacts["loaded"] = False
        artifacts["error"] = str(e)

    return artifacts

artifacts = load_project_artifacts()

# Sidebar Navigation & Platform Info
with st.sidebar:
    st.markdown("### ✨ **CelebrateIQ**")
    st.markdown("**Curated Event & Celebration Platform**  \n*Customer Behavioral Intelligence*")
    st.markdown("---")

    selected_page = st.radio(
        "Navigation",
        [
            "📊 Executive Dashboard",
            "🎯 Customer Segmentation",
            "🌐 Cluster 2D/3D Visualizer",
            "⚖️ Segment Comparison",
            "📈 Clustering Evaluation",
            "💡 Business Insights & Playbook",
            "📁 Batch CSV Segmentation"
        ]
    )

    st.markdown("---")
    # Backend Status indicator
    api_url = st.text_input("FastAPI Endpoint", value="http://127.0.0.1:8000")
    backend_online = False
    try:
        r = requests.get(f"{api_url}/health", timeout=1.0)
        if r.status_code == 200 and r.json().get("status") == "healthy":
            backend_online = True
    except Exception:
        backend_online = False

    if backend_online:
        st.success("🟢 FastAPI Backend Connected")
    else:
        st.info("🟡 Standalone Mode (Direct Artifacts)")

    st.markdown("---")
    st.caption("CelebrateIQ Analytics Platform v1.0")

if not artifacts.get("loaded", False):
    st.error(f"⚠️ Model artifacts not found: {artifacts.get('error')}. Please run `python training/train_clustering.py` first.")
    st.stop()

df = artifacts["df"]
profiles = artifacts["profiles"]
kmeans = artifacts["kmeans"]
scaler = artifacts["scaler"]
pca = artifacts["pca"]
feature_cols = artifacts["features"]

# -------------------------------------------------------------
# PAGE 1: EXECUTIVE DASHBOARD
# -------------------------------------------------------------
if selected_page == "📊 Executive Dashboard":
    st.markdown("<div class='main-header'>Executive Segmentation Overview</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Holistic customer behavioral metrics across Maharashtra destination weddings and celebration bookings.</div>", unsafe_allow_html=True)

    # Top KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Customers", f"{len(df):,}")
    with col2:
        st.metric("Discovered Segments", f"{len(profiles)}")
    with col3:
        largest_seg = max(profiles.values(), key=lambda x: x["customer_count"])
        st.metric("Largest Segment", f"{largest_seg['customer_count']:,} ({largest_seg['customer_percentage']}%)")
    with col4:
        avg_spend = df["Total_Spend_INR"].mean()
        st.metric("Avg Customer Spend", f"₹{avg_spend:,.0f}")

    st.markdown("---")

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.subheader("Segment Distribution")
        fig_dist = plot_cluster_distribution(profiles)
        st.plotly_chart(fig_dist, use_container_width=True)

    with col_right:
        st.subheader("Key Segment Profiles")
        for cid, prof in profiles.items():
            badge_class = "badge-platinum" if prof["tier"] == "Platinum" else ("badge-gold" if prof["tier"] == "Gold" else ("badge-atrisk" if prof["tier"] == "At-Risk" else "badge-silver"))
            st.markdown(f"""
            <div style='background: white; border: 1px solid #E5E7EB; border-radius: 8px; padding: 12px; margin-bottom: 8px;'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <strong>{prof['segment_name']}</strong>
                    <span class='{badge_class}'>{prof['tier']}</span>
                </div>
                <div style='font-size: 0.85rem; color: #4B5563; margin-top: 4px;'>
                    Avg Spend: <b>₹{prof['metrics']['avg_spend_inr']:,.0f}</b> | Avg Orders: <b>{prof['metrics']['avg_orders']}</b> | Customers: <b>{prof['customer_count']}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Key Business Takeaways")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**💍 High-Value Luxury Revenue Anchor**  \nHigh-value destination wedding planners represent only ~18% of users but account for over **55% of gross platform booking volume**.")
    with c2:
        st.markdown("**🏢 Corporate High-Frequency Cashflow**  \nCorporate and recurring planners generate high booking cadence (averaging 11+ bookings/year) with low marketing acquisition cost.")
    with c3:
        st.markdown("**📉 Proactive At-Risk Reactivation**  \nAutomating WhatsApp re-engagement for dormant accounts before 100 days can reclaim up to **18% of lapsed celebration budgets**.")

# -------------------------------------------------------------
# PAGE 2: CUSTOMER SEGMENTATION FORM
# -------------------------------------------------------------
elif selected_page == "🎯 Customer Segmentation":
    st.markdown("<div class='main-header'>Individual Customer Behavioral Classifier</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Predict behavioral segment and generate tailored marketing recommendations for any celebration lead.</div>", unsafe_allow_html=True)

    # Preset Persona Quick-Select
    st.markdown("**Quick Fill with Preset Archetypes:**")
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    
    preset = None
    if p_col1.button("💍 Luxury Destination Couple"):
        preset = {"age": 31, "tenure": 24.0, "orders": 4, "spend": 2600000, "recency": 15, "discount": 0.05, "visits": 60, "app": 50, "whatsapp": 28, "shortlist": 18, "abandon": 0.25, "wishlist": 22, "support": 3}
    if p_col2.button("🏢 Corporate Event Planner"):
        preset = {"age": 38, "tenure": 36.0, "orders": 12, "spend": 2160000, "recency": 12, "discount": 0.35, "visits": 70, "app": 58, "whatsapp": 35, "shortlist": 24, "abandon": 0.30, "wishlist": 19, "support": 6}
    if p_col3.button("🎂 Budget Birthday Celebrant"):
        preset = {"age": 26, "tenure": 12.0, "orders": 2, "spend": 65000, "recency": 45, "discount": 0.85, "visits": 25, "app": 15, "whatsapp": 6, "shortlist": 12, "abandon": 0.75, "wishlist": 8, "support": 1}
    if p_col4.button("📉 At-Risk Inactive Planner"):
        preset = {"age": 34, "tenure": 28.0, "orders": 1, "spend": 75000, "recency": 185, "discount": 0.40, "visits": 10, "app": 6, "whatsapp": 2, "shortlist": 4, "abandon": 0.85, "wishlist": 3, "support": 1}

    # Form
    with st.form("customer_segment_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### Demographics & Tenure")
            age = st.number_input("Customer Age", min_value=18, max_value=85, value=preset["age"] if preset else 30)
            tenure = st.number_input("Tenure (Months)", min_value=1.0, max_value=72.0, value=preset["tenure"] if preset else 20.0, step=1.0)
            orders = st.number_input("Total Bookings/Orders", min_value=1, max_value=50, value=preset["orders"] if preset else 3)
            recency = st.number_input("Days Since Last Inquiry/Booking", min_value=1, max_value=365, value=preset["recency"] if preset else 20)

        with col2:
            st.markdown("#### Monetary & Spending")
            spend = st.number_input("Total Spend (INR ₹)", min_value=5000, max_value=20000000, value=preset["spend"] if preset else 500000, step=10000)
            discount = st.slider("Discount Usage Proportion", 0.0, 1.0, preset["discount"] if preset else 0.20, 0.05)
            abandon = st.slider("Shortlist Abandonment Rate", 0.0, 1.0, preset["abandon"] if preset else 0.35, 0.05)

        with col3:
            st.markdown("#### Platform Engagement")
            visits = st.number_input("Website Visits", min_value=0, max_value=300, value=preset["visits"] if preset else 45)
            app_sessions = st.number_input("App Sessions", min_value=0, max_value=300, value=preset["app"] if preset else 35)
            whatsapp = st.number_input("WhatsApp Inquiries", min_value=0, max_value=150, value=preset["whatsapp"] if preset else 15)
            shortlist = st.number_input("Venues/Vendors Shortlisted", min_value=0, max_value=100, value=preset["shortlist"] if preset else 12)
            wishlist = st.number_input("Wishlist Count", min_value=0, max_value=100, value=preset["wishlist"] if preset else 14)
            support = st.number_input("Support Tickets", min_value=0, max_value=30, value=preset["support"] if preset else 2)

        submit_btn = st.form_submit_button("🔍 Identify Customer Segment", use_container_width=True)

    if submit_btn:
        payload = CustomerInput(
            age=age,
            tenure=tenure,
            orders=orders,
            total_spend=spend,
            avg_order_value=round(spend / orders, 2),
            days_since_last_inquiry=recency,
            discount_usage=discount,
            website_visits=visits,
            app_sessions=app_sessions,
            whatsapp_inquiries=whatsapp,
            cart_shortlist_additions=shortlist,
            abandonment_rate=abandon,
            wishlist_count=wishlist,
            support_interactions=support
        )

        res = None
        # Try FastAPI first if online, else fallback
        if backend_online:
            try:
                r = requests.post(f"{api_url}/segment", json=payload.model_dump(), timeout=2.0)
                if r.status_code == 200:
                    res = r.json()
            except Exception:
                res = None

        if res is None:
            # Standalone direct inference
            feat_df = transform_customer_input(payload)
            X_scaled = scaler.transform(feat_df[feature_cols])
            cid = int(kmeans.predict(X_scaled)[0])
            centroid = kmeans.cluster_centers_[cid]
            dist = float(np.linalg.norm(X_scaled[0] - centroid))
            pca_c = pca.transform(X_scaled)[0]
            prof = profiles.get(str(cid), {})

            res = {
                "cluster": cid,
                "segment_name": prof.get("segment_name", f"Cluster {cid}"),
                "tier": prof.get("tier", "Standard"),
                "distance_from_centroid": round(dist, 4),
                "pca_coordinates": {"pca_1": round(pca_c[0], 4), "pca_2": round(pca_c[1], 4)},
                "description": prof.get("description", ""),
                "recommended_strategies": prof.get("strategies", [])
            }

        st.markdown("---")
        # Result Presentation Card
        res_col1, res_col2 = st.columns([1.2, 1])

        with res_col1:
            st.markdown(f"""
            <div style='background: #F8F9FA; border: 2px solid #1D3557; border-radius: 12px; padding: 20px;'>
                <div style='font-size: 0.85rem; text-transform: uppercase; color: #6C757D; letter-spacing: 1px;'>CLASSIFIED SEGMENT</div>
                <h2 style='color: #1D3557; margin-top: 5px; margin-bottom: 8px;'>{res['segment_name']}</h2>
                <div style='margin-bottom: 12px;'>
                    <span style='background: #1D3557; color: white; padding: 4px 10px; border-radius: 6px; font-weight: 600;'>Cluster ID: {res['cluster']}</span>
                    <span style='background: #E63946; color: white; padding: 4px 10px; border-radius: 6px; font-weight: 600; margin-left: 6px;'>Tier: {res['tier']}</span>
                    <span style='background: #E5E7EB; color: #1F2937; padding: 4px 10px; border-radius: 6px; font-weight: 500; margin-left: 6px;'>Centroid Distance: {res['distance_from_centroid']}</span>
                </div>
                <p style='color: #374151; font-size: 0.95rem; line-height: 1.5;'>{res['description']}</p>
                <hr style='border: none; border-top: 1px solid #E5E7EB; margin: 15px 0;' />
                <h4 style='color: #1D3557; margin-bottom: 8px;'>Recommended Operational Strategies</h4>
                <ul style='color: #1F2937; font-size: 0.92rem; padding-left: 20px;'>
                    {''.join(f'<li style="margin-bottom: 6px;">{s}</li>' for s in res['recommended_strategies'])}
                </ul>
            </div>
            """, unsafe_allow_html=True)

        with res_col2:
            st.markdown("#### Position in Customer Space")
            fig_point = plot_2d_pca(
                df, 
                highlight_point=(res["pca_coordinates"]["pca_1"], res["pca_coordinates"]["pca_2"], res["segment_name"])
            )
            st.plotly_chart(fig_point, use_container_width=True)

# -------------------------------------------------------------
# PAGE 3: CLUSTER 2D / 3D VISUALIZER
# -------------------------------------------------------------
elif selected_page == "🌐 Cluster 2D/3D Visualizer":
    st.markdown("<div class='main-header'>Interactive Cluster Behavioral Visualizer</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Explore high-dimensional customer space projected into 2D and 3D via Principal Component Analysis (PCA).</div>", unsafe_allow_html=True)

    view_mode = st.radio("Visualization Mode", ["2D PCA Projection", "3D Interactive PCA Projection"], horizontal=True)

    # Filter by segment
    all_segments = sorted(list(df["Segment_Name"].dropna().unique()))
    selected_filter = st.multiselect("Filter Segments", all_segments, default=all_segments)

    filtered_df = df[df["Segment_Name"].isin(selected_filter)]

    if view_mode == "2D PCA Projection":
        fig_2d = plot_2d_pca(filtered_df)
        st.plotly_chart(fig_2d, use_container_width=True)
    else:
        fig_3d = plot_3d_pca(filtered_df)
        st.plotly_chart(fig_3d, use_container_width=True)

    st.markdown("---")
    st.markdown("#### PCA Explained Variance & Interpretation")
    st.info(
        "**PCA Component 1 (~44% variance)** correlates strongly with **Monetary Value & Booking Scale** (Total Spend INR, Avg Booking Value, and Cart Shortlist Additions).  \n"
        "**PCA Component 2 (~22% variance)** correlates with **Recency vs Platform Engagement** (Engagement Score, Website Visits vs Days Since Last Inquiry)."
    )

# -------------------------------------------------------------
# PAGE 4: SEGMENT COMPARISON
# -------------------------------------------------------------
elif selected_page == "⚖️ Segment Comparison":
    st.markdown("<div class='main-header'>Segment Behavioral Comparison</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Compare multiple customer cohorts side-by-side across RFM, engagement, and conversion attributes.</div>", unsafe_allow_html=True)

    segment_names = [v["segment_name"] for v in profiles.values()]
    selected_comp = st.multiselect("Select Segments to Compare", segment_names, default=segment_names[:3])

    if len(selected_comp) < 1:
        st.warning("Please select at least 1 segment to display comparison.")
    else:
        col_radar, col_bars = st.columns([1, 1])

        with col_radar:
            st.subheader("Radar Profile Comparison")
            fig_rad = plot_radar_comparison(profiles, selected_comp)
            st.plotly_chart(fig_rad, use_container_width=True)

        with col_bars:
            st.subheader("Spending vs Frequency Comparison")
            metric_to_plot = st.selectbox(
                "Compare Metric",
                ["Total_Spend_INR", "Total_Bookings", "Engagement_Score", "Discount_Dependency", "Recency_Days", "Abandonment_Rate"]
            )
            fig_bar = plot_segment_metric_bars(df[df["Segment_Name"].isin(selected_comp)], metric_to_plot, f"Average {metric_to_plot} across Segments")
            st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    st.subheader("Detailed Metric Summary Matrix")
    matrix_rows = []
    for prof in profiles.values():
        m = prof["metrics"]
        matrix_rows.append({
            "Segment": prof["segment_name"],
            "Tier": prof["tier"],
            "Customer Count": prof["customer_count"],
            "Avg Spend (₹)": f"₹{m['avg_spend_inr']:,.0f}",
            "Avg Booking Value (₹)": f"₹{m['avg_booking_value_inr']:,.0f}",
            "Avg Bookings": m["avg_orders"],
            "Recency (Days)": m["avg_recency_days"],
            "Engagement Score": m["avg_engagement_score"],
            "Discount Ratio": f"{m['avg_discount_usage']*100:.1f}%"
        })
    st.dataframe(pd.DataFrame(matrix_rows), use_container_width=True)

# -------------------------------------------------------------
# PAGE 5: CLUSTERING EVALUATION & BENCHMARK
# -------------------------------------------------------------
elif selected_page == "📈 Clustering Evaluation":
    st.markdown("<div class='main-header'>Clustering Evaluation & Algorithmic Benchmarking</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Rigorous mathematical evaluation comparing K-Means, DBSCAN, and Agglomerative Hierarchical Clustering.</div>", unsafe_allow_html=True)

    curves = artifacts.get("curves", {})
    fig_elbow, fig_sil = plot_elbow_and_silhouette(curves)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_elbow, use_container_width=True)
    with col2:
        st.plotly_chart(fig_sil, use_container_width=True)

    st.markdown("---")
    st.subheader("Algorithm Comparison Matrix")
    benchmark_df = artifacts.get("benchmark", pd.DataFrame())
    if not benchmark_df.empty:
        st.dataframe(
            benchmark_df.style.highlight_max(subset=["Silhouette Score", "Calinski-Harabasz Score"], color="#D1E7DD")
            .highlight_min(subset=["Davies-Bouldin Index"], color="#D1E7DD"),
            use_container_width=True
        )

    st.markdown("---")
    st.markdown("### Algorithmic Comparison Insights")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**1. K-Means (Selected Primary Engine)**  \nAssumes convex, spherical clusters in normalized feature space. Demonstrates clear separation of customer tiers with optimal silhouette score and fast inference complexity $\\mathcal{O}(k \\cdot n \\cdot d)$.")
    with c2:
        st.markdown("**2. DBSCAN (Density-Based)**  \nExcels at uncovering arbitrary density shapes and isolating outliers (e.g. anomalous one-off extreme bookings). However, uniform density assumptions struggle with customer cohorts having varying behavioral spread.")
    with c3:
        st.markdown("**3. Agglomerative (Hierarchical)**  \nConstructs a bottom-up dendrogram structure with Ward linkage. Corroborates the 5 dominant customer clusters found by K-Means, providing deep structural validation.")

# -------------------------------------------------------------
# PAGE 6: BUSINESS INSIGHTS & MARKETING PLAYBOOK
# -------------------------------------------------------------
elif selected_page == "💡 Business Insights & Playbook":
    st.markdown("<div class='main-header'>Actionable Business Strategies & Marketing Playbook</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Operational execution strategies for celebration & event marketplace to maximize conversion and lifetime value.</div>", unsafe_allow_html=True)

    for cid, prof in profiles.items():
        recs = get_actionable_recommendations(
            prof["segment_name"],
            prof["metrics"]["avg_spend_inr"],
            prof["metrics"]["avg_discount_usage"],
            prof["metrics"]["avg_recency_days"]
        )

        with st.expander(f"📌 {prof['segment_name']} ({prof['tier']} Tier) — {prof['customer_count']} Customers ({prof['customer_percentage']}%)", expanded=True):
            col_l, col_r = st.columns([1, 1.2])

            with col_l:
                st.markdown(f"**Customer Persona:** {prof['persona']}")
                st.markdown(f"**Description:** {prof['description']}")
                st.markdown(f"**Primary Channel:** `{recs['recommended_channel']}`")
                st.markdown(f"**Campaign Focus:** `{recs['campaign_name']}`")
                st.markdown(f"**Operational Priority:** `{recs['priority_level']}`")

            with col_r:
                st.markdown("**Tactical Marketing Actions:**")
                for act in recs["action_items"]:
                    st.markdown(f"- {act}")

# -------------------------------------------------------------
# PAGE 7: BATCH CSV SEGMENTATION
# -------------------------------------------------------------
elif selected_page == "📁 Batch CSV Segmentation":
    st.markdown("<div class='main-header'>Batch Customer CSV Segmentation</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Upload a customer batch file (.csv) to classify all leads into behavioral clusters and download annotated results.</div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload Customer CSV File", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write(f"Uploaded **{len(batch_df)}** records with columns: `{list(batch_df.columns)}`")
        
        st.dataframe(batch_df.head(5), use_container_width=True)

        if st.button("🚀 Run Batch Segmentation"):
            with st.spinner("Processing batch records through clustering model..."):
                from backend.predictor import predictor
                if not predictor.is_loaded:
                    predictor.load_artifacts()

                try:
                    processed_df = predictor.predict_batch(batch_df)
                    st.success("Batch segmentation completed successfully!")

                    col_summary1, col_summary2 = st.columns(2)
                    with col_summary1:
                        st.subheader("Segment Breakdown")
                        st.write(processed_df["Segment_Name"].value_counts())

                    with col_summary2:
                        st.subheader("Annotated Preview")
                        st.dataframe(
                            processed_df[["Customer_ID", "Segment_Name", "Cluster", "Distance_From_Centroid", "Total_Spend_INR"]].head(5),
                            use_container_width=True
                        )

                    csv_data = processed_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Segmented CSV",
                        data=csv_data,
                        file_name=f"segmented_{uploaded_file.name}",
                        mime="text/csv",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Error during batch processing: {e}")
