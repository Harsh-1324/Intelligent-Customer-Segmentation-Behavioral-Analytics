"""
Interactive visualization helpers using Plotly for Streamlit analytics dashboard.
"""

from typing import List, Dict, Any
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Professional celebration palette (Rose gold, Wine, Navy, Gold, Teal)
CLUSTER_PALETTE = [
    "#E63946",  # Vibrant Red / Rose
    "#2A9D8F",  # Teal
    "#E76F51",  # Warm Coral
    "#457B9D",  # Steel Blue
    "#9A8C98",  # Slate Mauve
    "#F4A261",  # Sandstone Gold
    "#1D3557"   # Deep Navy
]

def plot_2d_pca(df: pd.DataFrame, highlight_point: tuple = None) -> go.Figure:
    """
    Renders 2D PCA cluster visualization with custom hover metadata and centroids.
    """
    fig = px.scatter(
        df,
        x="PCA_1",
        y="PCA_2",
        color="Segment_Name",
        hover_name="Customer_ID",
        hover_data={
            "Location": True,
            "Total_Spend_INR": ":,.0f",
            "Total_Bookings": True,
            "Engagement_Score": True,
            "Recency_Days": True,
            "PCA_1": False,
            "PCA_2": False
        },
        color_discrete_sequence=CLUSTER_PALETTE,
        title="2D Customer Behavioral Space (PCA Projection)",
        labels={"PCA_1": "Principal Component 1 (Spending & Scale)", "PCA_2": "Principal Component 2 (Engagement vs Recency)"},
        template="plotly_white",
        opacity=0.75
    )

    # If single customer prediction point is provided, overlay it as a gold star
    if highlight_point:
        px_coord, py_coord, p_name = highlight_point
        fig.add_trace(
            go.Scatter(
                x=[px_coord],
                y=[py_coord],
                mode="markers+text",
                marker=dict(symbol="star", size=18, color="#FFB703", line=dict(width=2, color="#000000")),
                text=[f"Current Customer: {p_name}"],
                textposition="top center",
                name="Evaluated Customer"
            )
        )

    fig.update_layout(
        legend_title_text="Customer Segments",
        hoverlabel=dict(bgcolor="white", font_size=12),
        margin=dict(l=20, r=20, t=50, b=20),
        height=550
    )
    return fig

def plot_3d_pca(df: pd.DataFrame) -> go.Figure:
    """
    Renders 3D PCA cluster visualization with Plotly.
    """
    # Sample if large for smooth 3D rotation in browser
    sample_df = df.sample(min(len(df), 2500), random_state=42) if len(df) > 2500 else df

    fig = px.scatter_3d(
        sample_df,
        x="PCA_3D_1",
        y="PCA_3D_2",
        z="PCA_3D_3",
        color="Segment_Name",
        hover_name="Customer_ID",
        hover_data={
            "Location": True,
            "Total_Spend_INR": ":,.0f",
            "Total_Bookings": True,
            "Engagement_Score": True,
            "PCA_3D_1": False,
            "PCA_3D_2": False,
            "PCA_3D_3": False
        },
        color_discrete_sequence=CLUSTER_PALETTE,
        title="3D Customer Behavioral Space",
        template="plotly_white",
        opacity=0.75
    )

    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
        height=620,
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
    )
    return fig

def plot_cluster_distribution(profiles: dict) -> go.Figure:
    """
    Donut chart of customer distribution across segments.
    """
    labels = [v["segment_name"] for v in profiles.values()]
    counts = [v["customer_count"] for v in profiles.values()]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=counts,
                hole=0.45,
                marker=dict(colors=CLUSTER_PALETTE),
                textinfo="label+percent",
                hoverinfo="label+value+percent"
            )
        ]
    )
    fig.update_layout(
        title="Customer Distribution by Segment",
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20),
        height=380,
        showlegend=False
    )
    return fig

def plot_radar_comparison(profiles: dict, selected_segments: List[str]) -> go.Figure:
    """
    Multi-segment behavioral radar chart.
    """
    categories = [
        "Avg Spend (Normalized)",
        "Avg Bookings",
        "Engagement Score",
        "Recency (Inverse)",
        "Discount Dependency"
    ]

    # Find max values across all profiles for normalization
    max_spend = max([v["metrics"]["avg_spend_inr"] for v in profiles.values()]) or 1
    max_orders = max([v["metrics"]["avg_orders"] for v in profiles.values()]) or 1
    max_eng = max([v["metrics"]["avg_engagement_score"] for v in profiles.values()]) or 1
    max_rec = max([v["metrics"]["avg_recency_days"] for v in profiles.values()]) or 1

    fig = go.Figure()

    for idx, (cid, prof) in enumerate(profiles.items()):
        name = prof["segment_name"]
        if selected_segments and name not in selected_segments:
            continue

        m = prof["metrics"]
        # Normalize between 0 and 100
        val_spend = (m["avg_spend_inr"] / max_spend) * 100
        val_orders = (m["avg_orders"] / max_orders) * 100
        val_eng = (m["avg_engagement_score"] / max_eng) * 100
        val_rec_inv = (1 - (m["avg_recency_days"] / max_rec)) * 100  # higher = more recent
        val_disc = m["avg_discount_usage"] * 100

        values = [val_spend, val_orders, val_eng, val_rec_inv, val_disc]
        values.append(values[0])  # close the radar loop

        color = CLUSTER_PALETTE[idx % len(CLUSTER_PALETTE)]

        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                fill="toself",
                fillcolor=f"rgba{tuple(list(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + [0.25])}",
                line=dict(color=color, width=2),
                name=name
            )
        )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title="Behavioral Radar Comparison",
        template="plotly_white",
        margin=dict(l=40, r=40, t=50, b=40),
        height=450
    )
    return fig

def plot_segment_metric_bars(df: pd.DataFrame, metric_col: str, title: str) -> go.Figure:
    """
    Renders bar chart showing average of selected metric across segments.
    """
    summary = df.groupby("Segment_Name")[metric_col].mean().reset_index()
    summary = summary.sort_values(by=metric_col, ascending=False)

    fig = px.bar(
        summary,
        x="Segment_Name",
        y=metric_col,
        color="Segment_Name",
        color_discrete_sequence=CLUSTER_PALETTE,
        title=title,
        template="plotly_white"
    )
    fig.update_layout(
        showlegend=False,
        xaxis_title="Segment",
        yaxis_title=metric_col.replace("_", " "),
        margin=dict(l=20, r=20, t=40, b=30),
        height=380
    )
    return fig

def plot_elbow_and_silhouette(curves_dict: dict) -> Tuple[go.Figure, go.Figure]:
    """
    Generates Elbow (Inertia) curve and Silhouette score curve.
    """
    k_vals = curves_dict.get("k_values", [2, 3, 4, 5, 6, 7, 8])
    inertias = curves_dict.get("inertias", [])
    silhouettes = curves_dict.get("silhouettes", [])

    # Elbow Plot
    fig_elbow = go.Figure()
    fig_elbow.add_trace(
        go.Scatter(
            x=k_vals,
            y=inertias,
            mode="lines+markers",
            line=dict(color="#1D3557", width=3),
            marker=dict(size=9, color="#E63946"),
            name="Inertia (WCSS)"
        )
    )
    fig_elbow.update_layout(
        title="Elbow Method (Inertia / WCSS vs K)",
        xaxis=dict(title="Number of Clusters (K)", tickmode="linear", dtick=1),
        yaxis=dict(title="Within-Cluster Sum of Squares (Inertia)"),
        template="plotly_white",
        height=380,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    # Silhouette Plot
    fig_sil = go.Figure()
    fig_sil.add_trace(
        go.Scatter(
            x=k_vals,
            y=silhouettes,
            mode="lines+markers",
            line=dict(color="#2A9D8F", width=3),
            marker=dict(size=9, color="#F4A261"),
            name="Silhouette Score"
        )
    )
    fig_sil.update_layout(
        title="Silhouette Score vs K",
        xaxis=dict(title="Number of Clusters (K)", tickmode="linear", dtick=1),
        yaxis=dict(title="Mean Silhouette Coefficient"),
        template="plotly_white",
        height=380,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    return fig_elbow, fig_sil
