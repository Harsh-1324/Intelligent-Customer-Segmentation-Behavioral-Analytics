# Intelligent Customer Segmentation & Behavioral Analytics Platform

An enterprise-grade unsupervised machine learning platform designed for customer behavioral analytics and automated segmentation within an event and wedding planning marketplace (**Pheraas** by Skillminds LLP).

The system continuously discovers latent customer groups across transactional, behavioral, and engagement touchpoints without prior labeling, computes optimal cluster configurations, extracts production model artifacts, and serves predictions via a high-performance **FastAPI backend** and an interactive **Streamlit analytics dashboard**.

---

## 1. System Architecture

```
                                +--------------------------------------------+
                                |             Streamlit Dashboard            |
                                |  (Executive, 2D/3D PCA, Segmentation, CSV) |
                                +---------------------+----------------------+
                                                      |
                                                      | HTTP / Direct Fallback
                                                      v
                                +--------------------------------------------+
                                |            FastAPI REST Backend            |
                                |    (/segment, /batch-segment, /cluster)    |
                                +---------------------+----------------------+
                                                      |
                         +----------------------------+----------------------------+
                         |                                                         |
                         v                                                         v
          +-----------------------------+                           +-----------------------------+
          |     Feature Preprocessing   |                           |       Inference Engine      |
          |  (RFM & Engagement Signals) |                           |    (StandardScaler + PCA)   |
          +--------------+--------------+                           +--------------+--------------+
                         |                                                         |
                         +----------------------------+----------------------------+
                                                      |
                                                      v
                                        +---------------------------+
                                        |   Trained Model Artifacts |
                                        |    - clustering_model.pkl |
                                        |    - scaler.pkl           |
                                        |    - pca.pkl / pca_3d.pkl |
                                        |    - cluster_profiles.json|
                                        +---------------------------+
```

---

## 2. Business Problem & Objective

Event planning marketplaces (such as Pheraas, connecting celebration hosts with venues, decorators, caterers, and photographers across Mumbai, Pune, Lonavala, Alibaug, and Nashik) manage diverse user segments:
- **Luxury Destination Wedding Couples**: High budget, low price sensitivity, long planning cycles.
- **Corporate Event Organizers**: Frequent recurring bookings, high cadence, strict SLA expectations.
- **Budget-Conscious Celebrants**: Birthday/intimate party organizers seeking maximum value and package discounts.
- **Active Intimate Planners**: Moderate spend, high vendor inquiry rate, steady conversion.
- **At-Risk / Lapsed Accounts**: High recency (100+ days inactive), incomplete shortlists, churn risk.

Treating all leads uniformly leads to low conversion and inefficient ad spend. This platform automates behavioral grouping and generates tailored marketing strategies for each segment.

---

## 3. Machine Learning Pipeline

### Data & Feature Engineering
From transactional history and web/mobile touchpoints, the pipeline engineers composite behavioral indicators:
1. **Recency ($R$)**: Days since last vendor inquiry or booking.
2. **Frequency ($F$)**: Booking velocity normalized by tenure: $\text{Frequency} = \frac{\text{Total Bookings}}{\max(\text{Tenure Months}, 1)}$.
3. **Monetary ($M$)**: Total gross booking spend (INR ₹) and Average Order Value ($\text{AOV}$).
4. **Composite Engagement Score**: Weighted conversion intent signal:
   $$\text{Engagement} = 0.20 \times \text{Visits} + 0.30 \times \text{App Sessions} + 0.35 \times \text{WhatsApp Inquiries} + 0.15 \times \text{Wishlist}$$
5. **Discount Dependency**: Ratio of discounted bookings to total bookings.
6. **Shortlist Conversion Ratio**: Conversion rate from venue shortlist additions to confirmed bookings.

### Clustering Algorithms & Evaluation
- **K-Means**: Baseline partitioning model. The optimal $K=5$ is verified using the **Elbow Method (WCSS)** and **Silhouette Coefficient Analysis**.
- **DBSCAN**: Density-based clustering isolating core dense clusters from boundary noise and anomalies.
- **Agglomerative Hierarchical Clustering**: Bottom-up agglomeration using Ward linkage and dendrogram trees.
- **Cluster Evaluation Metrics**:
  - **Silhouette Score**: Measures intra-cluster cohesion vs inter-cluster separation ($[-1, 1]$).
  - **Davies-Bouldin Index**: Lower values indicate better separation between clusters.
  - **Calinski-Harabasz Score**: Ratio of between-cluster to within-cluster variance (higher is better).

### Dimensionality Reduction
High-dimensional behavioral vectors are projected using **Principal Component Analysis (PCA)** into 2D and 3D coordinate spaces for interactive visualization and centroid distance estimation.

---

## 4. Discovered Customer Segments

| Cluster ID | Segment Name | Tier | Characteristics | Tactical Strategy |
| :--- | :--- | :--- | :--- | :--- |
| **0** | **High-Value Destination Celebrants** | Platinum | High spend (> ₹10L), high AOV, low discount sensitivity | Dedicated VIP Wedding Concierge, private venue tours in Lonavala/Alibaug, no discount codes |
| **1** | **Corporate & Recurring Event Planners** | Gold | High booking cadence (> 10/yr), moderate spend, high app usage | Multi-event volume retainers, consolidated GST billing, dedicated account manager |
| **2** | **Budget-Conscious Occasional Celebrants** | Silver | Low spend (< ₹75k), high discount dependency (> 60%), high abandonment | Flash venue discounts, bundled starter party packages under ₹1,00,000, EMI options |
| **3** | **Active Intimate Celebration Organizers** | Bronze | Moderate spend, high vendor shortlist interactions, low cancellations | Curated local verified vendor packages, planning checklist guides, milestone payments |
| **4** | **At-Risk / Lapsed Inactive Planners** | At-Risk | High recency (> 150 days), declining sessions, high drop-off | Automated WhatsApp win-back campaigns, ₹5,000 celebration credits, drop-off surveys |

---

## 5. Repository Structure

```
Intelligent-Customer-Segmentation-Behavioral-Analytics/
│
├── data/
│   ├── generate_dataset.py          # Realistic customer transaction & engagement generator
│   └── customer_data.csv            # 6,000+ synthesized records across Maharashtra destinations
│
├── notebooks/
│   ├── 01_EDA.ipynb                 # Exploratory data analysis, distributions & correlation heatmaps
│   ├── 02_Feature_Engineering.ipynb # Behavioral RFM derivation, composite engagement & scaling
│   └── 03_Clustering.ipynb          # K-Means, DBSCAN, Agglomerative, PCA & metric comparison
│
├── training/
│   ├── preprocessing.py             # Feature engineering functions and StandardScaler fitting
│   ├── evaluate_clusters.py         # Silhouette, Davies-Bouldin, CH score and benchmark table
│   ├── cluster_profiling.py         # Centroid statistics, dynamic naming and strategies
│   └── train_clustering.py          # End-to-end training pipeline exporting models
│
├── models/
│   ├── clustering_model.pkl         # Trained optimal K-Means model
│   ├── scaler.pkl                   # Fitted StandardScaler
│   ├── pca.pkl                      # 2-component PCA projection
│   ├── pca_3d.pkl                   # 3-component PCA projection
│   ├── feature_columns.pkl          # Training feature schema
│   ├── cluster_profiles.json        # Segment metadata and operational playbooks
│   ├── evaluation_curves.json       # Elbow WCSS & Silhouette curves across K=2..8
│   └── algorithm_comparison.json    # Benchmark metrics across clustering algorithms
│
├── backend/
│   ├── app.py                       # FastAPI REST application
│   ├── schemas.py                   # Pydantic request/response validation schemas
│   ├── predictor.py                 # Artifact loader and real-time inference manager
│   └── preprocessing.py             # Real-time feature transformer
│
├── frontend/
│   └── streamlit_app.py             # 7-page executive analytics dashboard & predictor
│
├── utils/
│   ├── visualization.py             # Plotly 2D/3D PCA charts, radar comparison & donut charts
│   ├── clustering.py                # Distance and prediction helper utilities
│   └── recommendations.py           # Domain-specific business recommendation engine
│
├── outputs/
│   └── segmented_customers.csv      # Customer dataset annotated with clusters and PCA coordinates
│
├── requirements.txt                 # Python package dependencies
├── .gitignore                       # Standard git ignore configurations
└── README.md                        # Project documentation
```

---

## 6. Installation & Execution Guide

### Prerequisites
- Python 3.9+
- Virtual environment (recommended)

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/Harsh-1324/Intelligent-Customer-Segmentation-Behavioral-Analytics.git
cd Intelligent-Customer-Segmentation-Behavioral-Analytics

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Data & Train Clustering Models
```bash
# 1. Generate 6,000 realistic customer records
python data/generate_dataset.py

# 2. Train K-Means, evaluate algorithms, generate profiles, and export artifacts
python training/train_clustering.py
```

### 3. Launch FastAPI Backend Service
```bash
uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```
Interactive Swagger documentation is available at:
`http://127.0.0.1:8000/docs`

#### REST Endpoints:
- `GET /health`: Health status and artifact verification.
- `GET /cluster-info`: Segment profiles, centroid averages, and marketing recommendations.
- `POST /segment`: Single customer prediction payload returning cluster ID, distance to centroid, and tailored playbook.
- `POST /batch-segment`: Multipart CSV upload returning annotated CSV for download.

### 4. Launch Streamlit Analytics Dashboard
```bash
streamlit run frontend/streamlit_app.py
```
The dashboard features:
1. **Executive Dashboard**: KPI metrics, segment distribution, revenue share.
2. **Customer Segmentation**: Interactive form with preset personas and instant prediction card.
3. **Cluster 2D/3D Visualizer**: Interactive Plotly PCA projections with cluster toggles.
4. **Segment Comparison**: Multi-cluster radar comparisons across RFM metrics.
5. **Clustering Evaluation**: Elbow curve, Silhouette analysis, and benchmark table.
6. **Business Insights & Playbook**: Channel, priority, and tactics for every customer segment.
7. **Batch CSV Segmentation**: Drag-and-drop CSV processing and 1-click export.

---

## 7. Limitations & Future Scope

### Limitations
- K-Means assumes spherical cluster variance, which can slightly oversimplify complex multi-modal consumer behaviors.
- Temporal seasonality (e.g. peak Indian wedding season from November to February) is currently captured through static recency rather than dynamic time-series sequences.

### Future Scope
- **Sequence Modeling (RNN/Transformers)**: Incorporating sequential customer clickstream trajectories to predict next celebration milestones.
- **Automated CRM Webhooks**: Direct integration with WhatsApp Business API and customer communication channels (Braze / MoEngage) to trigger automated win-back workflows.
- **Semi-Supervised Active Learning**: Allowing business relationship managers to provide feedback on borderline customer segment assignments to refine boundary thresholds.
