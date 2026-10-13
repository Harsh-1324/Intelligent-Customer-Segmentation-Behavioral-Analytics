# CelebrateIQ | Intelligent Customer Segmentation & Behavioral Analytics Platform

An end-to-end **unsupervised machine learning platform** for customer behavioral analytics and automated segmentation.

CelebrateIQ analyzes transactional, behavioral, and engagement data to discover meaningful customer groups without predefined labels. The trained ML pipeline is integrated with a **FastAPI backend** for inference and an interactive **Streamlit dashboard** for analytics, visualization, and customer segment prediction.


---
Visit the live demo for a better understanding of the project's features and interactive insights:
https://intelligent-customer-segmentation-behavioral-analytics-7e6mssy.streamlit.app/
---

## 🚀 Key Features

* Automated customer segmentation using **K-Means clustering**
* Comparison with **DBSCAN** and **Agglomerative Clustering**
* RFM-based and engagement-oriented feature engineering
* Cluster evaluation using:

  * Silhouette Score
  * Davies-Bouldin Index
  * Calinski-Harabasz Score
* PCA-based **2D and 3D cluster visualization**
* Real-time customer segment prediction
* Batch customer segmentation using CSV files
* Interactive analytics dashboard
* Segment-level behavioral insights
* REST API-based ML inference using **FastAPI**

---

## 🧩 System Architecture

```text
                         ┌──────────────────────────┐
                         │    Streamlit Dashboard   │
                         │ Analytics & Prediction   │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │     FastAPI Backend      │
                         │     REST API / Inference │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │    ML Inference Engine   │
                         │ Scaling + Clustering     │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │   Trained ML Artifacts   │
                         │ Model + Preprocessing    │
                         └──────────────────────────┘
```

---

## 🎯 Business Problem

Event and celebration marketplaces serve customers with very different behaviors, spending patterns, engagement levels, and booking frequencies.

Treating all customers uniformly can make it difficult to identify:

* High-value customers
* Frequent or recurring planners
* Price-sensitive customers
* Highly engaged customers
* Customers showing signs of inactivity

CelebrateIQ addresses this problem by using **unsupervised machine learning** to automatically discover behavioral customer segments and provide actionable insights for customer engagement and retention.

---

## 🧠 Machine Learning Pipeline

The platform follows an end-to-end unsupervised learning workflow:

```text
Customer Data
      ↓
Data Exploration
      ↓
Feature Engineering
      ↓
RFM & Engagement Features
      ↓
Feature Scaling
      ↓
Clustering Algorithms
      ↓
Cluster Evaluation
      ↓
Optimal Model Selection
      ↓
PCA Visualization
      ↓
Customer Segment Prediction
```

### Feature Engineering

The model uses behavioral features derived from customer activity, including:

* **Recency** — time since the customer's recent activity
* **Frequency** — booking activity relative to customer tenure
* **Monetary Value** — total spending and average order value
* **Engagement Signals** — visits, sessions, inquiries, and wishlist activity
* **Discount Dependency**
* **Shortlist-to-Booking Behavior**

These features are transformed and scaled before being passed to the clustering models.

---

## 🔬 Clustering Approach

### K-Means

K-Means is used as the primary segmentation algorithm to group customers based on behavioral similarity.

The number of clusters is evaluated using techniques such as:

* **Elbow Method**
* **Silhouette Analysis**

### Additional Algorithms

The project also evaluates:

* **DBSCAN** — density-based clustering and anomaly identification
* **Agglomerative Clustering** — hierarchical bottom-up clustering

### Evaluation Metrics

Cluster quality is analyzed using:

| Metric                  | Purpose                                              |
| ----------------------- | ---------------------------------------------------- |
| Silhouette Score        | Measures cluster cohesion and separation             |
| Davies-Bouldin Index    | Evaluates similarity between clusters                |
| Calinski-Harabasz Score | Measures between-cluster and within-cluster variance |

---

## 📊 Customer Segments

The model identifies behavioral groups such as:

| Segment                        | Behavioral Profile                                  |
| ------------------------------ | --------------------------------------------------- |
| **High-Value Customers**       | Higher spending with strong engagement              |
| **Recurring Event Planners**   | Frequent bookings and consistent activity           |
| **Budget-Conscious Customers** | Lower spending with stronger price sensitivity      |
| **Active Planners**            | Moderate spending with high interaction             |
| **At-Risk Customers**          | Declining activity and longer periods of inactivity |

These segments can be used to support targeted engagement, retention, and customer analytics.

---

## 📈 Interactive Dashboard

The Streamlit dashboard provides multiple analytics and prediction views:

### 1. Executive Dashboard

* Customer and segment KPIs
* Segment distribution
* Revenue-level insights

### 2. Customer Segmentation

* Interactive customer input
* Predefined customer personas
* Instant segment prediction

### 3. Cluster Visualization

* Interactive **2D PCA visualization**
* Interactive **3D PCA visualization**
* Cluster-level exploration

### 4. Segment Comparison

* Comparative behavioral analysis
* RFM-based visual comparisons

### 5. Clustering Evaluation

* Elbow analysis
* Silhouette analysis
* Algorithm comparison

### 6. Business Insights

* Segment-level behavioral insights
* Customer engagement recommendations

### 7. Batch Segmentation

* CSV upload
* Multiple customer predictions
* Segmented dataset export

---

## 🛠️ Technology Stack

### Programming

* Python

### Machine Learning

* Scikit-learn
* K-Means
* DBSCAN
* Agglomerative Clustering
* PCA

### Data Processing

* Pandas
* NumPy

### Backend

* FastAPI
* Pydantic
* Uvicorn

### Frontend & Visualization

* Streamlit
* Plotly

### Model Persistence

* Joblib / Pickle

### Development

* Git
* GitHub

---

## 📁 Project Structure

```text
Intelligent-Customer-Segmentation-Behavioral-Analytics/
│
├── data/              # Dataset generation and data processing
├── notebooks/         # EDA, feature engineering and clustering analysis
├── training/          # Model training and evaluation pipelines
├── models/            # Trained ML artifacts
├── backend/           # FastAPI application and inference logic
├── frontend/          # Streamlit analytics dashboard
├── utils/             # Visualization and helper utilities
├── outputs/           # Segmented customer outputs
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🔮 Future Scope

* Sequence-based customer behavior modeling using RNNs or Transformers
* Real-time customer behavior and segmentation updates
* CRM and communication-platform integrations
* Feedback-driven or semi-supervised customer segmentation
* Advanced customer lifetime value and churn analytics

---

## 🎓 Project Outcome

CelebrateIQ demonstrates an end-to-end machine learning workflow:

**Data Generation → Feature Engineering → Unsupervised Learning → Model Evaluation → Model Deployment → API Integration → Interactive Analytics**

The project combines **machine learning, backend API development, data visualization, and business analytics** into a single deployable application.

---
