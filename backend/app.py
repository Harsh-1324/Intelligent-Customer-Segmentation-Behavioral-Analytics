"""
FastAPI application entry point for Intelligent Customer Segmentation Platform.
"""

import io
from typing import Dict, Any, List
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from backend.schemas import (
    CustomerInput,
    SegmentationResponse,
    HealthResponse,
    ClusterProfileItem
)
from backend.preprocessing import transform_customer_input
from backend.predictor import predictor

app = FastAPI(
    title="Intelligent Customer Segmentation API",
    description="Production ML inference backend for Pheraas Event & Celebration Analytics Platform.",
    version="1.0.0"
)

# Enable CORS for Streamlit frontend or web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """
    Returns system health status and model artifact loading state.
    """
    return {
        "status": "healthy" if predictor.is_loaded else "degraded",
        "model_loaded": predictor.is_loaded,
        "total_clusters": len(predictor.profiles),
        "version": "1.0.0"
    }

@app.get("/cluster-info", response_model=List[ClusterProfileItem], tags=["Clustering"])
def get_cluster_info():
    """
    Retrieves statistical profiles, averages, customer counts, and marketing strategies for each segment.
    """
    if not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Clustering models are not loaded. Run training first.")
    
    info_list = list(predictor.profiles.values())
    return info_list

@app.post("/segment", response_model=SegmentationResponse, tags=["Inference"])
def segment_single_customer(customer: CustomerInput):
    """
    Accepts individual customer behavioral inputs, scales features, predicts segment,
    and returns cluster ID, distance to centroid, and tailored business recommendations.
    """
    if not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Clustering models are not loaded. Run training first.")

    try:
        feature_df = transform_customer_input(customer)
        result = predictor.predict_single(feature_df)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference error: {str(e)}")

@app.post("/batch-segment", tags=["Inference"])
async def batch_segment_customers(file: UploadFile = File(...)):
    """
    Accepts a CSV upload of customer records, computes segment predictions for all rows,
    and returns an annotated CSV with Cluster, Segment_Name, and Distance_From_Centroid.
    """
    if not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Clustering models are not loaded. Run training first.")

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are supported.")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        annotated_df = predictor.predict_batch(df)

        output_stream = io.StringIO()
        annotated_df.to_csv(output_stream, index=False)
        output_stream.seek(0)

        response = StreamingResponse(
            io.BytesIO(output_stream.getvalue().encode("utf-8")),
            media_type="text/csv"
        )
        response.headers["Content-Disposition"] = f"attachment; filename=segmented_{file.filename}"
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch processing error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
