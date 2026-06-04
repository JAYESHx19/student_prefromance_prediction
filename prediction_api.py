#!/usr/bin/env python3
"""
FastAPI Server for Student Performance Prediction
This server provides a REST API endpoint for making student performance predictions.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import numpy as np
import uvicorn
import json
from typing import Optional, Any
from contextlib import asynccontextmanager

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                         np.int16, np.int32, np.int64, np.uint8,
                         np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def jsonable_encoder_custom(obj: Any) -> Any:
    return json.loads(json.dumps(obj, cls=NumpyEncoder))

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model when the server starts."""
    success = load_model()
    if not success:
        print("Warning: Model not loaded. Predictions will fail until model is available.")
    yield

# Initialize FastAPI app
app = FastAPI(
    title="Student Performance Prediction API",
    description="API for predicting student academic performance and risk categories",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for prediction request
class StudentData(BaseModel):
    previousGpa: float = Field(..., ge=0, le=4, description="Previous GPA (0.0-4.0)")
    attendance: float = Field(..., ge=0, le=100, description="Attendance percentage (0-100)")
    assignmentsCompleted: float = Field(..., ge=0, le=100, description="Assignments completed percentage (0-100)")
    studyHours: float = Field(..., ge=0, le=25, description="Weekly study hours (0-25)")
    parentalEducation: int = Field(..., ge=0, le=5, description="Parental education level (0-5)")
    socioEconomicStatus: str = Field(..., description="Socio-economic status (Low/Medium/High)")
    extracurricularActivities: int = Field(..., ge=0, le=3, description="Number of extracurricular activities (0-3)")
    hasTutor: int = Field(..., ge=0, le=1, description="Has tutor (0=No, 1=Yes)")
    travelTime: Optional[float] = Field(20, ge=5, le=120, description="Travel time to school in minutes")
    internetAccess: int = Field(..., ge=0, le=1, description="Internet access (0=No, 1=Yes)")
    age: int = Field(..., ge=16, le=18, description="Student age (16-18)")
    gender: int = Field(..., ge=0, le=1, description="Gender (0=Female, 1=Male)")
    # New subject scores (0-100)
    mathScore: float | None = Field(None, ge=0, le=100, description="Math score (0-100)")
    scienceScore: float | None = Field(None, ge=0, le=100, description="Science score (0-100)")
    englishScore: float | None = Field(None, ge=0, le=100, description="English score (0-100)")
    historyScore: float | None = Field(None, ge=0, le=100, description="History score (0-100)")
    # Optional final exam score provided by UI but NOT used as a feature to avoid leakage
    finalExamScore: float | None = Field(None, ge=0, le=100, description="Final exam score (optional, ignored by model)")

# Global variables to store loaded artifacts
model_data = None  # full artifacts dict
model = None  # trained classifier
label_encoders = None  # dict of encoders
scaler = None  # fitted scaler
feature_names = None  # list[str]
model_filename = None  # which file was loaded

def load_model():
    """Load the tuned model and preprocessing components once at startup."""
    global model_data, model, label_encoders, scaler, feature_names, model_filename

    tuned_model_path = 'student_performance_xgb_tuned.pkl'

    try:
        with open(tuned_model_path, 'rb') as f:
            loaded = joblib.load(f)

        # Persist all artifacts in globals
        model_data = loaded
        model = loaded.get('model')
        label_encoders = loaded.get('label_encoders', {})
        scaler = loaded.get('scaler')
        feature_names = loaded.get('feature_names', [])
        model_filename = tuned_model_path

        if model is None or scaler is None or not feature_names:
            raise RuntimeError('Model artifacts are incomplete')

        print(f"Model loaded successfully from '{tuned_model_path}'!")
        print(f"Model type: {type(model).__name__}")
        return True

    except FileNotFoundError:
        print(f"[ERROR] Tuned model file '{tuned_model_path}' not found!")
        print("Please run 'python train_xgboost_tuned.py' to create it.")
        return False
    except Exception as e:
        print(f"[ERROR] Error loading model: {e}")
        return False

def make_prediction(student_data: dict):
    """Make prediction for a single student using preloaded artifacts."""
    try:
        if model is None or scaler is None:
            raise HTTPException(status_code=500, detail="Model not loaded")

        prediction_model = model
        prediction_scaler = scaler
        prediction_label_encoders = label_encoders or {}
        required_feature_names = feature_names or []

        # Map frontend field names to model field names
        field_mapping = {
            'previousGpa': 'previous_grade_gpa',
            'attendance': 'attendance_percentage',
            'assignmentsCompleted': 'assignments_completed',
            'studyHours': 'weekly_study_hours',
            'parentalEducation': 'parental_education',
            'socioEconomicStatus': 'socio_economic_status',
            'extracurricularActivities': 'extracurricular_activities',
            'hasTutor': 'has_tutor',
            'travelTime': 'school_travel_time',
            'internetAccess': 'internet_access',
            'age': 'age',
            'gender': 'gender',
            # New subject fields
            'mathScore': 'math_score',
            'scienceScore': 'science_score',
            'englishScore': 'english_score',
            'historyScore': 'history_score',
            # finalExamScore is intentionally NOT mapped into features
        }

        # Convert frontend data to model format
        model_data_dict = {}
        for frontend_key, model_key in field_mapping.items():
            if frontend_key in student_data:
                model_data_dict[model_key] = student_data[frontend_key]

        # We do NOT default subject scores silently; if missing, they will be filled as 0 below,
        # but UI should provide them for best results.

        print("[DEBUG] Building dataframe from request")
        # Create DataFrame from student data
        df = pd.DataFrame([model_data_dict])

        print(f"[DEBUG] Initial df columns: {list(df.columns)}")
        # Encode categorical features
        for col, encoder in prediction_label_encoders.items():
            if col in df.columns:
                try:
                    df[col] = encoder.transform(df[col].astype(str))
                except ValueError as e:
                    # Handle unknown categories by using the most frequent category
                    print(f"[WARN] Unknown category for {col}, using fallback: {e}")
                    df[col] = encoder.transform([encoder.classes_[0]])[0]
        
        print("[DEBUG] Ensuring all required features present")
        # Ensure all features are present and in correct order
        for feature in required_feature_names:
            if feature not in df.columns:
                df[feature] = 0  # Default value for missing features
        
        print("[DEBUG] Reordering columns to match training data")
        # Reorder columns to match training data
        df = df[required_feature_names]
        
        print("[DEBUG] Scaling features")
        # Scale features (pass numpy array to match scaler fitting without feature names)
        df_scaled = prediction_scaler.transform(df.values)
        
        print("[DEBUG] Making prediction")
        # Make prediction
        prediction = prediction_model.predict(df_scaled)[0]
        prediction_proba = prediction_model.predict_proba(df_scaled)[0]

        print(f"[DEBUG] Raw prediction: {prediction} ({type(prediction)})")
        print(f"[DEBUG] Probabilities: {prediction_proba}")

        # Determine class names using the saved target label encoder if available
        target_le = None
        if isinstance(prediction_label_encoders, dict):
            target_le = prediction_label_encoders.get('risk_category')

        if target_le is not None and hasattr(target_le, 'classes_'):
            class_names = list(target_le.classes_)
        else:
            # Fallback: infer typical order
            class_names = ['High Risk', 'Medium Risk', 'Low Risk', 'Excellent'][:len(prediction_proba)]

        print(f"[DEBUG] Class names: {class_names}")
        # Map numeric prediction to label name safely
        try:
            predicted_label = class_names[int(prediction)]
        except Exception:
            predicted_label = str(prediction)

        # Build probability dict mapped to class names
        probabilities = {class_names[i]: float(prediction_proba[i]) for i in range(len(class_names))}

        # Compute a score 0-100 where better outcomes yield higher values.
        # Define desirability weights per label name; unseen names get medium weight.
        weights = {
            'Excellent': 1.00,
            'Low Risk': 0.85,
            'Medium Risk': 0.50,
            'High Risk': 0.20,
        }
        weighted_sum = 0.0
        for name, proba in probabilities.items():
            weighted_sum += proba * weights.get(name, 0.50)
        score_display = float(max(0.0, min(100.0, weighted_sum * 100.0)))

        result_payload = {
            'predicted_category': predicted_label,
            'score_display': score_display,
            'probabilities': probabilities,
            'confidence': float(max(prediction_proba))
        }
        print(f"[DEBUG] Result payload: {result_payload}")
        return result_payload
        
    except Exception as e:
        print(f"Error making prediction: {e} (type={type(e)})")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Student Performance Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Make a prediction",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    model_loaded = model is not None and scaler is not None and feature_names is not None
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "model_file": model_filename,
        "message": "API is running" if model_loaded else "API is running but model not loaded"
    }

@app.post("/predict")
async def predict(student_data: StudentData):
    """
    Make a prediction for student performance.
    
    Returns the predicted risk category and confidence scores.
    """
    try:
        # Convert Pydantic model to dict
        data_dict = student_data.model_dump()
        
        # Make prediction
        result = make_prediction(data_dict)
        
        # Convert numpy types to native Python types for JSON serialization
        response = {
            "success": True,
            "predicted_category": str(result['predicted_category']),  # Ensure string
            "score_display": float(result['score_display']),  # Convert to float
            "probabilities": {
                "High Risk": float(result['probabilities']['High Risk']),
                "Medium Risk": float(result['probabilities']['Medium Risk']),
                "Low Risk": float(result['probabilities']['Low Risk'])
            },
            "confidence": float(result['confidence']),
            "message": "Prediction completed successfully"
        }
        
        return jsonable_encoder_custom(response)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in predict endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    import os
    print("Starting Student Performance Prediction API...")
    print("API will be available at: http://127.0.0.1:8000")
    print("Health check: http://127.0.0.1:8000/health")
    print("API documentation: http://127.0.0.1:8000/docs")
    print("Press Ctrl+C to stop the server")
    uvicorn.run("prediction_api:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)
