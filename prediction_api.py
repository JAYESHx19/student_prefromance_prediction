#!/usr/bin/env python3
"""
FastAPI Server for Student Performance Prediction
This server provides a REST API endpoint for making student performance predictions.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import numpy as np
import uvicorn
from typing import Optional
from contextlib import asynccontextmanager

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
    allow_origins=["*"],  # In production, specify your frontend domain
<<<<<<< HEAD
    allow_credentials=True,
=======
    allow_origin_regex=".*",  # Allow any origin including null/file:// cases when served locally
    allow_credentials=False,  # Must be False when using wildcard origins
>>>>>>> dc5748a80e26c1ae85315f6c3ae94a31ebc1631d
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

# Global variables to store loaded artifacts
model_data = None  # full artifacts dict
model = None  # trained classifier
label_encoders = None  # dict of encoders
scaler = None  # fitted scaler
feature_names = None  # list[str]

def load_model():
    """Load the trained model and preprocessing components once at startup."""
    global model_data, model, label_encoders, scaler, feature_names

    try:
        with open('student_model.pkl', 'rb') as f:
            loaded = joblib.load(f)

        # Persist all artifacts in globals
        model_data = loaded
        model = loaded.get('model')
        label_encoders = loaded.get('label_encoders', {})
        scaler = loaded.get('scaler')
        feature_names = loaded.get('feature_names', [])

        if model is None or scaler is None or not feature_names:
            raise RuntimeError('Model artifacts are incomplete')

        print("Model loaded successfully!")
        print(f"Model type: {type(model).__name__}")
        return True

    except FileNotFoundError:
        print("Error: Model file 'student_model.pkl' not found!")
        return False
    except Exception as e:
        print(f"Error loading model: {e}")
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
            'gender': 'gender'
        }

        # Convert frontend data to model format
        model_data_dict = {}
        for frontend_key, model_key in field_mapping.items():
            if frontend_key in student_data:
                model_data_dict[model_key] = student_data[frontend_key]

        # Add default values for missing features that the model expects
        default_values = {
            'math_score': 75,  # Default average score
            'science_score': 75,
            'english_score': 75,
            'history_score': 75
        }

        for key, value in default_values.items():
            if key not in model_data_dict:
                model_data_dict[key] = value

        # Create DataFrame from student data
        df = pd.DataFrame([model_data_dict])

        # Encode categorical features
        for col, encoder in prediction_label_encoders.items():
            if col in df.columns:
                try:
                    df[col] = encoder.transform(df[col].astype(str))
                except ValueError as e:
                    # Handle unknown categories by using the most frequent category
                    df[col] = encoder.transform([encoder.classes_[0]])[0]
        
        # Ensure all features are present and in correct order
        for feature in required_feature_names:
            if feature not in df.columns:
                df[feature] = 0  # Default value for missing features
        
        # Reorder columns to match training data
        df = df[required_feature_names]
        
        # Scale features
        df_scaled = prediction_scaler.transform(df)
        
        # Make prediction
        prediction = prediction_model.predict(df_scaled)[0]
        prediction_proba = prediction_model.predict_proba(df_scaled)[0]
        
        # Map prediction to risk category
        risk_mapping = {0: 'High', 1: 'Medium', 2: 'Low'}
        risk_category = risk_mapping[prediction]
        
        # Calculate a score display (0-100 scale) based on probabilities
        # Higher probability of Low risk = higher score
        score_display = (prediction_proba[2] * 40 + prediction_proba[1] * 25 + prediction_proba[0] * 10) + 50
        score_display = min(100, max(0, score_display))  # Ensure 0-100 range
        
        return {
            'predicted_category': risk_category,
            'score_display': score_display,
            'probabilities': {
                'High Risk': float(prediction_proba[0]),
                'Medium Risk': float(prediction_proba[1]),
                'Low Risk': float(prediction_proba[2])
            },
            'confidence': float(max(prediction_proba))
        }
        
    except Exception as e:
        print(f"Error making prediction: {e}")
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
        
        return {
            "success": True,
            "predicted_category": result['predicted_category'],
            "score_display": result['score_display'],
            "probabilities": result['probabilities'],
            "confidence": result['confidence'],
            "message": "Prediction completed successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
<<<<<<< HEAD
    import uvicorn
    import os
    print("Starting Student Performance Prediction API...")
    print("API will be available at: http://127.0.0.1:8000")
    print("Health check: http://127.0.0.1:8000/health")
    print("API documentation: http://127.0.0.1:8000/docs")
    print("Press Ctrl+C to stop the server")
    uvicorn.run("prediction_api:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)
=======
    print("Starting Student Performance Prediction API...")
    print("API will be available at: http://127.0.0.1:8000")
    print("Health check: http://127.0.0.1:8000/health")
    print("API docs: http://127.0.0.1:8000/docs")
    print("\nPress Ctrl+C to stop the server")
    
    uvicorn.run(app, host="127.0.0.1", port=8000)
>>>>>>> dc5748a80e26c1ae85315f6c3ae94a31ebc1631d
