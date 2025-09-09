#!/usr/bin/env python3
"""
Demo Script: Student Performance Prediction
This script demonstrates how to use the saved model for making predictions.
"""

import pandas as pd
import joblib
import numpy as np

def load_model():
    """Load the saved model and preprocessing objects."""
    try:
        model_data = joblib.load('student_model.pkl')
        return model_data
    except FileNotFoundError:
        print("❌ Error: 'student_model.pkl' not found!")
        print("   Please run 'student_performance_analysis.py' first to train and save the model.")
        return None

def make_prediction(model_data, student_data):
    """Make prediction for a single student."""
    try:
        model = model_data['model']
        scaler = model_data['scaler']
        label_encoders = model_data['label_encoders']
        feature_names = model_data['feature_names']
        
        # Create DataFrame from student data
        df = pd.DataFrame([student_data])
        
        # Encode categorical features
        for col, encoder in label_encoders.items():
            if col in df.columns:
                df[col] = encoder.transform(df[col].astype(str))
        
        # Ensure all features are present and in correct order
        for feature in feature_names:
            if feature not in df.columns:
                df[feature] = 0  # Default value for missing features
        
        # Reorder columns to match training data
        df = df[feature_names]
        
        # Scale features
        df_scaled = scaler.transform(df)
        
        # Make prediction (suppress feature name warnings)
        prediction = model.predict(df_scaled)[0]
        prediction_proba = model.predict_proba(df_scaled)[0]
        
        # Map prediction to risk category
        risk_mapping = {0: 'High Risk', 1: 'Medium Risk', 2: 'Low Risk'}
        risk_category = risk_mapping[prediction]
        
        return risk_category, prediction_proba
        
    except Exception as e:
        print(f"❌ Error making prediction: {e}")
        return None, None

def demo_predictions():
    """Demonstrate predictions with sample student data."""
    print("🎓 Student Performance Prediction Demo")
    print("=" * 50)
    
    # Load the model
    model_data = load_model()
    if model_data is None:
        return
    
    print("✅ Model loaded successfully!")
    
    # Sample student data for demonstration
    sample_students = [
        {
            'name': 'High Risk Student',
            'data': {
                'age': 16,
                'gender': 1,
                'parental_education': 1,
                'socio_economic_status': 'Low',
                'previous_grade_gpa': 2.1,
                'attendance_percentage': 75,
                'assignments_completed': 60,
                'weekly_study_hours': 2,
                'extracurricular_activities': 0,
                'has_tutor': 0,
                'school_travel_time': 45,
                'internet_access': 0,
                'math_score': 55,
                'science_score': 62,
                'english_score': 68,
                'history_score': 58
            }
        },
        {
            'name': 'Medium Risk Student',
            'data': {
                'age': 17,
                'gender': 0,
                'parental_education': 2,
                'socio_economic_status': 'Medium',
                'previous_grade_gpa': 2.9,
                'attendance_percentage': 88,
                'assignments_completed': 85,
                'weekly_study_hours': 6,
                'extracurricular_activities': 2,
                'has_tutor': 0,
                'school_travel_time': 25,
                'internet_access': 1,
                'math_score': 72,
                'science_score': 78,
                'english_score': 80,
                'history_score': 74
            }
        },
        {
            'name': 'Low Risk Student',
            'data': {
                'age': 16,
                'gender': 1,
                'parental_education': 3,
                'socio_economic_status': 'High',
                'previous_grade_gpa': 3.8,
                'attendance_percentage': 98,
                'assignments_completed': 100,
                'weekly_study_hours': 15,
                'extracurricular_activities': 3,
                'has_tutor': 1,
                'school_travel_time': 10,
                'internet_access': 1,
                'math_score': 92,
                'science_score': 95,
                'english_score': 94,
                'history_score': 90
            }
        }
    ]
    
    # Make predictions for each sample student
    for student in sample_students:
        print(f"\n📊 {student['name']}:")
        print("-" * 30)
        
        # Display key metrics
        data = student['data']
        print(f"   GPA: {data['previous_grade_gpa']}")
        print(f"   Attendance: {data['attendance_percentage']}%")
        print(f"   Study Hours: {data['weekly_study_hours']}/week")
        print(f"   Math Score: {data['math_score']}")
        print(f"   Science Score: {data['science_score']}")
        
        # Make prediction
        risk_category, probabilities = make_prediction(model_data, data)
        
        if risk_category:
            print(f"\n🎯 Prediction: {risk_category}")
            print("   Confidence Scores:")
            for i, (risk, prob) in enumerate(zip(['High Risk', 'Medium Risk', 'Low Risk'], probabilities)):
                print(f"     {risk}: {prob:.2%}")
            
            # Provide recommendations
            print("\n💡 Recommendations:")
            if risk_category == 'High Risk':
                print("   - Increase study hours significantly")
                print("   - Improve attendance rate")
                print("   - Consider getting a tutor")
                print("   - Focus on completing all assignments")
                print("   - Seek additional academic support")
            elif risk_category == 'Medium Risk':
                print("   - Maintain current study habits")
                print("   - Focus on weak subjects")
                print("   - Consider additional practice")
                print("   - Set specific academic goals")
            else:  # Low Risk
                print("   - Keep up the excellent work!")
                print("   - Consider advanced courses")
                print("   - Help peers who might be struggling")
                print("   - Explore leadership opportunities")
    
    print("\n" + "=" * 50)
    print("🎉 Demo completed successfully!")
    print("💡 You can now use the saved model for real predictions!")

if __name__ == "__main__":
    demo_predictions()
