import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# Load the model and preprocessing components
def load_model():
    try:
        # Load the entire model dictionary
        model_data = joblib.load('student_model.pkl')
        return (
            model_data.get('model'),
            model_data.get('scaler'),
            model_data.get('label_encoders', {}),
            model_data.get('feature_names', [])
        )
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, []

# Initialize the model and components
model, scaler, label_encoders, feature_names = load_model()

# Custom CSS
st.markdown("""
    <style>
    .main-header {font-size: 2.5rem; color: #1E88E5; text-align: center; margin-bottom: 1rem;}
    .prediction-card {padding: 2rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 2rem;}
    .success {background-color: #E8F5E9; border-left: 5px solid #4CAF50;}
    .warning {background-color: #FFF8E1; border-left: 5px solid #FFC107;}
    .danger {background-color: #FFEBEE; border-left: 5px solid #F44336;}
    </style>
""", unsafe_allow_html=True)

def make_prediction(input_data):
    try:
        if model is None or scaler is None:
            return {"error": "Model not loaded correctly. Please check the model file."}
        
        # Convert input data to DataFrame
        df = pd.DataFrame([input_data])
        
        # Encode categorical features
        for col, encoder in label_encoders.items():
            if col in df.columns:
                try:
                    df[col] = encoder.transform(df[col].astype(str))
                except ValueError:
                    # Handle unknown categories by using the most frequent category
                    df[col] = encoder.transform([encoder.classes_[0]])[0]
        
        # Ensure all features are present and in correct order
        for feature in feature_names:
            if feature not in df.columns:
                df[feature] = 0  # Default value for missing features
        
        # Reorder columns to match training data
        df = df[feature_names]
        
        # Scale features
        df_scaled = scaler.transform(df)
        
        # Make prediction
        prediction = model.predict(df_scaled)[0]
        prediction_proba = model.predict_proba(df_scaled)[0]
        
        # Map prediction to risk category
        risk_mapping = {0: 'High', 1: 'Medium', 2: 'Low'}
        risk_category = risk_mapping.get(prediction, 'Medium')
        
        # Calculate a score display (0-100 scale)
        score_display = (prediction_proba[2] * 40 + prediction_proba[1] * 25 + prediction_proba[0] * 10) + 50
        score_display = min(100, max(0, score_display))
        
        return {
            "success": True,
            "predicted_category": risk_category,
            "score_display": score_display,
            "probabilities": {
                "High Risk": float(prediction_proba[0]),
                "Medium Risk": float(prediction_proba[1]),
                "Low Risk": float(prediction_proba[2])
            },
            "confidence": float(max(prediction_proba))
        }
        
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

def main():
    st.markdown("<h1 class='main-header'>🎓 Student Performance Predictor</h1>", unsafe_allow_html=True)
    
    # Check if model is loaded
    if model is None or scaler is None:
        st.error("⚠️ Error: Could not load the prediction model. Please make sure 'student_model.pkl' exists in the same directory.")
        return
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Make Prediction", "About"])
    
    if page == "Make Prediction":
        st.header("Enter Student Details")
        
        # Create form
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Academic Information")
                previous_gpa = st.slider("Previous GPA (0-4.0)", 0.0, 4.0, 3.0, 0.1)
                attendance = st.slider("Attendance (%)", 0, 100, 85)
                assignments_completed = st.slider("Assignments Completed (%)", 0, 100, 80)
                study_hours = st.slider("Weekly Study Hours", 0, 40, 15)
                
            with col2:
                st.subheader("Personal Information")
                parental_education = st.selectbox("Parental Education Level", 
                                               ["1 - Primary", "2 - Secondary", "3 - High School", 
                                                "4 - Bachelor's", "5 - Master's or higher"], 
                                               index=2)
                socio_economic = st.selectbox("Socio-Economic Status", 
                                           ["Low", "Medium", "High"], 
                                           index=1)
                extracurricular = st.slider("Extracurricular Activities (per week)", 0, 5, 2)
                has_tutor = st.checkbox("Has a tutor")
                internet_access = st.checkbox("Has internet access at home", value=True)
                age = st.slider("Age", 16, 18, 17)
                gender = st.radio("Gender", ["Male", "Female"])
            
            # Form submission button
            submitted = st.form_submit_button("Predict Performance")
            
            if submitted:
                # Prepare data for prediction
                prediction_data = {
                    "previous_grade_gpa": previous_gpa,
                    "attendance_percentage": attendance,
                    "assignments_completed": assignments_completed,
                    "weekly_study_hours": study_hours,
                    "parental_education": int(parental_education[0]),
                    "socio_economic_status": socio_economic,
                    "extracurricular_activities": extracurricular,
                    "has_tutor": 1 if has_tutor else 0,
                    "school_travel_time": 20,  # Default value
                    "internet_access": 1 if internet_access else 0,
                    "age": age,
                    "gender": 1 if gender == "Male" else 0
                }
                
                with st.spinner("Analyzing student data..."):
                    result = make_prediction(prediction_data)
                    
                    if result.get("success"):
                        st.balloons()
                        
                        # Display results
                        st.markdown("## 📊 Prediction Results")
                        
                        # Determine card style based on prediction
                        risk_level = result.get("predicted_category", "Medium")
                        if risk_level == "High":
                            card_class = "warning"
                            emoji = "⚠️"
                        elif risk_level == "Low":
                            card_class = "success"
                            emoji = "✅"
                        else:
                            card_class = "info"
                            emoji = "ℹ️"
                        
                        # Display prediction card
                        st.markdown(f"""
                        <div class="prediction-card {card_class}">
                            <h3>{emoji} Predicted Performance: {risk_level} Risk</h3>
                            <p>Confidence: <strong>{result.get('confidence', 0) * 100:.1f}%</strong></p>
                            <p>Score: <strong>{result.get('score_display', 0):.1f}/100</strong></p>
                            
                            <h4>Probabilities:</h4>
                            <div style="margin-top: 10px;">
                                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                                    <span>Low Risk:</span>
                                    <span><strong>{result.get('probabilities', {}).get('Low Risk', 0) * 100:.1f}%</strong></span>
                                </div>
                                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                                    <span>Medium Risk:</span>
                                    <span><strong>{result.get('probabilities', {}).get('Medium Risk', 0) * 100:.1f}%</strong></span>
                                </div>
                                <div style="display: flex; justify-content: space-between;">
                                    <span>High Risk:</span>
                                    <span><strong>{result.get('probabilities', {}).get('High Risk', 0) * 100:.1f}%</strong></span>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Display recommendations based on risk level
                        st.markdown("## 📝 Recommendations")
                        if risk_level == "High":
                            st.warning("""
                            ### 🚨 High Risk Detected
                            - Consider increasing study hours
                            - Seek additional tutoring or academic support
                            - Meet with academic advisors to create an improvement plan
                            - Focus on completing all assignments on time
                            - Attend all classes and participate actively
                            """)
                        elif risk_level == "Medium":
                            st.info("""
                            ### ℹ️ Medium Risk Detected
                            - Maintain consistent study habits
                            - Focus on areas needing improvement
                            - Consider joining study groups
                            - Stay on top of assignments and deadlines
                            - Attend all classes regularly
                            """)
                        else:
                            st.success("""
                            ### ✅ Low Risk Detected
                            - Continue with current study habits
                            - Consider taking on advanced coursework
                            - Help peers who might be struggling
                            - Maintain good attendance and participation
                            - Set new academic goals
                            """)
                        
                        # Save prediction history
                        if 'predictions' not in st.session_state:
                            st.session_state.predictions = []
                        
                        prediction_entry = {
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'data': prediction_data,
                            'result': result
                        }
                        st.session_state.predictions.append(prediction_entry)
                        
                    else:
                        st.error(f"Error: {result.get('error', 'Unknown error occurred')}")
    
    else:  # About page
        st.header("About This App")
        st.markdown("""
        ## Student Performance Predictor
        
        This application uses machine learning to predict student academic performance based on various factors.
        
        ### How It Works
        1. Fill in the student's academic and personal information
        2. Click 'Predict Performance' to get instant predictions
        3. View the predicted risk level and recommendations
        
        ### Model Information
        - **Model Type**: Machine Learning Classifier
        - **Features**: GPA, attendance, study hours, and more
        
        ### Need Help?
        For support, please contact the academic support team.
        """)

if __name__ == "__main__":
    main()
