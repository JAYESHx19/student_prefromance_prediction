#!/usr/bin/env python3
"""
Student Performance Analysis and Prediction System
This script analyzes student performance data and predicts academic risk categories.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import joblib
import warnings
warnings.filterwarnings('ignore')

class StudentPerformanceAnalyzer:
    def __init__(self, data_path='student_performance_data.csv'):
        """Initialize the analyzer with data path."""
        self.data_path = data_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.best_model = None
        self.feature_names = None
        
    def load_data(self):
        """Load the dataset using pandas."""
        try:
            self.data = pd.read_csv(self.data_path)
            print(f"✅ Dataset loaded successfully! Shape: {self.data.shape}")
            print(f"📊 Columns: {list(self.data.columns)}")
            return True
        except FileNotFoundError:
            print(f"❌ Error: File '{self.data_path}' not found!")
            return False
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def create_target_variable(self):
        """Create target variable based on final exam score."""
        # Create risk categories based on final exam score
        self.data['risk_category'] = pd.cut(
            self.data['final_exam_score'],
            bins=[0, 70, 85, 100],
            labels=['High', 'Medium', 'Low'],
            include_lowest=True
        )
        
        # Convert to numeric for modeling (0=High risk, 1=Medium risk, 2=Low risk)
        risk_mapping = {'High': 0, 'Medium': 1, 'Low': 2}
        self.data['risk_category_numeric'] = self.data['risk_category'].map(risk_mapping)
        
        print("🎯 Target variable created:")
        print(self.data['risk_category'].value_counts())
        
    def preprocess_data(self):
        """Perform data preprocessing: handle missing values, encode categorical features, and scale numerical features."""
        print("\n🔧 Starting data preprocessing...")
        
        # Create target variable
        self.create_target_variable()
        
        # Separate features and target
        target_cols = ['student_id', 'final_exam_score', 'risk_category', 'risk_category_numeric']
        feature_cols = [col for col in self.data.columns if col not in target_cols]
        
        X = self.data[feature_cols].copy()
        y = self.data['risk_category_numeric']
        
        print(f"📈 Features: {len(feature_cols)}")
        print(f"🎯 Target: risk_category_numeric")
        
        # Handle missing values
        print("\n1️⃣ Handling missing values...")
        missing_counts = X.isnull().sum()
        if missing_counts.sum() > 0:
            print("Missing values found:")
            print(missing_counts[missing_counts > 0])
            
            # Use median imputation for numerical columns
            numerical_cols = X.select_dtypes(include=[np.number]).columns
            imputer = SimpleImputer(strategy='median')
            X[numerical_cols] = imputer.fit_transform(X[numerical_cols])
        else:
            print("✅ No missing values found!")
        
        # Encode categorical features
        print("\n2️⃣ Encoding categorical features...")
        categorical_cols = X.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col in X.columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
                print(f"   Encoded: {col}")
        
        # Scale numerical features
        print("\n3️⃣ Scaling numerical features...")
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        X[numerical_cols] = self.scaler.fit_transform(X[numerical_cols])
        
        # Store feature names for later use
        self.feature_names = X.columns.tolist()
        
        # Split the data
        print("\n4️⃣ Splitting data into training and testing sets...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"   Training set: {self.X_train.shape}")
        print(f"   Testing set: {self.X_test.shape}")
        print("✅ Preprocessing completed!")
        
    def train_models(self):
        """Train Logistic Regression and Random Forest models."""
        print("\n🤖 Training models...")
        
        # Model 1: Logistic Regression (baseline)
        print("\n1️⃣ Training Logistic Regression...")
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        lr_model.fit(self.X_train, self.y_train)
        
        # Model 2: Random Forest Classifier
        print("2️⃣ Training Random Forest Classifier...")
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(self.X_train, self.y_train)
        
        return lr_model, rf_model
    
    def evaluate_models(self, lr_model, rf_model):
        """Compare models using various metrics."""
        print("\n📊 Model Evaluation...")
        
        models = {
            'Logistic Regression': lr_model,
            'Random Forest': rf_model
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\n{name}:")
            print("-" * 50)
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted')
            recall = recall_score(self.y_test, y_pred, average='weighted')
            f1 = f1_score(self.y_test, y_pred, average='weighted')
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'predictions': y_pred
            }
            
            print(f"   Accuracy:  {accuracy:.4f}")
            print(f"   Precision: {precision:.4f}")
            print(f"   Recall:    {recall:.4f}")
            print(f"   F1-Score:  {f1:.4f}")
            
            # Detailed classification report
            print("\n   Classification Report:")
            print(classification_report(self.y_test, y_pred, 
                                      target_names=['High Risk', 'Medium Risk', 'Low Risk']))
        
        return results
    
    def select_best_model(self, results):
        """Select the best performing model based on F1-score."""
        print("\n🏆 Selecting best model...")
        
        best_model_name = max(results.keys(), key=lambda x: results[x]['f1_score'])
        self.best_model = results[best_model_name]['model']
        
        print(f"✅ Best model: {best_model_name}")
        print(f"   F1-Score: {results[best_model_name]['f1_score']:.4f}")
        
        return best_model_name
    
    def save_model(self, model_name):
        """Save the best model using joblib."""
        try:
            # Save the model and preprocessing objects
            model_data = {
                'model': self.best_model,
                'scaler': self.scaler,
                'label_encoders': self.label_encoders,
                'feature_names': self.feature_names
            }
            
            joblib.dump(model_data, 'student_model.pkl')
            print(f"✅ Model saved as 'student_model.pkl'")
            return True
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return False
    
    def analyze_predictions(self, results):
        """Analyze distribution of predicted risk categories."""
        print("\n📈 Prediction Distribution Analysis...")
        
        for name, result in results.items():
            print(f"\n{name} Predictions:")
            predictions = result['predictions']
            
            # Count predictions for each category
            unique, counts = np.unique(predictions, return_counts=True)
            risk_labels = ['High Risk', 'Medium Risk', 'Low Risk']
            
            for i, (pred, count) in enumerate(zip(unique, counts)):
                percentage = (count / len(predictions)) * 100
                print(f"   {risk_labels[pred]}: {count} ({percentage:.1f}%)")
    
    def get_user_input(self):
        """Get user input for prediction."""
        print("\n🎯 Student Performance Prediction")
        print("=" * 50)
        print("Please enter the following student information:")
        
        # Define input fields with their expected ranges/types
        inputs = {
            'age': (int, "Age (15-20): "),
            'gender': (int, "Gender (0=Female, 1=Male): "),
            'parental_education': (int, "Parental Education (0=None, 1=High School, 2=Some College, 3=Bachelor's, 4=Graduate): "),
            'socio_economic_status': (str, "Socio-economic Status (Low/Medium/High): "),
            'previous_grade_gpa': (float, "Previous Grade GPA (0.0-4.0): "),
            'attendance_percentage': (float, "Attendance Percentage (0-100): "),
            'assignments_completed': (int, "Assignments Completed (0-100): "),
            'weekly_study_hours': (int, "Weekly Study Hours (0-20): "),
            'extracurricular_activities': (int, "Extracurricular Activities (0-4): "),
            'has_tutor': (int, "Has Tutor (0=No, 1=Yes): "),
            'school_travel_time': (int, "School Travel Time (minutes): "),
            'internet_access': (int, "Internet Access (0=No, 1=Yes): "),
            'math_score': (int, "Math Score (0-100): "),
            'science_score': (int, "Science Score (0-100): "),
            'english_score': (int, "English Score (0-100): "),
            'history_score': (int, "History Score (0-100): ")
        }
        
        user_data = {}
        
        for field, (data_type, prompt) in inputs.items():
            while True:
                try:
                    value = input(prompt)
                    if data_type == str:
                        user_data[field] = value
                    else:
                        user_data[field] = data_type(value)
                    break
                except ValueError:
                    print(f"❌ Invalid input for {field}. Please try again.")
        
        return user_data
    
    def predict_user_input(self, user_data):
        """Make prediction for user input using the saved model."""
        try:
            # Load the saved model
            model_data = joblib.load('student_model.pkl')
            model = model_data['model']
            scaler = model_data['scaler']
            label_encoders = model_data['label_encoders']
            feature_names = model_data['feature_names']
            
            # Create DataFrame from user input
            df = pd.DataFrame([user_data])
            
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
            
            # Make prediction
            prediction = model.predict(df_scaled)[0]
            prediction_proba = model.predict_proba(df_scaled)[0]
            
            # Map prediction to risk category
            risk_mapping = {0: 'High Risk', 1: 'Medium Risk', 2: 'Low Risk'}
            risk_category = risk_mapping[prediction]
            
            print(f"\n🎯 Prediction Results:")
            print(f"   Risk Category: {risk_category}")
            print(f"   Confidence Scores:")
            for i, (risk, prob) in enumerate(zip(['High Risk', 'Medium Risk', 'Low Risk'], prediction_proba)):
                print(f"     {risk}: {prob:.2%}")
            
            return risk_category, prediction_proba
            
        except Exception as e:
            print(f"❌ Error making prediction: {e}")
            return None, None

def main():
    """Main function to run the complete analysis pipeline."""
    print("🎓 Student Performance Analysis and Prediction System")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = StudentPerformanceAnalyzer()
    
    # Step 1: Load data
    if not analyzer.load_data():
        return
    
    # Step 2: Preprocess data
    analyzer.preprocess_data()
    
    # Step 3: Train models
    lr_model, rf_model = analyzer.train_models()
    
    # Step 4: Evaluate models
    results = analyzer.evaluate_models(lr_model, rf_model)
    
    # Step 5: Select and save best model
    best_model_name = analyzer.select_best_model(results)
    analyzer.save_model(best_model_name)
    
    # Step 6: Analyze predictions
    analyzer.analyze_predictions(results)
    
    # Step 7: Interactive prediction
    print("\n" + "=" * 60)
    print("🚀 Interactive Prediction Mode")
    print("=" * 60)
    
    while True:
        try:
            # Get user input
            user_data = analyzer.get_user_input()
            
            # Make prediction
            risk_category, probabilities = analyzer.predict_user_input(user_data)
            
            if risk_category:
                print(f"\n📊 Final Prediction: {risk_category}")
                
                # Provide recommendations based on risk category
                if risk_category == 'High Risk':
                    print("💡 Recommendations:")
                    print("   - Increase study hours")
                    print("   - Improve attendance")
                    print("   - Consider getting a tutor")
                    print("   - Focus on completing assignments")
                elif risk_category == 'Medium Risk':
                    print("💡 Recommendations:")
                    print("   - Maintain current study habits")
                    print("   - Focus on weak subjects")
                    print("   - Consider additional practice")
                else:  # Low Risk
                    print("💡 Recommendations:")
                    print("   - Keep up the excellent work!")
                    print("   - Consider advanced courses")
                    print("   - Help peers who might be struggling")
            
            # Ask if user wants to continue
            continue_prediction = input("\n🔁 Would you like to make another prediction? (y/n): ").lower()
            if continue_prediction != 'y':
                break
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ An error occurred: {e}")
            continue
    
    print("\n🎉 Thank you for using the Student Performance Analysis System!")

if __name__ == "__main__":
    main()







