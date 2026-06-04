#!/usr/bin/env python3
"""
XGBoost Model Training for Student Performance Prediction
This script trains an XGBoost classifier on student performance data and saves the model.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

class StudentPerformanceXGBoost:
    def __init__(self, data_path='student_performance_data.csv'):
        """Initialize the trainer with data path."""
        self.data_path = data_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.model = None
        self.feature_names = None
        
    def load_data(self):
        """Load the dataset using pandas."""
        try:
            self.data = pd.read_csv(self.data_path)
            print(f"[SUCCESS] Dataset loaded successfully! Shape: {self.data.shape}")
            return True
        except FileNotFoundError:
            print(f"[ERROR] File '{self.data_path}' not found!")
            return False
        except Exception as e:
            print(f"[ERROR] Error loading data: {e}")
            return False
    
    def create_target_variable(self, target_column='final_exam_score', bins=[0, 60, 75, 90, 101], 
                             labels=['High Risk', 'Medium Risk', 'Low Risk', 'Excellent']):
        """Create target variable based on final exam score."""
        if target_column not in self.data.columns:
            print(f"[ERROR] Target column '{target_column}' not found in data")
            return False
            
        self.data['risk_category'] = pd.cut(
            self.data[target_column], 
            bins=bins, 
            labels=labels,
            right=False
        )
        print("[SUCCESS] Created target variable 'risk_category'")
        return True
    
    def preprocess_data(self, test_size=0.2, random_state=42):
        """Preprocess the data: handle missing values, encode categorical features, and scale numerical features."""
        if self.data is None:
            print("[ERROR] No data loaded. Call load_data() first.")
            return False
            
        # Handle missing values
        numeric_cols = self.data.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns
        
        # Impute missing values
        num_imputer = SimpleImputer(strategy='median')
        cat_imputer = SimpleImputer(strategy='most_frequent')
        
        self.data[numeric_cols] = num_imputer.fit_transform(self.data[numeric_cols])
        self.data[categorical_cols] = cat_imputer.fit_transform(self.data[categorical_cols])
        
        # Encode categorical variables
        for col in categorical_cols:
            if col != 'risk_category':  # Skip target variable
                le = LabelEncoder()
                self.data[col] = le.fit_transform(self.data[col].astype(str))
                self.label_encoders[col] = le
        
        # Encode target variable
        le_target = LabelEncoder()
        self.data['risk_category_encoded'] = le_target.fit_transform(self.data['risk_category'])
        self.label_encoders['risk_category'] = le_target
        
        # Prepare features and target
        X = self.data.drop(['risk_category', 'risk_category_encoded'], axis=1, errors='ignore')
        y = self.data['risk_category_encoded']
        
        # Store feature names for later use
        self.feature_names = X.columns.tolist()
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        self.scaler.fit(self.X_train)
        self.X_train = self.scaler.transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print("[SUCCESS] Data preprocessing completed")
        return True
    
    def train_model(self, params=None):
        """Train XGBoost model with optional hyperparameters."""
        if self.X_train is None or self.y_train is None:
            print("[ERROR] No training data available. Call preprocess_data() first.")
            return None
            
        # Default parameters
        default_params = {
            'objective': 'multi:softprob',
            'num_class': len(np.unique(self.y_train)),
            'eval_metric': 'mlogloss',
            'learning_rate': 0.1,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'n_estimators': 100,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Update with user-provided parameters
        if params:
            default_params.update(params)
        
        # Create and train the model
        self.model = xgb.XGBClassifier(**default_params)
        self.model.fit(
            self.X_train, 
            self.y_train,
            eval_set=[(self.X_test, self.y_test)],
            verbose=False
        )
        
        print("[SUCCESS] Model training completed")
        return self.model
    
    def evaluate_model(self):
        """Evaluate the trained model on test data."""
        if self.model is None:
            print("[ERROR] No model trained. Call train_model() first.")
            return None
            
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred, target_names=self.label_encoders['risk_category'].classes_)
        conf_matrix = confusion_matrix(self.y_test, y_pred)
        
        print("\n[INFO] Model Evaluation")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(report)
        print("\nConfusion Matrix:")
        print(conf_matrix)
        
        return {
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': conf_matrix
        }
    
    def save_model(self, output_path='student_performance_xgb.pkl'):
        """Save the trained model and preprocessing objects."""
        if self.model is None:
            print("[ERROR] No model to save. Train a model first.")
            return False
            
        # Create a dictionary of all artifacts
        artifacts = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'model_type': 'xgboost'
        }
        
        # Save to file
        joblib.dump(artifacts, output_path)
        print(f"[SUCCESS] Model saved to {output_path}")
        return True

def main():
    """Main function to run the training pipeline."""
    # Initialize the trainer
    trainer = StudentPerformanceXGBoost()
    
    # Load data
    if not trainer.load_data():
        return
    
    # Create target variable
    if not trainer.create_target_variable():
        return
    
    # Preprocess data
    if not trainer.preprocess_data():
        return
    
    # Train model
    trainer.train_model()
    
    # Evaluate model
    trainer.evaluate_model()
    
    # Save model
    trainer.save_model()
    
    print("\n[SUCCESS] Training pipeline completed successfully!")

if __name__ == "__main__":
    main()
