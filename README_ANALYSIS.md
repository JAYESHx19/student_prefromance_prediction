# Student Performance Analysis and Prediction System

This comprehensive Python script analyzes student performance data and predicts academic risk categories using machine learning models.

## Features

- **Data Preprocessing**: Handles missing values, encodes categorical features, and scales numerical features
- **Model Training**: Trains both Logistic Regression (baseline) and Random Forest Classifier
- **Model Evaluation**: Compares models using accuracy, precision, recall, and F1-score
- **Interactive Prediction**: Takes user input and provides predictions with confidence scores
- **Risk Categories**: Classifies students into High, Medium, and Low risk categories
- **Model Persistence**: Saves the best-performing model for future use

## Dataset

The script uses `student_performance_data.csv` which contains the following features:

### Student Demographics
- `student_id`: Unique identifier
- `age`: Student age (15-20)
- `gender`: Gender (0=Female, 1=Male)
- `parental_education`: Parental education level (0-4)
- `socio_economic_status`: Socio-economic status (Low/Medium/High)

### Academic Performance
- `previous_grade_gpa`: Previous grade GPA (0.0-4.0)
- `attendance_percentage`: Attendance percentage (0-100)
- `assignments_completed`: Number of assignments completed (0-100)
- `weekly_study_hours`: Weekly study hours (0-20)
- `extracurricular_activities`: Number of extracurricular activities (0-4)
- `has_tutor`: Whether student has a tutor (0=No, 1=Yes)
- `school_travel_time`: School travel time in minutes
- `internet_access`: Internet access availability (0=No, 1=Yes)

### Subject Scores
- `math_score`: Math score (0-100)
- `science_score`: Science score (0-100)
- `english_score`: English score (0-100)
- `history_score`: History score (0-100)
- `final_exam_score`: Final exam score (0-100) - **Target Variable**

## Installation

1. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure the dataset is in the same directory**:
   - `student_performance_data.csv`

## Usage

### Running the Complete Analysis

```bash
python student_performance_analysis.py
```

The script will:
1. Load and preprocess the dataset
2. Train Logistic Regression and Random Forest models
3. Evaluate both models and select the best performer
4. Save the best model as `student_model.pkl`
5. Enter interactive prediction mode

### Interactive Prediction Mode

After the models are trained, you can input student information to get predictions:

1. **Enter student demographics** (age, gender, parental education, etc.)
2. **Enter academic metrics** (GPA, attendance, study hours, etc.)
3. **Enter subject scores** (math, science, english, history)
4. **Get prediction results** with risk category and confidence scores
5. **Receive personalized recommendations** based on the risk level

### Example Input

```
Age (15-20): 16
Gender (0=Female, 1=Male): 1
Parental Education (0=None, 1=High School, 2=Some College, 3=Bachelor's, 4=Graduate): 3
Socio-economic Status (Low/Medium/High): Medium
Previous Grade GPA (0.0-4.0): 3.2
Attendance Percentage (0-100): 95
Assignments Completed (0-100): 98
Weekly Study Hours (0-20): 10
Extracurricular Activities (0-4): 2
Has Tutor (0=No, 1=Yes): 1
School Travel Time (minutes): 15
Internet Access (0=No, 1=Yes): 1
Math Score (0-100): 85
Science Score (0-100): 88
English Score (0-100): 92
History Score (0-100): 80
```

### Example Output

```
🎯 Prediction Results:
   Risk Category: Low Risk
   Confidence Scores:
     High Risk: 15.23%
     Medium Risk: 28.45%
     Low Risk: 56.32%

📊 Final Prediction: Low Risk

💡 Recommendations:
   - Keep up the excellent work!
   - Consider advanced courses
   - Help peers who might be struggling
```

## Risk Categories

The system classifies students into three risk categories based on their final exam scores:

- **High Risk** (0-70): Students who may need additional support
- **Medium Risk** (71-85): Students performing adequately but with room for improvement
- **Low Risk** (86-100): Students performing well academically

## Model Performance

The script evaluates models using:
- **Accuracy**: Overall prediction accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

## Files Generated

- `student_model.pkl`: Saved model with preprocessing objects
- Model evaluation reports in the console output

## Technical Details

### Preprocessing Steps
1. **Missing Value Handling**: Median imputation for numerical features
2. **Categorical Encoding**: Label encoding for categorical variables
3. **Feature Scaling**: StandardScaler for numerical features
4. **Data Splitting**: 80% training, 20% testing with stratification

### Models Used
1. **Logistic Regression**: Linear baseline model
2. **Random Forest**: Ensemble model with 100 trees, max depth 10

### Model Selection
The best model is selected based on the highest F1-score, which provides a balanced measure of precision and recall.

## Troubleshooting

### Common Issues

1. **File Not Found Error**:
   - Ensure `student_performance_data.csv` is in the same directory as the script

2. **Import Errors**:
   - Install required packages: `pip install -r requirements.txt`

3. **Memory Issues**:
   - The dataset is relatively small, but if you encounter memory issues, consider reducing the number of trees in Random Forest

### Error Messages

- `❌ Error: File 'student_performance_data.csv' not found!`: Check file location
- `❌ Invalid input for [field]`: Enter valid data types and ranges
- `❌ Error making prediction`: Ensure the model file exists and is not corrupted

## Contributing

Feel free to enhance the script by:
- Adding more machine learning models
- Implementing cross-validation
- Adding visualization capabilities
- Expanding the recommendation system

## License

This project is open source and available under the MIT License.









