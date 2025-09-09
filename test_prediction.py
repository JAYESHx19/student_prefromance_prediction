import requests
import json

# Test the prediction API
url = "http://127.0.0.1:8000/predict"
test_data = {
    "previousGpa": 3.5,
    "studyHours": 25,
    "absences": 2,
    "attendance": 95,
    "assignmentsCompleted": 85,
    "parentEducation": 3,
    "parentalEducation": 3,
    "familySupport": 1,
    "socioEconomicStatus": "Medium",
    "extracurricular": 1,
    "extracurricularActivities": 1,
    "hasTutor": 0,
    "travelTime": 20,
    "internetAccess": 1,
    "age": 17,
    "gender": 1
}

try:
    response = requests.post(url, json=test_data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"Error: {e}")
