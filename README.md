# Student Performance Dashboard with Firebase Integration

A modern, responsive dashboard for predicting student academic performance with Firebase backend integration.

## Features

### 🔐 Authentication
- **Firebase Authentication** with email/password
- User registration and login
- Password reset functionality
- Persistent login sessions
- Secure logout

### 📊 Data Management
- **Firestore Database** integration
- Save prediction data to cloud
- User profile management
- Real-time data synchronization
- Secure data access rules

### 🎯 Prediction System
- Collect student performance data
- Store predictions in Firestore
- Track prediction history
- Risk level assessment

### 🎨 Modern UI
- Dark/Light theme support
- Responsive design
- Interactive charts
- Loading states and error handling
- Beautiful animations

## Firebase Setup Instructions

### 1. Create Firebase Project
1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Click "Create a project" or select existing project
3. Follow the setup wizard

### 2. Add Web App
1. In your Firebase project, click "Add app" (</> icon)
2. Select "Web"
3. Register app with a nickname (e.g., "student-dashboard")
4. Copy the `firebaseConfig` object

### 3. Update Configuration
Replace the placeholder values in `mini.html`:

```javascript
const firebaseConfig = {
    apiKey: "your-actual-api-key",
    authDomain: "your-project-id.firebaseapp.com",
    projectId: "your-project-id",
    storageBucket: "your-project-id.appspot.com",
    messagingSenderId: "your-sender-id",
    appId: "your-app-id"
};
```

### 4. Enable Authentication
1. In Firebase Console, go to **Authentication**
2. Click **Sign-in method**
3. Enable **Email/Password** provider
4. Optionally enable **Password reset**

### 5. Set Up Firestore Database
1. Go to **Firestore Database**
2. Click **Create database**
3. Choose **Start in test mode** (for development)
4. Select a location for your database

### 6. Security Rules (Optional)
In Firestore Database > Rules, add these security rules:

```javascript
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    // Allow users to read/write their own data
    match /users/{userId} {
      allow read, write: if request.auth != null && request.auth.uid == userId;
    }
    
    // Allow users to read/write their own predictions
    match /predictions/{predictionId} {
      allow read, write: if request.auth != null && 
        request.auth.uid == resource.data.userId;
    }
  }
}
```

## Usage

### Authentication
- **Login**: Use email/password to sign in
- **Register**: Click "Sign up" to create new account
- **Password Reset**: Click "Forgot Password" to reset

### Making Predictions
1. Navigate to **Predictions** page
2. Fill in student performance data:
   - Previous Grade GPA
   - Attendance percentage
   - Assignments completed
   - Weekly study hours
   - Parental education level
   - Socio-economic status
   - Extracurricular activities
   - Tutor availability
   - School travel time
   - Internet access
3. Click **Run Prediction**
4. Data is saved to Firestore with user association

### Study Materials
- Browse curated educational resources
- Direct links to Khan Academy, Coursera, etc.
- Organized by subject and difficulty

## File Structure

```
mini-project/
├── mini.html              # Main dashboard file
├── firebase-config.js     # Firebase configuration template
└── README.md             # This file
```

## Firebase Collections

### Users Collection
```javascript
{
  email: "user@example.com",
  role: "student",
  createdAt: timestamp,
  profile: {
    name: "User Name",
    grade: "10th",
    school: "School Name"
  }
}
```

### Predictions Collection
```javascript
{
  userId: "user-uid",
  userEmail: "user@example.com",
  predictionData: {
    previousGpa: 3.2,
    attendance: 92,
    assignmentsCompleted: 85,
    studyHours: 10,
    // ... other fields
  },
  predictedScore: 78,
  riskLevel: "Medium Risk",
  timestamp: timestamp
}
```

## Features Added

### ✅ Firebase Integration
- [x] Firebase Authentication
- [x] Firestore Database
- [x] User registration/login
- [x] Password reset
- [x] Data persistence
- [x] Security rules

### ✅ Enhanced UI/UX
- [x] Loading states
- [x] Error handling
- [x] Form validation
- [x] Success messages
- [x] Responsive design

### ✅ Data Management
- [x] Save predictions to cloud
- [x] User-specific data
- [x] Real-time updates
- [x] Data validation

## Next Steps

1. **Replace Firebase Config**: Update the configuration in `mini.html` with your actual Firebase project details
2. **Test Authentication**: Try registering and logging in with different accounts
3. **Test Predictions**: Fill out the prediction form and verify data is saved to Firestore
4. **Customize**: Add more fields, validation, or features as needed
5. **Deploy**: Host your dashboard on Firebase Hosting or any web server

## Troubleshooting

### Common Issues
- **Authentication errors**: Check if Email/Password is enabled in Firebase Console
- **Database errors**: Ensure Firestore is created and rules allow read/write
- **CORS errors**: Make sure your domain is added to authorized domains in Firebase Console

### Debug Mode
Open browser console to see detailed error messages and Firebase logs.

## Security Notes

- Never commit your actual Firebase API keys to version control
- Use environment variables in production
- Set up proper Firestore security rules
- Enable Firebase App Check for additional security

---

**Happy coding! 🚀** 