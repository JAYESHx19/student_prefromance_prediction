// Firebase Configuration Template
// Replace the placeholder values with your actual Firebase project configuration

window.firebaseConfig = {
    apiKey: "AIzaSyDVWw8FqNJQho7V1fKieeCsbUAEfFckyno",
    authDomain: "student-prediction-568af.firebaseapp.com",
    projectId: "student-prediction-568af",
    storageBucket: "student-prediction-568af.appspot.com",
    messagingSenderId: "360498732807",
    appId: "1:360498732807:web:e3f2b87a69ea8e022a2fa2"
};

// Instructions to set up Firebase:
// 1. Go to https://console.firebase.google.com/
// 2. Create a new project or select an existing one
// 3. Click on "Add app" and select Web
// 4. Register your app with a nickname
// 5. Copy the firebaseConfig object from the provided code
// 6. Include this file in your HTML BEFORE initializing Firebase
// 7. Enable Authentication in Firebase Console (Email/Password or Google)
// 8. Set up Firestore Database (optional)

// Security Rules for Firestore (optional):
/*
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
*/
