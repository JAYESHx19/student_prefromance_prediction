# Troubleshooting Guide

## Google Sign-In Issues

### Error: "auth/unauthorized-domain"

If you're seeing this error when trying to use Google Sign-In, it means your domain is not authorized in Firebase.

#### Quick Fix for Localhost Testing:

1. **Go to Firebase Console**
   - Visit: https://console.firebase.google.com/
   - Select your project: `student-prediction-568af`

2. **Navigate to Authentication Settings**
   - Click on "Authentication" in the left sidebar
   - Click on the "Settings" tab
   - Scroll down to "Authorized domains"

3. **Add localhost**
   - Click "Add domain"
   - Enter: `localhost`
   - Click "Add"

4. **Test Again**
   - Refresh your webpage
   - Try Google Sign-In again

#### For Production Deployment:

When you deploy your website to a real domain (like `yourdomain.com`), you'll need to add that domain to the authorized domains list as well.

### Other Common Issues:

#### 1. Google Sign-In Provider Not Enabled
- Go to Firebase Console > Authentication > Sign-in method
- Enable "Google" provider
- Add your support email

#### 2. Network Issues
- Check your internet connection
- Try refreshing the page
- Clear browser cache

#### 3. Browser Popup Blocked
- Allow popups for your localhost domain
- Try using a different browser

### Testing Steps:

1. **Verify Firebase Configuration**
   - Open browser console (F12)
   - Check for any Firebase initialization errors
   - Ensure all Firebase modules are loaded

2. **Test Authentication**
   - Try email/password login first
   - Then try Google Sign-In
   - Check console for detailed error messages

3. **Check Network Tab**
   - Open browser DevTools > Network tab
   - Try Google Sign-In
   - Look for failed requests to Google/Firebase

### Console Commands for Debugging:

```javascript
// Check if Firebase is initialized
console.log('Firebase Auth:', window.firebaseAuth);
console.log('Firebase DB:', window.firebaseDb);

// Test Google Sign-In manually
const provider = new window.firebaseAuthFunctions.GoogleAuthProvider();
window.firebaseAuthFunctions.signInWithPopup(window.firebaseAuth, provider)
  .then(result => console.log('Success:', result))
  .catch(error => console.log('Error:', error));
```

### Still Having Issues?

1. **Check Firebase Project Settings**
   - Verify your project ID matches: `student-prediction-568af`
   - Confirm your API key is correct
   - Ensure Google Sign-In is enabled

2. **Browser Compatibility**
   - Try Chrome, Firefox, or Edge
   - Disable browser extensions temporarily
   - Try incognito/private mode

3. **Firebase Console Logs**
   - Go to Firebase Console > Authentication > Users
   - Check if authentication attempts are being recorded
   - Look for any error messages in the console

### Contact Information:

If you continue to have issues after following these steps, please provide:
- The exact error message from the browser console
- Your browser and version
- The steps you've already tried
- Screenshots of your Firebase Console settings 