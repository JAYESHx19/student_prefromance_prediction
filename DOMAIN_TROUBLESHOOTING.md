# Domain Authorization Troubleshooting Guide

## Issue: "auth/unauthorized-domain" persists after adding localhost

If you're still getting the unauthorized domain error after adding `localhost` to Firebase Console, try these solutions:

### 1. Check Your Exact Domain

First, check what domain you're actually using:

1. **Open browser console (F12)**
2. **Look for the debug logs** that show your current domain
3. **Note the exact domain** - it might be:
   - `localhost`
   - `127.0.0.1`
   - `localhost:3000` (with port)
   - `127.0.0.1:8080` (with port)

### 2. Add All Possible Localhost Variants

In Firebase Console, add **ALL** of these domains:

```
localhost
127.0.0.1
localhost:3000
localhost:8080
localhost:5000
127.0.0.1:3000
127.0.0.1:8080
127.0.0.1:5000
```

**Steps:**
1. Go to Firebase Console > Authentication > Settings
2. Scroll to "Authorized domains"
3. Add each domain one by one
4. Click "Add" after each one

### 3. Clear Browser Cache and Cookies

1. **Clear all browser data:**
   - Press `Ctrl+Shift+Delete` (Windows) or `Cmd+Shift+Delete` (Mac)
   - Select "All time" for time range
   - Check all boxes
   - Click "Clear data"

2. **Or try incognito/private mode:**
   - Open a new incognito window
   - Navigate to your localhost site
   - Try Google Sign-In

### 4. Check Firebase Project Configuration

Verify your Firebase config matches exactly:

```javascript
const firebaseConfig = {
  apiKey: "AIzaSyDVWw8FqNJQho7V1fKieeCsbUAEfFckyno",
  authDomain: "student-prediction-568af.firebaseapp.com",
  projectId: "student-prediction-568af",
  storageBucket: "student-prediction-568af.firebasestorage.app",
  messagingSenderId: "360498732807",
  appId: "1:360498732807:web:e3f2b87a69ea8e022a2fa2",
  measurementId: "G-24EYM1PR9V"
};
```

### 5. Enable Google Sign-In Provider

Make sure Google Sign-In is enabled:

1. Go to Firebase Console > Authentication > Sign-in method
2. Click on "Google" provider
3. Enable it if not already enabled
4. Add a support email
5. Save

### 6. Check for HTTPS/HTTP Issues

If you're using HTTPS locally, you might need to:

1. **Add HTTPS localhost variants:**
   ```
   https://localhost
   https://127.0.0.1
   ```

2. **Or force HTTP:**
   - Make sure you're accessing via `http://localhost` not `https://localhost`

### 7. Alternative Testing Methods

#### Method 1: Use a Different Port
Try accessing your site on a different port:
- `http://localhost:3000`
- `http://localhost:8080`
- `http://localhost:5000`

#### Method 2: Use IP Address
Instead of `localhost`, try:
- `http://127.0.0.1`
- `http://127.0.0.1:3000`

#### Method 3: Use a Local Server
If you're opening the HTML file directly, try using a local server:

**Using Python:**
```bash
# Python 3
python -m http.server 8000

# Python 2
python -m SimpleHTTPServer 8000
```

**Using Node.js:**
```bash
npx http-server -p 8000
```

Then access via: `http://localhost:8000`

### 8. Browser-Specific Solutions

#### Chrome:
1. Go to `chrome://settings/content/popups`
2. Add `localhost` to allowed sites

#### Firefox:
1. Go to `about:preferences#privacy`
2. Click "Settings" under "Permissions"
3. Add `localhost` to allowed sites

### 9. Check Network Tab

1. Open DevTools > Network tab
2. Try Google Sign-In
3. Look for failed requests to:
   - `identitytoolkit.googleapis.com`
   - `securetoken.googleapis.com`
4. Check the response status codes

### 10. Firebase Console Verification

Double-check in Firebase Console:

1. **Project Settings:**
   - Go to Project Settings (gear icon)
   - Verify your project ID: `student-prediction-568af`

2. **Authentication Settings:**
   - Go to Authentication > Settings
   - Check "Authorized domains" list
   - Make sure `localhost` is there

3. **Sign-in Methods:**
   - Go to Authentication > Sign-in method
   - Ensure "Google" is enabled

### 11. Debug Console Commands

Run these in browser console to debug:

```javascript
// Check current domain
console.log('Hostname:', window.location.hostname);
console.log('Origin:', window.location.origin);

// Test Firebase config
console.log('Firebase config:', {
  authDomain: 'student-prediction-568af.firebaseapp.com',
  projectId: 'student-prediction-568af'
});

// Test Google provider
const provider = new window.firebaseAuthFunctions.GoogleAuthProvider();
console.log('Google provider created:', provider);
```

### 12. Common Solutions That Work

**Most Common Fix:**
1. Add `127.0.0.1` to authorized domains
2. Clear browser cache
3. Try incognito mode

**If Still Not Working:**
1. Use a different browser (Chrome, Firefox, Edge)
2. Disable browser extensions temporarily
3. Try accessing via IP address instead of localhost

### 13. Final Verification

After making changes:

1. **Wait 5-10 minutes** (Firebase changes can take time)
2. **Clear browser cache completely**
3. **Try in incognito mode**
4. **Check console for any new errors**

### Still Having Issues?

If none of the above works, please provide:

1. **Exact error message** from console
2. **Your current domain** (from console logs)
3. **Browser and version**
4. **Screenshot of Firebase Console authorized domains**
5. **Network tab errors** (if any)

This will help identify the specific cause of the issue. 