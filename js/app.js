/* ============================================
   app.js — Core app: auth, navigation, sidebar,
             chatbot, toasts, study materials
   ============================================ */

document.addEventListener('DOMContentLoaded', () => {

  /* ── DOM refs ─────────────────────────────── */
  const loginPage    = document.getElementById('login-page');
  const signupPage   = document.getElementById('signup-page');
  const dashboardMain = document.getElementById('dashboard-main');
  const loginForm    = document.getElementById('login-form');
  const signupForm   = document.getElementById('signup-form');
  const logoutButton = document.getElementById('logout-button');
  const sidebar      = document.getElementById('sidebar');
  const sidebarToggle = document.getElementById('sidebar-toggle');
  const pageTitle    = document.getElementById('page-title');
  const contentPages = document.querySelectorAll('.content-page');
  const navContainer = document.querySelector('#sidebar nav');
  const mobileOverlay = document.getElementById('sidebar-overlay');

  const emailInput    = document.getElementById('email-input');
  const passwordInput = document.getElementById('password-input');
  const loginButton   = document.getElementById('login-button');
  const loginText     = document.getElementById('login-text');
  const loginLoading  = document.getElementById('login-loading');
  const authError     = document.getElementById('auth-error');
  const signupLink    = document.getElementById('signup-link');
  const forgotPassword = document.getElementById('forgot-password');

  const signupFirstname = document.getElementById('signup-firstname-input');
  const signupEmail     = document.getElementById('signup-email-input');
  const signupPassword  = document.getElementById('signup-password-input');
  const signupConfirm   = document.getElementById('signup-confirm-password-input');
  const signupButton    = document.getElementById('signup-button');
  const signupText      = document.getElementById('signup-text');
  const signupLoading   = document.getElementById('signup-loading');
  const signupError     = document.getElementById('signup-error');
  const loginLink       = document.getElementById('login-link');

  const googleSigninBtn = document.getElementById('google-signin-btn');
  const googleText      = document.getElementById('google-text');
  const googleLoading   = document.getElementById('google-loading');

  const chatbotToggle   = document.getElementById('chatbot-toggle');
  const chatbotWindow   = document.getElementById('chatbot-window');
  const chatbotClose    = document.getElementById('chatbot-close');
  const chatbotMessages = document.getElementById('chatbot-messages');
  const chatbotInput    = document.getElementById('chatbot-input');
  const chatbotSend     = document.getElementById('chatbot-send');
  const chatbotQuick    = document.getElementById('chatbot-quick');

  const goToPredictionsBtn    = document.getElementById('go-to-predictions');
  const goToStudyMaterialsBtn = document.getElementById('go-to-study-materials');

  /* ── State ────────────────────────────────── */
  let activePage       = 'dashboard';
  let currentUserRole  = 'student';
  let currentUser      = null;
  window.__subscriptionStatus = 'free';

  /* ── Navigation data ─────────────────────── */
  const navLinks = {
    student: [
      { id: 'dashboard',       icon: 'home',         label: 'Dashboard'      },
      { id: 'predictions',     icon: 'bar-chart-2',  label: 'Predictions'    },
      { id: 'study-materials', icon: 'book-open',    label: 'Study Materials'},
      { id: 'recommendations', icon: 'lightbulb',    label: 'Recommendations'},
      { id: 'reports',         icon: 'file-text',    label: 'Reports'        },
      { id: 'subscription',    icon: 'credit-card',  label: 'Subscription'   },
      { id: 'settings',        icon: 'settings',     label: 'Settings'       },
    ],
  };

  /* ── Toast system ────────────────────────── */
  let toastContainer = document.getElementById('toast-container');
  if (!toastContainer) {
    toastContainer = document.createElement('div');
    toastContainer.id = 'toast-container';
    document.body.appendChild(toastContainer);
  }

  window.showToast = function (message, type = 'info') {
    const icons = { success: 'check-circle', error: 'x-circle', info: 'info' };
    const colors = { success: '#10b981', error: '#ef4444', info: '#3b82f6' };
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `
      <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none"
        stroke="${colors[type]||colors.info}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"
        style="flex-shrink:0"><circle cx="12" cy="12" r="10"/></svg>
      <span style="flex:1">${message}</span>`;
    toastContainer.appendChild(toast);
    setTimeout(() => {
      toast.classList.add('removing');
      setTimeout(() => toast.remove(), 250);
    }, 3500);
  };

  /* ── Show/hide page helpers ──────────────── */
  const showEl  = (el, displayVal = 'flex') => { if (el) el.style.display = displayVal; };
  const hideEl  = (el)                       => { if (el) el.style.display = 'none'; };
  const showLoading = (btn, textEl, spinEl) => {
    if (textEl) textEl.classList.add('hidden');
    if (spinEl) spinEl.classList.remove('hidden');
    if (btn) btn.disabled = true;
  };
  const hideLoading = (btn, textEl, spinEl) => {
    if (textEl) textEl.classList.remove('hidden');
    if (spinEl) spinEl.classList.add('hidden');
    if (btn) btn.disabled = false;
  };
  const showAuthError = (msg) => {
    if (!authError) return;
    authError.textContent = msg;
    authError.classList.remove('hidden');
    setTimeout(() => authError.classList.add('hidden'), 5000);
  };
  const showSignupError = (msg) => {
    if (!signupError) return;
    signupError.textContent = msg;
    signupError.classList.remove('hidden');
    setTimeout(() => signupError.classList.add('hidden'), 5000);
  };
  const clearLoginForm = () => {
    if (emailInput) emailInput.value = '';
    if (passwordInput) passwordInput.value = '';
    authError?.classList.add('hidden');
  };
  const clearSignupForm = () => {
    [signupFirstname, signupEmail, signupPassword, signupConfirm].forEach(el => { if (el) el.value = ''; });
    signupError?.classList.add('hidden');
  };

  /* ── Auth functions ───────────────────────── */
  async function signInWithEmail(email, password) {
    const cred = await window.firebaseAuthFunctions.signInWithEmailAndPassword(window.firebaseAuth, email, password);
    currentUser = cred.user;
    return cred;
  }

  async function createUserWithEmail(email, password, firstName) {
    const cred = await window.firebaseAuthFunctions.createUserWithEmailAndPassword(window.firebaseAuth, email, password);
    currentUser = cred.user;
    await window.firebaseAuthFunctions.updateProfile(cred.user, { displayName: firstName });
    const { setDoc, doc, serverTimestamp } = window.firebaseDbFunctions;
    await setDoc(doc(window.firebaseDb, 'users', cred.user.uid), {
      uid: cred.user.uid, email, displayName: firstName, role: 'student',
      createdAt: serverTimestamp(), subscription_status: 'free',
      profile: { name: firstName, grade: '', school: '' },
    }, { merge: true });
    return cred;
  }

  async function signInWithGoogle() {
    const { GoogleAuthProvider, signInWithPopup, signInWithRedirect } = window.firebaseAuthFunctions;
    const provider = new GoogleAuthProvider();
    provider.setCustomParameters({ prompt: 'select_account' });
    provider.addScope('email'); provider.addScope('profile');
    let cred;
    try {
      cred = await signInWithPopup(window.firebaseAuth, provider);
    } catch (err) {
      if (err.code === 'auth/popup-blocked' || err.code === 'auth/popup-closed-by-user') {
        await signInWithRedirect(window.firebaseAuth, provider); return;
      }
      throw err;
    }
    currentUser = cred.user;

    // Use getDoc by UID (not a collection query) to avoid Firestore permission errors.
    // Wrap in try/catch so a Firestore rules issue never kills the sign-in flow.
    try {
      const { getDoc, setDoc, doc, serverTimestamp } = window.firebaseDbFunctions;
      const userRef  = doc(window.firebaseDb, 'users', cred.user.uid);
      const userSnap = await getDoc(userRef);
      if (!userSnap.exists()) {
        const firstName = (cred.user.displayName || 'User').split(' ')[0];
        await window.firebaseAuthFunctions.updateProfile(cred.user, { displayName: firstName });
        await setDoc(userRef, {
          uid: cred.user.uid, email: cred.user.email, displayName: firstName,
          photoURL: cred.user.photoURL, role: 'student', createdAt: serverTimestamp(),
          subscription_status: 'free', profile: { name: firstName, grade: '', school: '' },
        }, { merge: true });
      }
    } catch (firestoreErr) {
      // Firestore write failed (e.g. rules not set up yet) — auth still succeeded.
      console.warn('Firestore profile upsert skipped:', firestoreErr.code || firestoreErr.message);
    }

    return cred;
  }

  /* ── Sidebar ──────────────────────────────── */
  const renderSidebar = (role) => {
    const links = navLinks[role] || [];
    navContainer.innerHTML = '';
    links.forEach(link => {
      const a = document.createElement('a');
      a.href = '#';
      a.dataset.page = `page-${link.id}`;
      a.className = 'sidebar-link';
      if (`page-${link.id}` === `page-${activePage}`) a.classList.add('active');
      a.innerHTML = `
        <i data-lucide="${link.icon}" style="width:18px;height:18px;flex-shrink:0"></i>
        <span class="sidebar-label">${link.label}</span>`;
      navContainer.appendChild(a);
    });
    if (window.lucide) lucide.createIcons();
  };

  /* ── Navigation ───────────────────────────── */
  window.navigateToPage = function (pageId) {
    activePage = pageId.replace('page-', '');
    contentPages.forEach(p => p.classList.remove('active'));
    document.getElementById(pageId)?.classList.add('active');
    renderSidebar(currentUserRole);

    // Update header title
    const match = (navLinks[currentUserRole] || []).find(l => l.id === activePage);
    if (pageTitle && match) pageTitle.textContent = match.label;

    // Close mobile sidebar
    sidebar?.classList.remove('mobile-open');
    mobileOverlay?.classList.add('hidden');

    // Page-specific init
    const pageInitMap = {
      'page-dashboard':       () => window.initDashboard?.(),
      'page-predictions':     () => window.initPredictionsPage?.(),
      'page-study-materials': () => initStudyMaterialsPage(),
      'page-recommendations': () => {
        if (window.lastPredictionResult && window.lastPredictionInputs) {
          window.generateRecommendations?.(window.lastPredictionResult, window.lastPredictionInputs, {
            listElId: 'recs-list', planElId: 'recs-plan',
            planDaysElId: 'recs-plan-days', riskChipId: 'recs-risk-chip', emptyNoteId: 'recs-empty-note',
          });
        }
        bindOnce('recs-refresh', () => {
          if (window.lastPredictionResult) {
            window.generateRecommendations?.(window.lastPredictionResult, window.lastPredictionInputs, {
              listElId: 'recs-list', planElId: 'recs-plan',
              planDaysElId: 'recs-plan-days', riskChipId: 'recs-risk-chip', emptyNoteId: 'recs-empty-note',
            });
            window.showToast('Refreshed!', 'success');
          } else {
            window.showToast('Make a prediction first', 'error');
          }
        });
      },
      'page-reports':       () => window.initReportsPage?.(),
      'page-subscription':  () => window.initSubscriptionPage?.(),
      'page-payment':       () => window.initPaymentPage?.(),
      'page-payment-success': () => window.initPaymentSuccessPage?.(),
      'page-settings':      () => window.initSettingsPage?.(),
    };
    pageInitMap[pageId]?.();
    if (window.lucide) lucide.createIcons();
  };

  function bindOnce(id, fn) {
    const el = document.getElementById(id);
    if (el && el.dataset.bound !== 'true') { el.dataset.bound = 'true'; el.addEventListener('click', fn); }
  }

  /* ── Auth state ───────────────────────────── */
  window.firebaseAuthFunctions.onAuthStateChanged(window.firebaseAuth, async (user) => {
    if (user) {
      currentUser = user;
      hideEl(loginPage);
      hideEl(signupPage);
      showEl(dashboardMain, 'block');

      // Avatar + name
      const avatar = document.getElementById('user-avatar');
      const uname  = document.getElementById('user-name');
      if (avatar) {
        avatar.src = user.photoURL ||
          `https://placehold.co/40x40/4f8eff/ffffff?text=${encodeURIComponent((user.displayName || user.email || 'U').charAt(0).toUpperCase())}`;
      }
      if (uname) uname.textContent = user.displayName || user.email || 'User';

      // Load subscription — silently ignore Firestore permission errors
      try {
        const { getDoc, doc } = window.firebaseDbFunctions;
        const snap = await getDoc(doc(window.firebaseDb, 'users', user.uid));
        if (snap.exists()) {
          window.__subscriptionStatus = snap.data().subscription_status || 'free';
        }
      } catch (e) {
        console.warn('Could not load subscription status:', e.code || e.message);
      }

      renderSidebar(currentUserRole);
      window.navigateToPage('page-dashboard');
    } else {
      currentUser = null;
      hideEl(dashboardMain);
      showEl(loginPage, 'flex');
      clearLoginForm();
    }
  });

  /* ── Auth events ──────────────────────────── */
  loginForm?.addEventListener('submit', async e => {
    e.preventDefault();
    const email    = emailInput.value.trim();
    const password = passwordInput.value.trim();
    if (!email || !password) { showAuthError('Please enter email and password'); return; }
    showLoading(loginButton, loginText, loginLoading);
    try {
      await signInWithEmail(email, password);
    } catch (err) {
      hideLoading(loginButton, loginText, loginLoading);
      const msgs = {
        'auth/user-not-found':  'No account with this email.',
        'auth/wrong-password':  'Incorrect password.',
        'auth/invalid-email':   'Invalid email address.',
        'auth/too-many-requests': 'Too many attempts. Try again later.',
        'auth/invalid-credential': 'Invalid email or password.',
      };
      showAuthError(msgs[err.code] || 'Login failed. Please try again.');
    }
  });

  signupForm?.addEventListener('submit', async e => {
    e.preventDefault();
    const firstName = signupFirstname?.value.trim();
    const email     = signupEmail?.value.trim();
    const password  = signupPassword?.value.trim();
    const confirm   = signupConfirm?.value.trim();
    if (!firstName || !email || !password || !confirm) { showSignupError('Please fill all fields'); return; }
    if (password.length < 6) { showSignupError('Password must be at least 6 characters'); return; }
    if (password !== confirm) { showSignupError('Passwords do not match'); return; }
    showLoading(signupButton, signupText, signupLoading);
    try {
      await createUserWithEmail(email, password, firstName);
    } catch (err) {
      hideLoading(signupButton, signupText, signupLoading);
      const msgs = {
        'auth/email-already-in-use': 'Account already exists with this email.',
        'auth/invalid-email':        'Invalid email address.',
        'auth/weak-password':        'Password is too weak.',
      };
      showSignupError(msgs[err.code] || 'Signup failed. Please try again.');
    }
  });

  logoutButton?.addEventListener('click', async e => {
    e.preventDefault();
    try { await window.firebaseAuthFunctions.signOut(window.firebaseAuth); } catch (_) {}
  });

  forgotPassword?.addEventListener('click', async e => {
    e.preventDefault();
    const email = emailInput?.value.trim();
    if (!email) { showAuthError('Enter your email address first'); return; }
    try {
      await window.firebaseAuthFunctions.sendPasswordResetEmail(window.firebaseAuth, email);
      showAuthError('Password reset email sent!');
    } catch (err) {
      showAuthError('Failed to send reset email. Please try again.');
    }
  });

  googleSigninBtn?.addEventListener('click', async e => {
    e.preventDefault();
    showLoading(googleSigninBtn, googleText, googleLoading);
    try {
      await signInWithGoogle();
      // Auth state change handles UI — just reset the button
      hideLoading(googleSigninBtn, googleText, googleLoading);
    } catch (err) {
      hideLoading(googleSigninBtn, googleText, googleLoading);
      const msgs = {
        'auth/popup-closed-by-user':            'Sign-in was cancelled.',
        'auth/popup-blocked':                   'Popup blocked. Please allow popups.',
        'auth/account-exists-with-different-credential': 'Account exists with different sign-in method.',
        'auth/network-request-failed':          'Network error. Check connection.',
        'auth/unauthorized-domain':             'Domain not authorized. Add localhost to Firebase Console.',
        'auth/cancelled-popup-request':         'Sign-in was cancelled.',
        'auth/too-many-requests':               'Too many attempts. Try later.',
      };
      showAuthError(msgs[err.code] || `Google sign-in failed: ${err.message}`);
    }
  });

  signupLink?.addEventListener('click', e => { e.preventDefault(); clearLoginForm(); hideEl(loginPage); showEl(signupPage, 'flex'); });
  loginLink?.addEventListener('click',  e => { e.preventDefault(); clearSignupForm(); hideEl(signupPage); showEl(loginPage, 'flex'); });

  /* ── Sidebar toggle ───────────────────────── */
  sidebarToggle?.addEventListener('click', () => {
    if (window.innerWidth < 768) {
      sidebar?.classList.toggle('mobile-open');
      mobileOverlay?.classList.toggle('hidden');
    } else {
      sidebar?.classList.toggle('collapsed');
    }
  });
  mobileOverlay?.addEventListener('click', () => {
    sidebar?.classList.remove('mobile-open');
    mobileOverlay?.classList.add('hidden');
  });

  navContainer?.addEventListener('click', e => {
    const link = e.target.closest('.sidebar-link');
    if (link?.dataset.page) window.navigateToPage(link.dataset.page);
  });

  /* ── Dashboard quick-nav ─────────────────── */
  goToPredictionsBtn?.addEventListener('click',    e => { e.preventDefault(); window.navigateToPage('page-predictions'); });
  goToStudyMaterialsBtn?.addEventListener('click', e => { e.preventDefault(); window.navigateToPage('page-study-materials'); });

  /* ── Chatbot ──────────────────────────────── */
  chatbotToggle?.addEventListener('click', () => chatbotWindow?.classList.toggle('hidden'));
  chatbotClose?.addEventListener('click',  () => chatbotWindow?.classList.add('hidden'));

  (function initBot() {
    if (!chatbotMessages || !chatbotInput || !chatbotSend) return;

    const appendMsg = (text, sender = 'bot') => {
      const wrap   = document.createElement('div');
      wrap.style.cssText = `display:flex;${sender === 'user' ? 'justify-content:flex-end' : ''}`;
      const bubble = document.createElement('div');
      bubble.className = sender === 'user' ? 'chat-bubble-user' : 'chat-bubble-bot';
      bubble.innerHTML = `<p style="margin:0;line-height:1.5">${text}</p>`;
      wrap.appendChild(bubble);
      chatbotMessages.appendChild(wrap);
      chatbotMessages.scrollTop = chatbotMessages.scrollHeight;
    };

    const norm = s => (s || '').toLowerCase().trim();
    const reply = msg => {
      const m = norm(msg);
      if (/^(hi|hello|hey|yo|hola|namaste)\b/.test(m))
        return 'Hi! I\'m the EduPredict Helpdesk Bot. Ask me about predictions, inputs, reports, or how to use the app.';
      if (/how are (you|u)/.test(m))
        return 'Doing great! Ready to help. What would you like to know?';
      if (/(what|tell).*(app|platform|project)/.test(m))
        return 'EduPredict is a Student Performance Dashboard. Log in, enter academic data, and get AI-powered predictions, risk levels, and improvement recommendations.';
      if (/how (do i|to) (use|get started|begin)/.test(m))
        return 'Steps: 1) Log in. 2) Go to Predictions. 3) Fill in your academic details. 4) Click Predict Performance to see your score and risk level.';
      if (/(what|which) (inputs|fields|data)/.test(m))
        return 'Required inputs: GPA, attendance, assignments %, study hours, subject scores (Math, Science, English, History), plus personal info like age, gender, parental education, internet access.';
      if (/(how|what).*(prediction|model|work)/.test(m))
        return 'Your inputs are processed by an XGBoost ML model. Features are encoded and scaled, then the model predicts a risk category (High/Medium/Low Risk). A 0–100 display score is computed from the probability output.';
      if (/privacy|data.*(safe|secure)/.test(m))
        return 'Your auth is managed by Firebase Auth. Prediction data is stored in Firestore under your account. No data is shared externally.';
      if (/report|recommendation|study material/.test(m))
        return 'Navigate using the sidebar to access Reports, Recommendations, or Study Materials pages.';
      if (/error|not working|issue|bug|failed/.test(m))
        return 'Try: refresh the page, log out and back in, check your internet, and ensure all fields are filled. If the issue persists, describe the exact error message.';
      if (/contact|support/.test(m))
        return 'Use the Settings page for account management. For technical issues, describe them here and I\'ll guide you.';
      return 'Not sure about that. Try asking: "how to use", "what inputs are needed", "how do predictions work", or "privacy".';
    };

    // Quick suggestion chips
    const suggestions = ['Hi', 'How to use', 'What inputs are needed', 'How predictions work', 'Reports & Recommendations', 'Privacy'];
    if (chatbotQuick) {
      chatbotQuick.innerHTML = '';
      suggestions.forEach(q => {
        const chip = document.createElement('button');
        chip.type = 'button';
        chip.style.cssText = 'flex-shrink:0;padding:4px 10px;border-radius:99px;font-size:.75rem;font-weight:500;background:var(--bg-subtle);color:var(--text-secondary);border:1px solid var(--border);cursor:pointer;white-space:nowrap;transition:background .15s';
        chip.textContent = q;
        chip.addEventListener('click', () => { chatbotInput.value = q; chatbotSend.click(); });
        chatbotQuick.appendChild(chip);
      });
    }

    const handleSend = () => {
      const text = chatbotInput.value.trim();
      if (!text) return;
      appendMsg(text, 'user');
      chatbotInput.value = '';
      setTimeout(() => appendMsg(reply(text), 'bot'), 220);
    };
    chatbotSend?.addEventListener('click', handleSend);
    chatbotInput?.addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); handleSend(); } });
  })();

  /* ── Study materials filter ───────────────── */
  function initStudyMaterialsPage() {
    setupStudyMaterialsFilters();
    if (window.lucide) lucide.createIcons();
  }
  window.initStudyMaterialsPage = initStudyMaterialsPage;

  function setupStudyMaterialsFilters() {
    const searchInput  = document.getElementById('search-resources');
    const catFilter    = document.getElementById('filter-category');
    const resetBtn     = document.getElementById('reset-filters');
    const cards        = document.querySelectorAll('.resource-card');
    const container    = document.getElementById('resources-container');

    const filter = () => {
      const q   = searchInput?.value.toLowerCase() || '';
      const cat = catFilter?.value || 'all';
      let count = 0;
      cards.forEach(card => {
        const title = card.querySelector('h4')?.textContent.toLowerCase() || '';
        const desc  = card.querySelector('p')?.textContent.toLowerCase()  || '';
        const cats  = (card.dataset.category || '').split(' ');
        const show  = (q === '' || title.includes(q) || desc.includes(q)) &&
                      (cat === 'all' || cats.includes(cat));
        card.style.display = show ? 'flex' : 'none';
        if (show) count++;
      });
      let noMsg = document.getElementById('no-results-message');
      if (count === 0 && container) {
        if (!noMsg) {
          noMsg = document.createElement('div');
          noMsg.id = 'no-results-message';
          noMsg.className = 'fade-in';
          noMsg.style.cssText = 'grid-column:1/-1;text-align:center;padding:48px 0;color:var(--text-muted)';
          noMsg.innerHTML = `<i data-lucide="search-x" style="width:48px;height:48px;display:block;margin:0 auto 12px"></i>
            <p style="font-size:1rem;font-weight:600">No resources found</p>
            <p style="font-size:.85rem;margin-top:4px">Try different search terms or category</p>`;
          container.appendChild(noMsg);
          if (window.lucide) lucide.createIcons();
        }
      } else if (noMsg) noMsg.remove();
    };

    searchInput?.addEventListener('input', filter);
    catFilter?.addEventListener('change', filter);
    resetBtn?.addEventListener('click', () => {
      if (searchInput) searchInput.value = '';
      if (catFilter)   catFilter.value   = 'all';
      filter();
    });
  }

  /* ── Upgrade FAB injection ────────────────── */
  function injectUpgradeFAB() {
    if (document.getElementById('upgrade-fab')) return;
    const fab = document.createElement('a');
    fab.id = 'upgrade-fab';
    fab.href = '#';
    fab.className = 'fixed bottom-6 z-50 hidden';
    fab.innerHTML = `<button class="btn btn-primary" style="border-radius:99px;box-shadow:0 4px 14px rgba(59,130,246,.4)">⭐ Upgrade</button>`;
    fab.addEventListener('click', e => { e.preventDefault(); window.navigateToPage('page-payment'); });
    document.body.appendChild(fab);
  }
  injectUpgradeFAB();

});
