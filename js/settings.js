/* ============================================
   settings.js — Settings page logic
   ============================================ */

window.initSettingsPage = function () {
  loadUserData();
  setTimeout(setupSettingsEventListeners, 120);
  const saved = localStorage.getItem('theme') ||
    (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  applyTheme(saved);
  const tt = document.getElementById('theme-toggle');
  if (tt) tt.checked = saved === 'dark';
  loadAppearanceSettings();
  if (window.lucide) lucide.createIcons();
};

/* ---------- Profile ---------- */
function loadUserData() {
  const user = window.firebaseAuth?.currentUser;
  if (!user) return;
  const fn = document.getElementById('full-name');
  const em = document.getElementById('user-email');
  if (fn) fn.value = user.displayName || '';
  if (em) em.value = user.email || '';
  loadUserProfile(user.uid);
  loadAppearanceSettings();
}
window.loadUserData = loadUserData;

async function loadUserProfile(uid) {
  try {
    const { getDoc, doc } = window.firebaseDbFunctions;
    const snap = await getDoc(doc(window.firebaseDb, 'users', uid));
    if (!snap.exists()) return;
    const d = snap.data();
    const set = (id, val) => { const el = document.getElementById(id); if (el) el.value = val || ''; };
    set('user-school', d.school);
    set('student-id',  d.studentId);
    set('phone-number',d.phoneNumber);
    if (d.grade) { const el = document.getElementById('user-grade'); if (el) el.value = d.grade; }

    if (d.privacySettings) {
      const ps = d.privacySettings;
      setCheck('data-collection',            ps.dataCollection !== false);
      setCheck('personalized-recommendations',ps.personalizedRecommendations !== false);
      setCheck('performance-monitoring',     ps.performanceMonitoring !== false);
      setCheck('learning-analytics',         ps.learningAnalytics !== false);
      setCheck('share-with-teachers',        ps.shareWithTeachers !== false);
      setCheck('public-profile',             ps.publicProfile === true);
    }
    if (d.notificationSettings) {
      const ns = d.notificationSettings;
      setCheck('email-notifications', ns.emailNotifications !== false);
      setCheck('study-reminders',     ns.studyReminders !== false);
      setCheck('progress-updates',    ns.progressUpdates !== false);
    }
  } catch (e) { console.error('loadUserProfile error:', e); }
}

function setCheck(id, val) {
  const el = document.getElementById(id);
  if (el) el.checked = val;
}

/* ---------- Appearance ---------- */
function loadAppearanceSettings() {
  const saved = localStorage.getItem('theme') || 'auto';
  const radio = document.querySelector(`input[name="theme-mode"][value="${saved}"]`);
  if (radio) radio.checked = true;
  applyTheme(saved);

  const savedSize = localStorage.getItem('fontSize') || '16';
  const sizeDisp = document.getElementById('font-size-display');
  if (sizeDisp) sizeDisp.textContent = savedSize + 'px';
  applyFontSize(savedSize);

  const compact = localStorage.getItem('compactMode') === 'true';
  const compEl = document.getElementById('compact-mode');
  if (compEl) compEl.checked = compact;
  applyCompactMode(compact);
}

function applyTheme(mode) {
  let isDark = mode === 'dark' || (mode === 'auto' && window.matchMedia('(prefers-color-scheme: dark)').matches);
  document.documentElement.classList.toggle('dark', isDark);
  document.documentElement.classList.toggle('light', !isDark);
  document.documentElement.setAttribute('data-theme', isDark ? 'dark' : 'light');
  localStorage.setItem('theme', mode);
}
window.applyTheme = applyTheme;

function applyFontSize(size) {
  document.documentElement.style.fontSize = size + 'px';
  localStorage.setItem('fontSize', size);
}

function applyCompactMode(enabled) {
  document.documentElement.classList.toggle('compact', enabled);
  localStorage.setItem('compactMode', enabled);
}

/* ---------- Validation ---------- */
function validateProfileForm() {
  let ok = true;
  document.querySelectorAll('[id$="-error"]').forEach(el => el.classList.add('hidden'));
  const full   = document.getElementById('full-name')?.value.trim();
  const school = document.getElementById('user-school')?.value.trim();
  const grade  = document.getElementById('user-grade')?.value;
  if (!full)   { showFieldErr('name-error',   'Full name is required');  ok = false; }
  if (!school) { showFieldErr('school-error', 'School is required');     ok = false; }
  if (!grade)  { showFieldErr('grade-error',  'Grade level is required'); ok = false; }
  return ok;
}

function showFieldErr(id, msg) {
  const el = document.getElementById(id);
  if (el) { el.textContent = msg; el.classList.remove('hidden'); }
}

/* ---------- Event Listeners ---------- */
function setupSettingsEventListeners() {
  const editBtn    = document.getElementById('edit-profile-btn');
  const cancelBtn  = document.getElementById('cancel-profile-edit');
  const profileForm = document.getElementById('profile-form');
  if (!profileForm) return;

  const inputs = profileForm.querySelectorAll('input, select');

  editBtn?.addEventListener('click', e => {
    e.preventDefault();
    inputs.forEach(el => { if (el.id !== 'user-email') el.disabled = false; });
    document.getElementById('profile-btns')?.classList.remove('hidden');
    editBtn.classList.add('hidden');
  });

  cancelBtn?.addEventListener('click', () => {
    inputs.forEach(el => el.disabled = true);
    document.getElementById('profile-btns')?.classList.add('hidden');
    editBtn?.classList.remove('hidden');
    loadUserData();
  });

  profileForm.addEventListener('submit', async e => {
    e.preventDefault();
    if (!validateProfileForm()) return;
    const user = window.firebaseAuth?.currentUser;
    if (!user) return;
    const fullName    = document.getElementById('full-name').value;
    const school      = document.getElementById('user-school').value;
    const grade       = document.getElementById('user-grade').value;
    const studentId   = document.getElementById('student-id')?.value || '';
    const phoneNumber = document.getElementById('phone-number')?.value || '';
    try {
      await window.firebaseAuthFunctions.updateProfile(user, { displayName: fullName });
      const { setDoc, doc, serverTimestamp } = window.firebaseDbFunctions;
      await setDoc(doc(window.firebaseDb, 'users', user.uid),
        { displayName: fullName, school, grade, studentId, phoneNumber, updatedAt: serverTimestamp() },
        { merge: true }
      );
      inputs.forEach(el => el.disabled = true);
      document.getElementById('profile-btns')?.classList.add('hidden');
      editBtn?.classList.remove('hidden');
      if (window.updateWelcomeName) window.updateWelcomeName();
      window.showToast('Profile updated!', 'success');
    } catch (err) {
      console.error(err);
      window.showToast('Failed to update profile', 'error');
    }
  });

  // Theme radios
  document.querySelectorAll('input[name="theme-mode"]').forEach(r =>
    r.addEventListener('change', e => applyTheme(e.target.value))
  );

  // Font size controls
  let fontSize = parseInt(localStorage.getItem('fontSize') || '16');
  const sizeDisp = document.getElementById('font-size-display');
  document.getElementById('font-decrease')?.addEventListener('click', () => {
    if (fontSize > 12) { fontSize -= 1; sizeDisp && (sizeDisp.textContent = fontSize + 'px'); applyFontSize(fontSize); }
  });
  document.getElementById('font-increase')?.addEventListener('click', () => {
    if (fontSize < 22) { fontSize += 1; sizeDisp && (sizeDisp.textContent = fontSize + 'px'); applyFontSize(fontSize); }
  });

  // Compact mode
  document.getElementById('compact-mode')?.addEventListener('change', e => applyCompactMode(e.target.checked));

  // Change Password Modal
  document.getElementById('change-password-btn')?.addEventListener('click', () =>
    document.getElementById('change-password-modal')?.classList.remove('hidden')
  );
  ['close-password-modal','cancel-password-change'].forEach(id =>
    document.getElementById(id)?.addEventListener('click', () =>
      document.getElementById('change-password-modal')?.classList.add('hidden')
    )
  );
  document.getElementById('change-password-form')?.addEventListener('submit', async e => {
    e.preventDefault();
    const curr    = document.getElementById('current-password').value;
    const newPwd  = document.getElementById('new-password').value;
    const confirm = document.getElementById('confirm-password').value;
    if (newPwd !== confirm) { window.showToast('Passwords do not match', 'error'); return; }
    const user = window.firebaseAuth?.currentUser;
    if (!user) return;
    try {
      const { reauthenticateWithCredential, EmailAuthProvider, updatePassword } = window.firebaseAuthFunctions;
      const cred = EmailAuthProvider.credential(user.email, curr);
      await reauthenticateWithCredential(user, cred);
      await updatePassword(user, newPwd);
      document.getElementById('change-password-modal')?.classList.add('hidden');
      e.target.reset();
      window.showToast('Password updated!', 'success');
    } catch (err) {
      console.error(err);
      window.showToast(`Failed: ${err.message}`, 'error');
    }
  });

  // 2FA Modal
  document.getElementById('two-factor-btn')?.addEventListener('click', () =>
    document.getElementById('two-factor-modal')?.classList.remove('hidden')
  );
  ['close-two-factor-modal','cancel-2fa-setup'].forEach(id =>
    document.getElementById(id)?.addEventListener('click', () =>
      document.getElementById('two-factor-modal')?.classList.add('hidden')
    )
  );
  document.getElementById('setup-2fa-btn')?.addEventListener('click', () =>
    window.showToast('2FA setup requires additional configuration', 'info')
  );

  // Deactivate Modal
  document.getElementById('deactivate-account-btn')?.addEventListener('click', () =>
    document.getElementById('deactivate-account-modal')?.classList.remove('hidden')
  );
  document.getElementById('cancel-deactivate-account')?.addEventListener('click', () =>
    document.getElementById('deactivate-account-modal')?.classList.add('hidden')
  );
  document.getElementById('confirm-deactivate-account')?.addEventListener('click', async () => {
    try {
      await window.firebaseAuthFunctions.signOut(window.firebaseAuth);
      window.showToast('Account deactivated. You\'ve been signed out.', 'info');
    } catch (e) { console.error(e); }
  });

  // Delete Modal
  document.getElementById('delete-account-btn')?.addEventListener('click', () =>
    document.getElementById('delete-account-modal')?.classList.remove('hidden')
  );
  document.getElementById('cancel-delete-account')?.addEventListener('click', () =>
    document.getElementById('delete-account-modal')?.classList.add('hidden')
  );
  document.getElementById('confirm-delete-account')?.addEventListener('click', async () => {
    const user = window.firebaseAuth?.currentUser;
    if (!user) return;
    try {
      const { deleteDoc, doc } = window.firebaseDbFunctions;
      await deleteDoc(doc(window.firebaseDb, 'users', user.uid));
      await user.delete();
      window.showToast('Account deleted', 'info');
    } catch (err) {
      window.showToast(`Delete failed: ${err.message}`, 'error');
    }
  });

  // Export data
  document.getElementById('export-data-btn')?.addEventListener('click', exportUserData);
}

async function exportUserData() {
  const user = window.firebaseAuth?.currentUser;
  if (!user) return;
  try {
    const { getDoc, doc } = window.firebaseDbFunctions;
    const snap = await getDoc(doc(window.firebaseDb, 'users', user.uid));
    const payload = {
      profile: { displayName: user.displayName, email: user.email, uid: user.uid },
      settings: snap.exists() ? snap.data() : {},
      exportDate: new Date().toISOString(),
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
    const url  = URL.createObjectURL(blob);
    const a    = Object.assign(document.createElement('a'), { href: url, download: `edupredict-data-${user.uid}.json` });
    document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url);
    window.showToast('Data exported!', 'success');
  } catch (e) {
    window.showToast('Export failed', 'error');
  }
}
