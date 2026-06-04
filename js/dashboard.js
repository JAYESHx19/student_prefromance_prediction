/* ============================================
   dashboard.js — Dashboard page logic
   ============================================ */

window.initDashboard = function () {
  updateWelcomeName();
  loadDashboardData();
  renderRecentActivity();
  if (window.lucide) lucide.createIcons();
};

function updateWelcomeName() {
  const user = window.firebaseAuth?.currentUser;
  let name = 'Student';
  if (user) {
    if (user.displayName && user.displayName.trim()) {
      name = user.displayName.trim();
    } else if (user.email) {
      const raw = user.email.split('@')[0];
      name = raw.charAt(0).toUpperCase() + raw.slice(1);
    }
  }
  const el = document.getElementById('welcome-user-name');
  if (el) el.textContent = name;
}
window.updateWelcomeName = updateWelcomeName;

async function loadDashboardData() {
  const user = window.firebaseAuth?.currentUser;
  if (!user) return;
  try {
    const { getDoc, doc } = window.firebaseDbFunctions;
    const userDoc = await getDoc(doc(window.firebaseDb, 'users', user.uid));
    if (userDoc.exists()) {
      const d = userDoc.data();
      setStatEl('current-gpa',   d.currentGPA    ? d.currentGPA              : '--');
      setStatEl('predicted-gpa', d.predictedGPA  ? d.predictedGPA            : '--');
      setStatEl('study-hours',   d.studyHours    ? d.studyHours + 'h/wk'     : '--');
      setStatEl('risk-level',    d.riskLevel     ? d.riskLevel                : '--');
    }
  } catch (e) {
    console.error('Dashboard data error:', e);
  }
}

function setStatEl(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}

function renderRecentActivity() {
  const container = document.getElementById('recent-activity');
  if (!container) return;
  const user = window.firebaseAuth?.currentUser;
  const name = user?.displayName || 'you';

  const items = [
    { icon: 'user-plus',   color: 'blue',   text: `Welcome, ${name}! Your account is ready.`,         time: 'Just now' },
    { icon: 'bar-chart-2', color: 'purple', text: 'Make your first prediction to see insights.',       time: '' },
    { icon: 'book-open',   color: 'green',  text: 'Explore curated study resources for every subject.',time: '' },
  ];

  container.innerHTML = items.map(item => `
    <div class="flex items-center gap-3 p-3 rounded-lg bg-[var(--bg-subtle)] transition hover:bg-[var(--border)]">
      <div class="w-8 h-8 rounded-lg bg-${item.color}-100 dark:bg-${item.color}-900/30 flex items-center justify-center flex-shrink-0">
        <i data-lucide="${item.icon}" class="w-4 h-4 text-${item.color}-600 dark:text-${item.color}-400"></i>
      </div>
      <div class="flex-1 min-w-0">
        <p class="text-sm text-[var(--text-primary)] truncate">${item.text}</p>
        ${item.time ? `<p class="text-xs text-[var(--text-muted)]">${item.time}</p>` : ''}
      </div>
    </div>
  `).join('');
  if (window.lucide) lucide.createIcons();
}

window.updateDashboardWithPrediction = function (result) {
  const label = result.predicted_category || '';
  const riskSimple = /high/i.test(label) ? 'High Risk' : /medium/i.test(label) ? 'Medium Risk' : 'Low Risk';

  setStatEl('risk-level', riskSimple);

  const gpaScore = ((result.score_display / 100) * 3 + 1).toFixed(1);
  setStatEl('predicted-gpa', gpaScore);
};
