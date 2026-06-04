/* ============================================
   reports.js — Report generation + print/PDF
   ============================================ */

window.initReportsPage = function () {
  setupReportButtons();
  if (window.lastPredictionResult && window.lastPredictionInputs) {
    generateReport(window.lastPredictionResult, window.lastPredictionInputs);
  }
  if (window.lucide) lucide.createIcons();
};

function setupReportButtons() {
  bindOnce('generate-report-btn', () => {
    if (window.lastPredictionResult && window.lastPredictionInputs) {
      generateReport(window.lastPredictionResult, window.lastPredictionInputs);
      window.showToast('Report generated!', 'success');
    } else {
      window.showToast('Make a prediction first', 'error');
    }
  });

  bindOnce('print-report-btn', () => {
    if (document.getElementById('report-content')?.classList.contains('hidden')) {
      window.showToast('Generate a report first', 'error');
      return;
    }
    window.print();
  });

  bindOnce('download-report-btn', () => {
    if (document.getElementById('report-content')?.classList.contains('hidden')) {
      window.showToast('Generate a report first', 'error');
      return;
    }
    downloadReportAsPDF();
  });
}

function bindOnce(id, fn) {
  const el = document.getElementById(id);
  if (el && el.dataset.bound !== 'true') {
    el.dataset.bound = 'true';
    el.addEventListener('click', fn);
  }
}

function generateReport(result, inputs) {
  const reportContent = document.getElementById('report-content');
  const reportEmpty   = document.getElementById('report-empty');
  if (!result || !inputs) {
    reportContent?.classList.add('hidden');
    reportEmpty?.classList.remove('hidden');
    return;
  }

  reportContent?.classList.remove('hidden');
  reportEmpty?.classList.add('hidden');

  const user = window.firebaseAuth?.currentUser;
  const studentName = user?.displayName || user?.email?.split('@')[0] || 'Student';

  // Header
  const dateEl = document.getElementById('report-date');
  if (dateEl) dateEl.textContent = `Generated on: ${new Date().toLocaleString()}`;
  const nameEl = document.getElementById('report-student-name');
  if (nameEl) nameEl.textContent = studentName;

  // Score
  const score = Math.round(result.score_display);
  const scoreEl = document.getElementById('report-score');
  if (scoreEl) scoreEl.textContent = `${score}%`;

  // Risk badge
  const label      = result.predicted_category || '';
  const riskSimple = /high/i.test(label) ? 'High' : /medium/i.test(label) ? 'Medium' : 'Low';
  const badge      = document.getElementById('report-risk-badge');
  if (badge) {
    badge.textContent = `${riskSimple} Risk`;
    badge.className = 'badge ' + (riskSimple === 'High' ? 'badge-red' : riskSimple === 'Medium' ? 'badge-yellow' : 'badge-green');
    badge.style.fontSize = '1rem';
    badge.style.padding  = '8px 20px';
  }

  // Probability bars
  const highRisk   = Math.round(((result.probabilities?.['High Risk'])   || 0) * 100);
  const mediumRisk = Math.round(((result.probabilities?.['Medium Risk']) || 0) * 100);
  const lowRisk    = Math.round(((result.probabilities?.['Low Risk'])    || 0) * 100);

  setBar('report-high-risk',   'report-high-risk-bar',   highRisk,   '#ef4444');
  setBar('report-medium-risk', 'report-medium-risk-bar', mediumRisk, '#f59e0b');
  setBar('report-low-risk',    'report-low-risk-bar',    lowRisk,    '#10b981');

  // Academic profile
  const academicFields = {
    previousGpa: 'Previous GPA', attendance: 'Attendance %', assignmentsCompleted: 'Assignments %',
    studyHours: 'Weekly Study Hours', mathScore: 'Math Score', scienceScore: 'Science Score',
    englishScore: 'English Score', historyScore: 'History Score',
  };
  const academicContainer = document.getElementById('report-academic-inputs');
  if (academicContainer) {
    academicContainer.innerHTML = Object.entries(academicFields)
      .filter(([k]) => inputs[k] !== undefined && inputs[k] !== null)
      .map(([k, lbl]) => {
        let val = inputs[k];
        if (k === 'previousGpa') val = `${val} / 4.0`;
        else if (['attendance','assignmentsCompleted','mathScore','scienceScore','englishScore','historyScore'].includes(k)) val = `${val}%`;
        else if (k === 'studyHours') val = `${val} hrs/wk`;
        return `
          <div style="background:var(--bg-subtle);padding:12px;border-radius:var(--radius-md)">
            <p style="font-size:.75rem;color:var(--text-muted);margin-bottom:3px">${lbl}</p>
            <p style="font-weight:600;color:var(--text-primary)">${val}</p>
          </div>`;
      }).join('');
  }

  // Personal profile
  const educationLevels = ['None','Primary','Middle School','High School','Bachelor\'s','Master\'s+'];
  const activityLevels  = ['None','1-2 Activities','3+ Activities'];
  const personalFields  = {
    age: 'Age', gender: 'Gender', parentalEducation: 'Parental Education',
    socioEconomicStatus: 'Socio-Economic', internetAccess: 'Internet Access',
    hasTutor: 'Has Tutor', travelTime: 'Travel Time', extracurricularActivities: 'Extracurricular',
  };
  const personalContainer = document.getElementById('report-personal-inputs');
  if (personalContainer) {
    personalContainer.innerHTML = Object.entries(personalFields)
      .filter(([k]) => inputs[k] !== undefined && inputs[k] !== null)
      .map(([k, lbl]) => {
        let val = inputs[k];
        if (k === 'gender') val = val === 1 ? 'Male' : 'Female';
        else if (k === 'internetAccess') val = val === 1 ? 'Yes' : 'No';
        else if (k === 'hasTutor') val = val === 1 ? 'Yes' : 'No';
        else if (k === 'parentalEducation') val = educationLevels[val] || val;
        else if (k === 'extracurricularActivities') val = activityLevels[val] || val;
        else if (k === 'travelTime') val = `${val} min`;
        else if (k === 'age') val = `${val} yrs`;
        return `
          <div style="background:var(--bg-subtle);padding:12px;border-radius:var(--radius-md)">
            <p style="font-size:.75rem;color:var(--text-muted);margin-bottom:3px">${lbl}</p>
            <p style="font-weight:600;color:var(--text-primary)">${val}</p>
          </div>`;
      }).join('');
  }

  // Recommendations
  const recsContainer = document.getElementById('report-recommendations');
  if (recsContainer && window.generateRecommendationsList) {
    const recs = window.generateRecommendationsList(result, inputs);
    recsContainer.innerHTML = recs.map(rec => `
      <div style="display:flex;align-items:flex-start;gap:12px;padding:12px;background:var(--bg-subtle);border-radius:var(--radius-md)">
        <div style="font-size:1.4rem;flex-shrink:0">${rec.emoji}</div>
        <div>
          <h5 style="font-weight:600;font-size:.9rem;color:var(--text-primary);margin-bottom:2px">${rec.title}</h5>
          <p style="font-size:.82rem;color:var(--text-secondary)">${rec.desc}</p>
        </div>
      </div>`).join('');
  }

  // Action plan
  const actionContainer = document.getElementById('report-action-plan');
  if (actionContainer) {
    const isLow = riskSimple === 'Low';
    const days = [
      { d: 'Day 1', t: 'Setup & Plan',       i: '🗂️' },
      { d: 'Day 2', t: 'Focus Session A',     i: '🎧' },
      { d: 'Day 3', t: 'Weak Subject Drill',  i: '🎯' },
      { d: 'Day 4', t: 'Recall + Quiz',       i: '🧠' },
      { d: 'Day 5', t: 'Focus Session B',     i: '⏱️' },
      { d: 'Day 6', t: isLow ? 'Mock Test (Advanced)' : 'Mock Test (Core)', i: '📄' },
      { d: 'Day 7', t: 'Review & Reset',      i: '🔁' },
    ];
    actionContainer.innerHTML = days.map(day => `
      <div class="card" style="padding:14px">
        <p style="font-size:.75rem;color:var(--text-muted);margin-bottom:4px">${day.d}</p>
        <div style="display:flex;align-items:center;gap:8px">
          <span style="font-size:1.2rem">${day.i}</span>
          <span style="font-size:.85rem;font-weight:600;color:var(--text-primary)">${day.t}</span>
        </div>
      </div>`).join('');
  }

  if (window.lucide) lucide.createIcons();
}

function setBar(textId, barId, pct, color) {
  const text = document.getElementById(textId);
  const bar  = document.getElementById(barId);
  if (text) { text.textContent = `${pct}%`; text.style.color = color; }
  if (bar)  { bar.style.width = `${pct}%`; bar.style.background = color; }
}

function downloadReportAsPDF() {
  window.showToast('Opening print dialog — choose "Save as PDF" and enable Background graphics', 'info');
  setTimeout(() => { window.scrollTo(0,0); window.print(); }, 1200);
}
