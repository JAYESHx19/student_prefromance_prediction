/* ============================================
   predictions.js — Prediction form + results
   ============================================ */

window.initPredictionsPage = function () {
  setupPredictionForm();
  if (window.lucide) lucide.createIcons();
};

function setupPredictionForm() {
  const runBtn  = document.getElementById('run-prediction-btn');
  const fillBtn = document.getElementById('fill-sample-data-btn');

  if (runBtn && runBtn.dataset.bound !== 'true') {
    runBtn.dataset.bound = 'true';
    runBtn.addEventListener('click', handlePredictionSubmit);
  }
  if (fillBtn && fillBtn.dataset.bound !== 'true') {
    fillBtn.dataset.bound = 'true';
    fillBtn.addEventListener('click', fillSampleData);
  }
}

function fillSampleData(e) {
  if (e) e.preventDefault();
  const form = document.querySelector('#page-predictions form');
  if (!form) return;
  const sample = {
    previousGpa: '3.2', attendance: '92', assignmentsCompleted: '88',
    studyHours: '15', parentalEducation: '3', socioEconomicStatus: 'Medium',
    extracurricularActivities: '2', hasTutor: '0', travelTime: '30',
    internetAccess: '1', age: '17', gender: '1',
    mathScore: '85', scienceScore: '88', englishScore: '90', historyScore: '82',
  };
  Object.entries(sample).forEach(([name, value]) => {
    const el = form.querySelector(`[name="${name}"]`);
    if (el) {
      el.value = value;
      el.dispatchEvent(new Event('change', { bubbles: true }));
    }
  });
  window.showToast('Sample data filled!', 'success');
}

async function handlePredictionSubmit() {
  const runBtn       = document.getElementById('run-prediction-btn');
  const btnText      = document.getElementById('prediction-text');
  const btnSpinner   = document.getElementById('prediction-loading');
  const form         = document.querySelector('#page-predictions form');
  if (!form) return;

  // Loading state
  if (runBtn) runBtn.disabled = true;
  if (btnText) btnText.textContent = 'Analyzing…';
  if (btnSpinner) btnSpinner.classList.remove('hidden');

  try {
    const fd   = new FormData(form);
    const data = Object.fromEntries(fd.entries());

    // Validate required fields
    const required = [
      'previousGpa','attendance','assignmentsCompleted','studyHours',
      'parentalEducation','socioEconomicStatus','extracurricularActivities',
      'hasTutor','internetAccess','age','gender',
      'mathScore','scienceScore','englishScore','historyScore',
    ];
    const missing = required.filter(f => !data[f] || String(data[f]).trim() === '');
    if (missing.length) {
      window.showToast(`Please fill: ${missing.slice(0,3).join(', ')}${missing.length>3?' …':''}`, 'error');
      return;
    }

    // Type coercion
    const numericFields = [
      'previousGpa','attendance','assignmentsCompleted','studyHours',
      'parentalEducation','extracurricularActivities','hasTutor',
      'travelTime','internetAccess','age','gender',
      'mathScore','scienceScore','englishScore','historyScore',
    ];
    numericFields.forEach(f => { if (data[f] !== undefined) data[f] = parseFloat(data[f]); });

    const response = await fetch('http://127.0.0.1:8000/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      const errBody = await response.json().catch(() => ({}));
      throw new Error(errBody.detail || `HTTP ${response.status}`);
    }

    const result = await response.json();

    // Persist for other pages
    window.lastPredictionResult = result;
    window.lastPredictionInputs = data;

    updatePredictionResults(result);
    window.showToast('Prediction complete!', 'success');

    // Scroll to inline result
    const simple = document.getElementById('simple-result');
    if (simple) setTimeout(() => simple.scrollIntoView({ behavior: 'smooth', block: 'nearest' }), 80);

  } catch (err) {
    console.error('Prediction error:', err);
    window.showToast(`Prediction failed: ${err.message}`, 'error');
  } finally {
    if (runBtn) runBtn.disabled = false;
    if (btnText) btnText.textContent = 'Predict Performance';
    if (btnSpinner) btnSpinner.classList.add('hidden');
  }
}

function updatePredictionResults(data) {
  const score      = Math.round(data.score_display);
  const label      = data.predicted_category || '';
  const riskSimple = /high/i.test(label) ? 'High' : /medium/i.test(label) ? 'Medium' : 'Low';
  const highPct    = Math.round(((data.probabilities?.['High Risk'])   || 0) * 100);
  const medPct     = Math.round(((data.probabilities?.['Medium Risk']) || 0) * 100);
  const lowPct     = Math.round(((data.probabilities?.['Low Risk'])    || 0) * 100);

  // Big score
  const scoreEl = document.getElementById('predicted-score-label');
  if (scoreEl) scoreEl.textContent = `${score}%`;

  // Risk label + pill
  const riskText  = document.getElementById('risk-level-text');
  const riskDot   = document.getElementById('risk-level-dot');
  const riskWrap  = document.getElementById('risk-level-label');
  const colorMap  = {
    High:   { bg: 'risk-high',   dot: 'bg-red-500'    },
    Medium: { bg: 'risk-medium', dot: 'bg-yellow-500' },
    Low:    { bg: 'risk-low',    dot: 'bg-green-500'  },
  };
  const c = colorMap[riskSimple] || colorMap.Low;
  if (riskText)  riskText.textContent = label;
  if (riskWrap)  riskWrap.className   = `inline-flex items-center gap-2 px-4 py-2 rounded-full text-sm font-semibold mb-4 ${c.bg}`;
  if (riskDot)   riskDot.className    = `w-2.5 h-2.5 rounded-full ${c.dot}`;

  // Inline quick result
  const simpleResult = document.getElementById('simple-result');
  const simpleScore  = document.getElementById('simple-prediction-score');
  const simpleChip   = document.getElementById('simple-risk-chip');
  if (simpleResult) simpleResult.classList.remove('hidden');
  if (simpleScore)  simpleScore.textContent = `${score}%`;
  if (simpleChip) {
    simpleChip.textContent = `${riskSimple} Risk`;
    simpleChip.className = 'badge ' + (riskSimple === 'Low' ? 'badge-green' : riskSimple === 'Medium' ? 'badge-yellow' : 'badge-red');
  }

  // Probability percentages
  const setEl = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = `${val}%`; };
  setEl('high-risk-percent',   highPct);
  setEl('medium-risk-percent', medPct);
  setEl('low-risk-percent',    lowPct);

  // Insight text
  const insight = document.getElementById('performance-insight');
  if (insight) {
    insight.textContent = score >= 80
      ? 'Excellent performance! You\'re on track for great results. Keep it up!'
      : score >= 60
      ? 'Good progress. With targeted effort on weak areas, you can push higher.'
      : 'There is room to grow. Focus on attendance, study time, and key subjects.';
  }

  // Update score ring if present
  const ring = document.getElementById('score-ring-progress');
  if (ring) {
    const r = parseFloat(ring.getAttribute('r')) || 52;
    const circ = 2 * Math.PI * r;
    ring.style.strokeDasharray  = circ;
    ring.style.strokeDashoffset = circ - (circ * score / 100);
    ring.setAttribute('stroke', riskSimple === 'High' ? '#ef4444' : riskSimple === 'Medium' ? '#f59e0b' : '#10b981');
  }

  // Dashboard sync
  if (window.updateDashboardWithPrediction) window.updateDashboardWithPrediction(data);
}
window.updatePredictionResults = updatePredictionResults;
