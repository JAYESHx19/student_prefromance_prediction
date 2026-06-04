/* ============================================
   recommendations.js — Recommendation engine
   ============================================ */

window.generateRecommendations = function (result, inputs, opts = {}) {
  const listId     = opts.listElId    || 'recs-list';
  const planId     = opts.planElId    || 'recs-plan';
  const planDaysId = opts.planDaysElId || 'recs-plan-days';
  const riskChipId = opts.riskChipId  || 'recs-risk-chip';
  const emptyNoteId = opts.emptyNoteId || 'recs-empty-note';

  const listEl     = document.getElementById(listId);
  const planEl     = document.getElementById(planId);
  const planDaysEl = document.getElementById(planDaysId);
  const riskChip   = document.getElementById(riskChipId);
  const emptyNote  = document.getElementById(emptyNoteId);
  if (!listEl) return;

  listEl.innerHTML = '';
  if (planDaysEl) planDaysEl.innerHTML = '';

  if (!result) {
    if (planEl) planEl.classList.add('hidden');
    if (riskChip) {
      riskChip.textContent = 'No prediction yet';
      riskChip.className = 'badge badge-gray';
    }
    return;
  }

  const label    = String(result.predicted_category || '').toLowerCase();
  const score    = Number(result.score_display || 0);
  const isHigh   = label.includes('high');
  const isMedium = label.includes('medium');
  const isLow    = !isHigh && !isMedium;

  const attendance   = Number(inputs?.attendance || 0);
  const assignments  = Number(inputs?.assignmentsCompleted || 0);
  const studyHours   = Number(inputs?.studyHours || 0);
  const math         = Number(inputs?.mathScore || 0);
  const science      = Number(inputs?.scienceScore || 0);
  const english      = Number(inputs?.englishScore || 0);
  const history      = Number(inputs?.historyScore || 0);
  const travel       = Number(inputs?.travelTime || 20);
  const hasTutor     = Number(inputs?.hasTutor || 0) === 1;
  const internet     = Number(inputs?.internetAccess || 0) === 1;

  const weakSubjects = [];
  if (math    && math    < 70) weakSubjects.push('Math');
  if (science && science < 70) weakSubjects.push('Science');
  if (english && english < 70) weakSubjects.push('English');
  if (history && history < 70) weakSubjects.push('History');

  // Risk chip
  if (riskChip) {
    const riskText = isHigh ? 'High' : isMedium ? 'Medium' : 'Low';
    riskChip.textContent = `${riskText} Risk • ${Math.round(score)}%`;
    riskChip.className = 'badge ' + (isHigh ? 'badge-red' : isMedium ? 'badge-yellow' : 'badge-green');
  }

  // Card factory
  const makeCard = (emoji, title, desc, tags) => {
    const tagHtml = (tags || []).map(t =>
      `<span class="badge badge-gray" style="font-size:.7rem">${t}</span>`
    ).join('');
    return `
      <div class="rec-card fade-in">
        <div style="display:flex;align-items:flex-start;gap:12px">
          <div style="font-size:1.5rem;flex-shrink:0;line-height:1">${emoji}</div>
          <div style="flex:1">
            <h6 style="font-weight:600;font-size:.9rem;color:var(--text-primary);margin-bottom:4px">${title}</h6>
            <p style="font-size:.82rem;color:var(--text-secondary);line-height:1.5">${desc}</p>
            ${tagHtml ? `<div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:10px">${tagHtml}</div>` : ''}
          </div>
        </div>
      </div>`;
  };

  const cards = [];

  if (isHigh) {
    cards.push(makeCard('🧭', 'Stabilize Your Routine',
      'Set a fixed daily study window and start with 20–30 min focus blocks to build consistency.',
      ['Consistency', 'Focus Blocks']));
    cards.push(makeCard('🤝', 'Get an Accountability Partner',
      'Weekly check-ins with a buddy or mentor — review goals and identify blockers.',
      ['Buddy System', 'Weekly Review']));
  } else if (isMedium) {
    cards.push(makeCard('⏱️', 'Try Pomodoro + Active Recall',
      'Use 25/5 Pomodoro sessions with active recall and spaced repetition for better retention.',
      ['Pomodoro', 'Spaced Repetition']));
    cards.push(makeCard('🎯', 'Focus on Weak Spots',
      'Allocate two extra study sessions to your weakest subjects this week.',
      weakSubjects.length ? weakSubjects : ['Weak Areas']));
  } else {
    cards.push(makeCard('🏆', 'Challenge Yourself',
      'Add weekly mock tests and challenge problems to push your performance even higher.',
      ['Mock Tests', 'Challenge Problems']));
    cards.push(makeCard('📚', 'Teach to Master',
      'Teach a topic or make summary notes — teaching locks in learning permanently.',
      ['Peer Teaching', 'Notes']));
  }

  if (attendance < 85)   cards.push(makeCard('📅', 'Improve Attendance',
    'Prepare mornings better, reduce avoidable absences, and plan time buffers.',
    ['Morning Prep', 'Reduce Absences']));
  if (assignments < 80)  cards.push(makeCard('✅', 'Track Assignments',
    'Use a checklist with deadlines and target submitting 48 hours early.',
    ['Checklist', 'Early Submit']));
  if (studyHours < 10)   cards.push(makeCard('➕', 'Add 90 More Minutes/Week',
    'Add three 30-minute focused sessions to your weekly schedule.',
    ['3×30 Sessions']));
  if (weakSubjects.length) cards.push(makeCard('📈', `Boost ${weakSubjects.join(', ')}`,
    'Use topic-wise practice and spaced repetition to reach 80%+ in weak subjects.',
    weakSubjects));
  if (!internet)         cards.push(makeCard('⬇️', 'Build Offline Resources',
    'Download PDFs and videos; keep a physical notebook backup for offline study.',
    ['Offline Study']));
  if (!hasTutor && (isHigh || isMedium)) cards.push(makeCard('🧑‍🏫', 'Consider a Tutor',
    'A short weekly tutoring session can unblock progress quickly.',
    ['Guidance', '1-on-1']));
  if (travel > 45)       cards.push(makeCard('🚇', 'Use Commute Time',
    'Use commute for flashcards or audio lessons; prep materials the night before.',
    ['Anki', 'Audio Lessons']));

  listEl.innerHTML = cards.join('');

  // 7-Day Plan
  if (planEl && planDaysEl) {
    const days = [
      { d: 'Day 1', t: 'Setup & Plan',       i: '🗂️' },
      { d: 'Day 2', t: 'Focus Session A',     i: '🎧' },
      { d: 'Day 3', t: 'Weak Subject Drill',  i: '🎯' },
      { d: 'Day 4', t: 'Recall + Quiz',       i: '🧠' },
      { d: 'Day 5', t: 'Focus Session B',     i: '⏱️' },
      { d: 'Day 6', t: isLow ? 'Mock Test (Advanced)' : 'Mock Test (Core)', i: '📄' },
      { d: 'Day 7', t: 'Review & Reset',      i: '🔁' },
    ];
    planDaysEl.innerHTML = days.map(x => `
      <div class="card" style="padding:14px">
        <p style="font-size:.75rem;color:var(--text-muted);margin-bottom:4px">${x.d}</p>
        <div style="display:flex;align-items:center;gap:8px">
          <span style="font-size:1.2rem">${x.i}</span>
          <span style="font-size:.85rem;font-weight:600;color:var(--text-primary)">${x.t}</span>
        </div>
      </div>`).join('');
    planEl.classList.remove('hidden');
    if (emptyNote) emptyNote.classList.add('hidden');
  }

  if (window.lucide) lucide.createIcons();
};

/* Helper also used by reports.js */
window.generateRecommendationsList = function (result, inputs) {
  const label    = String(result?.predicted_category || '').toLowerCase();
  const isHigh   = label.includes('high');
  const isMedium = label.includes('medium');
  const isLow    = !isHigh && !isMedium;
  const attendance  = Number(inputs?.attendance || 0);
  const assignments = Number(inputs?.assignmentsCompleted || 0);
  const studyHours  = Number(inputs?.studyHours || 0);
  const math        = Number(inputs?.mathScore || 0);
  const science     = Number(inputs?.scienceScore || 0);
  const english     = Number(inputs?.englishScore || 0);
  const history     = Number(inputs?.historyScore || 0);
  const hasTutor    = Number(inputs?.hasTutor || 0) === 1;
  const internet    = Number(inputs?.internetAccess || 0) === 1;

  const weakSubjects = [];
  if (math    && math    < 70) weakSubjects.push('Math');
  if (science && science < 70) weakSubjects.push('Science');
  if (english && english < 70) weakSubjects.push('English');
  if (history && history < 70) weakSubjects.push('History');

  const recs = [];
  if (isHigh) {
    recs.push({ emoji: '🧭', title: 'Stabilize Routine',       desc: 'Set fixed daily study windows with 20–30 min focus blocks.' });
    recs.push({ emoji: '🤝', title: 'Accountability Partner',  desc: 'Weekly check-ins with a buddy or mentor to review goals.' });
  } else if (isMedium) {
    recs.push({ emoji: '⏱️', title: 'Pomodoro + Recall',       desc: 'Use 25/5 Pomodoro with active recall and spaced repetition.' });
    recs.push({ emoji: '🎯', title: 'Fix Weak Areas',          desc: 'Allocate two extra sessions to weakest subjects this week.' });
  } else {
    recs.push({ emoji: '🏆', title: 'Stretch With Mocks',      desc: 'Add weekly mock tests and challenge problems to push higher.' });
    recs.push({ emoji: '📚', title: 'Teach to Master',         desc: 'Teaching a topic or making notes locks in learning permanently.' });
  }
  if (attendance  < 85)   recs.push({ emoji: '📅', title: 'Improve Attendance',   desc: 'Prepare mornings, reduce absences, plan time buffers.' });
  if (assignments < 80)   recs.push({ emoji: '✅', title: 'Track Assignments',    desc: 'Use a checklist with deadlines; submit 48 hours early.' });
  if (studyHours  < 10)   recs.push({ emoji: '➕', title: 'Add Study Time',       desc: 'Add three 30-minute focused sessions per week.' });
  if (weakSubjects.length) recs.push({ emoji: '📈', title: `Boost ${weakSubjects.join(', ')}`, desc: 'Topic-wise practice and spaced repetition to reach 80%+.' });
  if (!internet)           recs.push({ emoji: '⬇️', title: 'Offline Resources',   desc: 'Download PDFs/videos; keep a physical notebook backup.' });
  if (!hasTutor && (isHigh || isMedium)) recs.push({ emoji: '🧑‍🏫', title: 'Consider a Tutor', desc: 'A weekly tutoring session can unblock progress fast.' });
  return recs;
};
