/* ============================================
   subscription.js — Subscription, Payment, Receipts
   ============================================ */

window.initSubscriptionPage = function () {
  const statusText = document.getElementById('sub-status-text');
  const statusDot  = document.getElementById('sub-status-dot');
  const status     = window.__subscriptionStatus === 'premium' ? 'Premium' : 'Free';

  if (statusText) statusText.textContent = `Status: ${status}`;
  if (statusDot) {
    statusDot.classList.remove('bg-yellow-400', 'bg-emerald-400');
    statusDot.classList.add(status === 'Premium' ? 'bg-emerald-400' : 'bg-yellow-400');
  }

  bindOnce('sub-upgrade-btn', () => window.navigateToPage('page-payment'));
  bindOnce('sub-receipts-btn', async () => {
    injectReceiptsModal();
    await loadReceipts();
    document.getElementById('receipts-modal')?.classList.remove('hidden');
  });
};

window.initPaymentPage = function () {
  const modeSelect = document.getElementById('payment-mode');
  const cardInputs = document.getElementById('card-inputs');
  const payBtn     = document.getElementById('pay-btn');
  const payStatus  = document.getElementById('pay-status');

  modeSelect?.addEventListener('change', () => {
    if (cardInputs) cardInputs.style.display = modeSelect.value === 'Card' ? 'grid' : 'none';
  });

  bindOnce('pay-btn', async () => {
    if (!payBtn) return;
    payBtn.disabled = true;
    if (payStatus) payStatus.textContent = 'Processing payment…';
    await new Promise(r => setTimeout(r, 1800));

    const user = window.firebaseAuth?.currentUser;
    if (!user) { if (payStatus) payStatus.textContent = 'Please log in first.'; payBtn.disabled = false; return; }

    try {
      const { setDoc, doc, addDoc, collection, serverTimestamp } = window.firebaseDbFunctions;
      await setDoc(doc(window.firebaseDb, 'users', user.uid),
        { subscription_status: 'premium', upgradedAt: serverTimestamp() },
        { merge: true }
      );
      window.__subscriptionStatus = 'premium';

      const receiptId = 'RCP-' + Date.now();
      await addDoc(collection(window.firebaseDb, `users/${user.uid}/receipts`), {
        id: receiptId, plan: 'premium', amount: 499, currency: 'INR',
        createdAt: serverTimestamp(), createdAtLocal: new Date().toISOString(),
      });

      window.navigateToPage('page-payment-success');
    } catch (e) {
      if (payStatus) payStatus.textContent = 'Payment failed: ' + e.message;
      payBtn.disabled = false;
    }
  });
};

window.initPaymentSuccessPage = function () {
  const successStatus = document.getElementById('success-status');
  const dlBtn         = document.getElementById('download-receipt-btn');
  const dashBtn       = document.getElementById('go-dashboard-btn');

  if (successStatus) successStatus.textContent = 'Your premium subscription is now active!';
  if (dlBtn) dlBtn.classList.remove('hidden');

  dlBtn?.addEventListener('click', async () => {
    try {
      const user = window.firebaseAuth?.currentUser;
      const { jsPDF } = window.jspdf || {};
      if (!jsPDF) { window.showToast('PDF library not loaded', 'error'); return; }
      const pdf = new jsPDF();
      pdf.setFontSize(16); pdf.text('Payment Receipt — EduPredict', 20, 20);
      pdf.setFontSize(12);
      pdf.text(`User: ${user?.email || ''}`,   20, 36);
      pdf.text('Plan: Premium (₹499/year)',      20, 46);
      pdf.text(`Date: ${new Date().toLocaleString()}`, 20, 56);
      pdf.save('edupredict-receipt.pdf');
    } catch (e) { window.showToast('Failed to download receipt', 'error'); }
  });

  dashBtn?.addEventListener('click', () => window.navigateToPage('page-dashboard'));
};

/* ---------- Receipts Modal ---------- */
function injectReceiptsModal() {
  if (document.getElementById('receipts-modal')) return;
  const modal = document.createElement('div');
  modal.id = 'receipts-modal';
  modal.className = 'modal-backdrop hidden';
  modal.innerHTML = `
    <div class="modal-box" style="max-width:560px">
      <div style="display:flex;justify-content:space-between;align-items:center;padding:20px 24px;border-bottom:1px solid var(--border)">
        <h2 style="font-weight:700;font-size:1.1rem;color:var(--text-primary)">Payment Receipts</h2>
        <button id="close-receipts-modal" class="btn btn-icon btn-ghost" aria-label="Close">
          <i data-lucide="x" class="w-4 h-4"></i>
        </button>
      </div>
      <div id="receipts-list" style="padding:20px;max-height:380px;overflow:auto" class="space-y-3">
        <p style="color:var(--text-muted);font-size:.875rem">Loading receipts…</p>
      </div>
    </div>`;
  document.body.appendChild(modal);
  document.getElementById('close-receipts-modal')?.addEventListener('click', () =>
    modal.classList.add('hidden')
  );
  modal.addEventListener('click', e => { if (e.target === modal) modal.classList.add('hidden'); });
  if (window.lucide) lucide.createIcons();
}

async function loadReceipts() {
  const listEl = document.getElementById('receipts-list');
  if (!listEl) return;
  if (!window.firebaseAuth || !window.firebaseDb || !window.firebaseDbFunctions) {
    listEl.innerHTML = '<p style="color:#ef4444;font-size:.875rem">Firebase not initialized.</p>';
    return;
  }
  const user = window.firebaseAuth.currentUser;
  if (!user) { listEl.innerHTML = '<p style="color:#ef4444;font-size:.875rem">Please log in.</p>'; return; }

  listEl.innerHTML = '<p style="color:var(--text-muted);font-size:.875rem">Loading…</p>';
  try {
    const { collection, query, orderBy, getDocs } = window.firebaseDbFunctions;
    const snap = await getDocs(query(
      collection(window.firebaseDb, `users/${user.uid}/receipts`),
      orderBy('createdAt', 'desc')
    ));
    if (snap.empty) { listEl.innerHTML = '<p style="color:var(--text-muted);font-size:.875rem">No receipts yet.</p>'; return; }
    listEl.innerHTML = '';
    snap.forEach(docSnap => {
      const d    = docSnap.data();
      const date = d.createdAtLocal ? new Date(d.createdAtLocal).toLocaleString() : '—';
      const row  = document.createElement('div');
      row.style.cssText = 'display:flex;justify-content:space-between;align-items:center;padding:12px 14px;background:var(--bg-subtle);border-radius:var(--radius-md)';
      row.innerHTML = `
        <div>
          <p style="font-weight:600;font-size:.85rem;color:var(--text-primary)">Receipt ${d.id || docSnap.id}</p>
          <p style="font-size:.78rem;color:var(--text-secondary)">Plan: ${d.plan || 'premium'} · ₹${d.amount||0} ${d.currency||'INR'}</p>
          <p style="font-size:.75rem;color:var(--text-muted)">${date}</p>
        </div>
        <button class="btn btn-sm btn-secondary download-r" 
          data-id="${d.id||docSnap.id}" data-amount="${d.amount||0}" data-currency="${d.currency||'INR'}" 
          data-plan="${d.plan||'premium'}" data-date="${date}">
          Download
        </button>`;
      listEl.appendChild(row);
    });
    listEl.querySelectorAll('.download-r').forEach(btn =>
      btn.addEventListener('click', () => downloadReceiptPDF(btn.dataset))
    );
  } catch (e) {
    listEl.innerHTML = `<p style="color:#ef4444;font-size:.875rem">Failed to load: ${e.message}</p>`;
  }
}

async function downloadReceiptPDF(d) {
  try {
    const { jsPDF } = window.jspdf || {};
    if (!jsPDF) { window.showToast('PDF library not loaded', 'error'); return; }
    const user = window.firebaseAuth?.currentUser;
    const pdf  = new jsPDF();
    pdf.setFontSize(16); pdf.text('Payment Receipt — EduPredict', 20, 20);
    pdf.setFontSize(12);
    pdf.text(`Receipt ID: ${d.id}`,  20, 36);
    pdf.text(`User: ${user?.email||''}`, 20, 46);
    pdf.text(`Plan: ${d.plan}`,       20, 56);
    pdf.text(`Amount: ${d.currency} ${d.amount}`, 20, 66);
    pdf.text(`Date: ${d.date}`,       20, 76);
    pdf.save(`receipt_${d.id}.pdf`);
  } catch (e) { window.showToast('Download failed', 'error'); }
}

function bindOnce(id, fn) {
  const el = document.getElementById(id);
  if (el && el.dataset.bound !== 'true') { el.dataset.bound = 'true'; el.addEventListener('click', fn); }
}
