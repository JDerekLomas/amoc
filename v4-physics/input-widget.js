// InputWidget - vanilla JS version for static HTML sites
// Activates when ?getinput is in URL or on localhost
// Edit text inline, leave comments, copy feedback as JSON
(function() {
  const params = new URLSearchParams(window.location.search);
  const host = window.location.hostname;
  const isShareMode = params.has('getinput');
  const isLocalhost = host === 'localhost' || host === '127.0.0.1';
  if (!isShareMode && !isLocalhost) return;

  let mode = 'idle'; // idle | editing | commenting | viewing
  let activeElement = null;
  let originalText = '';
  let feedbackItems = [];

  // --- Helpers ---
  function getSelector(el) {
    const path = [];
    let cur = el;
    while (cur && cur !== document.body) {
      let sel = cur.tagName.toLowerCase();
      if (cur.id) { path.unshift('#' + cur.id); break; }
      if (cur.className && typeof cur.className === 'string') {
        const cls = cur.className.split(' ').filter(c => c && !c.startsWith('__')).slice(0, 2).join('.');
        if (cls) sel += '.' + cls;
      }
      path.unshift(sel);
      cur = cur.parentElement;
    }
    return path.join(' > ');
  }

  function isTextElement(el) {
    const tags = ['P','H1','H2','H3','H4','H5','H6','SPAN','A','LI','TD','TH','LABEL','BUTTON','DIV'];
    if (!tags.includes(el.tagName)) return false;
    const text = (el.textContent || '').trim();
    const children = el.querySelectorAll('*').length;
    return text.length > 0 && text.length < 500 && children < 5;
  }

  // --- Styles ---
  const css = document.createElement('style');
  css.textContent = `
    #gi-widget{position:fixed;bottom:16px;right:16px;z-index:99999;font-family:-apple-system,system-ui,sans-serif}
    #gi-widget *{box-sizing:border-box}
    .gi-btn{display:inline-flex;align-items:center;gap:6px;border:none;border-radius:999px;padding:8px 16px;font-size:13px;font-weight:500;color:#fff;cursor:pointer;box-shadow:0 4px 12px rgba(0,0,0,.25);transition:.15s}
    .gi-btn:hover{filter:brightness(1.1);transform:translateY(-1px)}
    .gi-btn-edit{background:#d97706}
    .gi-btn-comment{background:#2563eb}
    .gi-btn-copy{background:#16a34a}
    .gi-btn-copy:disabled{background:#9ca3af;cursor:not-allowed;filter:none;transform:none}
    .gi-btn-count{width:32px;height:32px;padding:0;justify-content:center;background:#f3f4f6;color:#4b5563;font-size:11px;border:1px solid #e5e7eb}
    .gi-btn-count:hover{background:#e5e7eb}
    .gi-panel{background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:14px;box-shadow:0 8px 24px rgba(0,0,0,.15);color:#111}
    .gi-panel p{margin:0}
    .gi-cancel{background:none;border:none;color:#9ca3af;cursor:pointer;font-size:13px;padding:4px 0}
    .gi-cancel:hover{color:#6b7280}
    .gi-toast{position:absolute;bottom:52px;right:0;background:#16a34a;color:#fff;padding:6px 14px;border-radius:8px;font-size:13px;box-shadow:0 4px 12px rgba(0,0,0,.2);opacity:0;transition:opacity .2s}
    .gi-toast.show{opacity:1}
    .gi-textarea{width:100%;border:1px solid #e5e7eb;background:#f9fafb;border-radius:6px;padding:8px;font-size:13px;color:#111;resize:none;outline:none;font-family:inherit}
    .gi-textarea:focus{border-color:#2563eb}
    .gi-fb-item{background:#f9fafb;border:1px solid #f3f4f6;border-radius:8px;padding:8px;font-size:11px;margin-bottom:6px}
    .gi-fb-edit-label{color:#d97706;font-weight:600}
    .gi-fb-comment-label{color:#2563eb;font-weight:600}
    .gi-fb-old{color:#9ca3af;text-decoration:line-through}
    .gi-fb-new{color:#4b5563}
    .gi-fb-on{color:#9ca3af;font-size:10px;margin-top:2px}
    .gi-onboard{position:absolute;bottom:52px;right:0;width:260px;background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:12px;box-shadow:0 8px 24px rgba(0,0,0,.15);color:#111}
    .gi-onboard::after{content:'';position:absolute;bottom:-8px;right:32px;width:0;height:0;border-left:8px solid transparent;border-right:8px solid transparent;border-top:8px solid #fff}
    .gi-link{font-size:10px;color:#9ca3af;text-decoration:none;text-align:right;display:block;margin-top:4px}
    .gi-link:hover{color:#6b7280}
    .gi-row{display:flex;gap:8px;align-items:center}
    .gi-view-panel{width:300px}
    .gi-view-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px}
    .gi-view-header h3{font-size:13px;font-weight:600;margin:0}
    .gi-view-close{background:none;border:none;color:#9ca3af;cursor:pointer;padding:2px}
    .gi-view-close:hover{color:#6b7280}
    .gi-view-list{max-height:200px;overflow-y:auto;margin-bottom:10px}
    .gi-view-empty{text-align:center;color:#9ca3af;font-size:13px;padding:16px 0}
  `;
  document.head.appendChild(css);

  // --- Widget DOM ---
  const widget = document.createElement('div');
  widget.id = 'gi-widget';
  document.body.appendChild(widget);

  const toast = document.createElement('div');
  toast.className = 'gi-toast';
  widget.appendChild(toast);

  function showToast(msg) {
    toast.textContent = msg;
    toast.classList.add('show');
    setTimeout(() => toast.classList.remove('show'), 2000);
  }

  function render() {
    // Remove everything except toast
    Array.from(widget.children).forEach(c => { if (c !== toast) widget.removeChild(c); });

    if (mode === 'idle' && !activeElement) {
      const row = document.createElement('div');
      row.style.display = 'flex';
      row.style.flexDirection = 'column';
      row.style.alignItems = 'flex-end';
      row.style.gap = '8px';

      const btns = document.createElement('div');
      btns.className = 'gi-row';

      const editBtn = document.createElement('button');
      editBtn.className = 'gi-btn gi-btn-edit';
      editBtn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M17 3a2.85 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z"/></svg>Edit';
      editBtn.onclick = () => { mode = 'editing'; render(); };

      const commentBtn = document.createElement('button');
      commentBtn.className = 'gi-btn gi-btn-comment';
      commentBtn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>Comment';
      commentBtn.onclick = () => { mode = 'commenting'; render(); };

      btns.appendChild(editBtn);
      btns.appendChild(commentBtn);

      if (isShareMode) {
        const copyBtn = document.createElement('button');
        copyBtn.className = 'gi-btn gi-btn-copy';
        copyBtn.disabled = feedbackItems.length === 0;
        copyBtn.innerHTML = '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>Copy feedback' + (feedbackItems.length ? ' (' + feedbackItems.length + ')' : '');
        copyBtn.onclick = copyFeedback;
        btns.appendChild(copyBtn);
      } else if (feedbackItems.length > 0) {
        const countBtn = document.createElement('button');
        countBtn.className = 'gi-btn gi-btn-count';
        countBtn.textContent = feedbackItems.length;
        countBtn.onclick = () => { mode = 'viewing'; render(); };
        btns.appendChild(countBtn);
      }

      row.appendChild(btns);

      if (isShareMode) {
        const link = document.createElement('a');
        link.className = 'gi-link';
        link.href = 'https://getinput.io';
        link.target = '_blank';
        link.textContent = 'Powered by getinput';
        row.appendChild(link);
      }

      widget.appendChild(row);
    }

    else if ((mode === 'editing' || mode === 'commenting') && !activeElement) {
      const panel = document.createElement('div');
      panel.className = 'gi-panel';
      panel.innerHTML = '<p style="font-size:13px;margin-bottom:8px;color:#111">' +
        (mode === 'editing' ? 'Click any text to edit it directly' : 'Click any element to leave a comment') + '</p>';
      const cancel = document.createElement('button');
      cancel.className = 'gi-cancel';
      cancel.textContent = 'Cancel';
      cancel.onclick = () => { mode = 'idle'; document.body.style.cursor = ''; render(); };
      panel.appendChild(cancel);
      widget.appendChild(panel);
    }

    else if (mode === 'commenting' && activeElement) {
      const panel = document.createElement('div');
      panel.className = 'gi-panel';
      panel.style.width = '280px';

      const preview = document.createElement('p');
      preview.style.cssText = 'font-size:11px;color:#9ca3af;margin-bottom:8px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap';
      preview.textContent = (activeElement.textContent || '').slice(0, 50) || getSelector(activeElement);
      panel.appendChild(preview);

      const ta = document.createElement('textarea');
      ta.className = 'gi-textarea';
      ta.rows = 2;
      ta.placeholder = 'What should change?';
      ta.onkeydown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submitComment(ta.value); }
        if (e.key === 'Escape') cancelActive();
      };
      panel.appendChild(ta);

      const actions = document.createElement('div');
      actions.style.cssText = 'display:flex;justify-content:flex-end;gap:8px;margin-top:8px';
      const cancelBtn = document.createElement('button');
      cancelBtn.className = 'gi-cancel';
      cancelBtn.textContent = 'Cancel';
      cancelBtn.onclick = cancelActive;
      const saveBtn = document.createElement('button');
      saveBtn.className = 'gi-btn gi-btn-comment';
      saveBtn.style.cssText = 'padding:6px 14px;font-size:12px;box-shadow:none';
      saveBtn.textContent = 'Save';
      saveBtn.onclick = () => submitComment(ta.value);
      actions.appendChild(cancelBtn);
      actions.appendChild(saveBtn);
      panel.appendChild(actions);

      widget.appendChild(panel);
      setTimeout(() => ta.focus(), 50);
    }

    else if (mode === 'viewing') {
      const panel = document.createElement('div');
      panel.className = 'gi-panel gi-view-panel';

      const header = document.createElement('div');
      header.className = 'gi-view-header';
      header.innerHTML = '<h3>Feedback (' + feedbackItems.length + ')</h3>';
      const closeBtn = document.createElement('button');
      closeBtn.className = 'gi-view-close';
      closeBtn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 6L6 18M6 6l12 12"/></svg>';
      closeBtn.onclick = () => { mode = 'idle'; render(); };
      header.appendChild(closeBtn);
      panel.appendChild(header);

      if (feedbackItems.length === 0) {
        const empty = document.createElement('p');
        empty.className = 'gi-view-empty';
        empty.textContent = 'No feedback yet.';
        panel.appendChild(empty);
      } else {
        const list = document.createElement('div');
        list.className = 'gi-view-list';
        feedbackItems.forEach(item => {
          const div = document.createElement('div');
          div.className = 'gi-fb-item';
          if (item.type === 'text-edit') {
            div.innerHTML = '<span class="gi-fb-edit-label">Edit:</span>' +
              '<p class="gi-fb-old">' + esc(item.original.slice(0, 60)) + '</p>' +
              '<p class="gi-fb-new">' + esc(item.edited.slice(0, 60)) + '</p>';
          } else {
            div.innerHTML = '<span class="gi-fb-comment-label">Comment:</span>' +
              '<p class="gi-fb-new">' + esc(item.comment) + '</p>' +
              '<p class="gi-fb-on">on: ' + esc((item.elementText || '').slice(0, 40)) + '</p>';
          }
          list.appendChild(div);
        });
        panel.appendChild(list);

        const copyBtn = document.createElement('button');
        copyBtn.className = 'gi-btn gi-btn-comment';
        copyBtn.style.cssText = 'width:100%;justify-content:center;border-radius:8px;box-shadow:none';
        copyBtn.innerHTML = '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>Copy feedback';
        copyBtn.onclick = copyFeedback;
        panel.appendChild(copyBtn);
      }

      widget.appendChild(panel);
    }

    // Update cursor
    if (mode === 'editing') document.body.style.cursor = 'text';
    else if (mode === 'commenting') document.body.style.cursor = 'crosshair';
    else document.body.style.cursor = '';
  }

  function esc(s) {
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
  }

  function copyFeedback() {
    navigator.clipboard.writeText(JSON.stringify(feedbackItems, null, 2));
    showToast('Copied to clipboard!');
  }

  function submitComment(text) {
    if (!activeElement || !text.trim()) return;
    feedbackItems.push({
      type: 'comment',
      selector: getSelector(activeElement),
      path: window.location.pathname,
      elementText: (activeElement.textContent || '').slice(0, 100),
      comment: text.trim(),
      timestamp: new Date().toISOString()
    });
    clearActiveHighlight();
    activeElement = null;
    mode = 'idle';
    showToast('Added!');
    render();
  }

  function cancelActive() {
    clearActiveHighlight();
    activeElement = null;
    mode = 'idle';
    render();
  }

  function clearActiveHighlight() {
    if (activeElement) {
      activeElement.style.outline = '';
      activeElement.style.outlineOffset = '';
    }
  }

  // --- Click handler ---
  function handleClick(e) {
    if (mode === 'idle' || mode === 'viewing') return;
    if (widget.contains(e.target)) return;

    e.preventDefault();
    e.stopPropagation();
    const target = e.target;

    if (mode === 'editing' && isTextElement(target)) {
      activeElement = target;
      originalText = target.textContent || '';
      target.contentEditable = 'true';
      target.style.outline = '2px solid #d97706';
      target.style.outlineOffset = '2px';
      target.focus();

      const onBlur = () => {
        target.contentEditable = 'false';
        target.style.outline = '';
        target.style.outlineOffset = '';
        const newText = target.textContent || '';
        if (newText !== originalText) {
          feedbackItems.push({
            type: 'text-edit',
            selector: getSelector(target),
            path: window.location.pathname,
            original: originalText,
            edited: newText,
            timestamp: new Date().toISOString()
          });
          showToast('Added!');
        }
        activeElement = null;
        mode = 'idle';
        target.removeEventListener('blur', onBlur);
        render();
      };
      target.addEventListener('blur', onBlur);
      render();
    }
    else if (mode === 'commenting') {
      if (activeElement) clearActiveHighlight();
      activeElement = target;
      target.style.outline = '2px solid #2563eb';
      target.style.outlineOffset = '2px';
      render();
    }
  }

  document.addEventListener('click', handleClick, true);

  // --- Init ---
  render();
})();
