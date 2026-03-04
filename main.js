document.addEventListener('DOMContentLoaded', () => {
  const toggle = document.getElementById('search-toggle');
  const area = document.getElementById('search-area');
  const input = document.getElementById('search-input');
  const btn = document.getElementById('search-btn');
  const result = document.getElementById('result');

  toggle.addEventListener('click', () => {
    const hidden = area.getAttribute('aria-hidden') === 'true';
    area.setAttribute('aria-hidden', hidden ? 'false' : 'true');
    if (hidden) input.focus();
  });

  btn.addEventListener('click', async () => {
    const q = input.value.trim();
    if (!q) {
      result.textContent = 'Please enter a URL or query.';
      return;
    }

    result.textContent = 'Checking...';
    try {
      const res = await fetch('/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ q })
      });
      const data = await res.json();

      if (data.error) {
        result.textContent = 'Error: ' + data.error;
        return;
      }

      if (data.type === 'url') {
        result.innerHTML = `<strong>${escapeHtml(data.query)}</strong> → <em>${data.label}</em>`;
      } else if (data.type === 'lookup') {
        if (data.results && data.results.length) {
          result.innerHTML = data.results.map(
            r => `<div>${escapeHtml(r.url)} — <strong>${r.label}</strong></div>`
          ).join('');
        } else {
          result.textContent = 'No matches found.';
        }
      }
    } catch (e) {
      result.textContent = 'Request failed: ' + e.message;
    }
  });

  function escapeHtml(str) {
    return str.replace(/[&<>\"']/g, (m) => ({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',\"'\":\"&#39;\"})[m]);
  }
});
