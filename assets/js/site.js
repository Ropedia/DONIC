/* DONIC project page — interactions. Content lives in content.js. */
(() => {
  'use strict';
  const C = window.DONIC;
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };
  const media = stem => `assets/media/${stem}`;
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const hasIO = 'IntersectionObserver' in window;
  const ease = t => 1 - Math.pow(1 - Math.min(Math.max(t, 0), 1), 3);
  const fmtTime = s => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
  const onView = (node, fn, threshold = 0.25, once = true) => {
    if (!hasIO) return fn(true);
    const io = new IntersectionObserver(es => es.forEach(e => {
      fn(e.isIntersecting);
      if (once && e.isIntersecting) io.disconnect();
    }), { threshold });
    io.observe(node);
  };

  /* ------------------------------------------------ identity & links */
  const site = C.site;
  const authors = $('#authors'), team = $('#team');
  const nameSpans = () => site.authors.map(a => `<span class="au">${a.name}${a.sup ? `<sup>${a.sup}</sup>` : ''}</span>`).join('');
  if (site.authors && site.authors.length) {
    authors.innerHTML = `<span class="au-list">${nameSpans()}</span>` +
      (site.affiliations ? `<span class="aff">${site.affiliations}</span>` : '');
    team.innerHTML = `<div class="team-names">${nameSpans()}</div>` +
      (site.affiliations ? `<p class="team-aff">${site.affiliations}</p>` : '');
  } else { authors.textContent = site.anonymousLine; team.hidden = true; }
  [$('#nav-paper'), $('#hero-paper')].forEach(a => {
    if (site.paperUrl) { a.href = site.paperUrl; a.target = '_blank'; a.rel = 'noopener'; }
    else {
      a.classList.add('soon'); a.setAttribute('aria-disabled', 'true'); a.title = 'Link coming soon';
      a.textContent = 'Paper · soon'; a.addEventListener('click', e => e.preventDefault());
    }
  });

  /* ------------------------------------------------ nav state */
  const nav = $('#nav');
  const onScroll = () => nav.classList.toggle('scrolled', scrollY > 24);
  addEventListener('scroll', onScroll, { passive: true }); onScroll();
  if (hasIO) {
    const links = new Map($$('.nav nav a').map(a => [a.getAttribute('href').slice(1), a]));
    const io = new IntersectionObserver(es => es.forEach(e => {
      if (e.isIntersecting) { links.forEach(a => a.classList.remove('active')); links.get(e.target.id)?.classList.add('active'); }
    }), { rootMargin: '-45% 0px -50% 0px' });
    links.forEach((_, id) => { const s = document.getElementById(id); if (s) io.observe(s); });
  }

  /* ------------------------------------------------ hero tape */
  const cells = $('#tape-cells');
  C.tape.forEach(([k, v], i) => cells.append(el('div', 'cell' + (i === 0 ? ' on' : ''), `<small>${k}</small><b>${v}</b>`)));
  if (!reduce) {
    let i = 0; const all = $$('.cell', cells);
    setInterval(() => { all[i].classList.remove('on'); i = (i + 1) % all.length; all[i].classList.add('on'); }, 1700);
  }
  const wall = $('.hero-wall');
  if (reduce) wall.pause();

  /* ------------------------------------------------ film */
  const film = $('#film-video'), cover = $('#film-cover');
  cover.addEventListener('click', () => {
    cover.hidden = true; film.hidden = false; film.preload = 'auto';
    film.play().catch(() => {});
    film.focus();
  });

  /* ------------------------------------------------ gallery strips */
  const lazyPlay = v => onView(v, vis => { if (vis) v.play().catch(() => {}); else v.pause(); }, 0.1, false);
  const tile = ([stem, name, src]) => {
    const f = el('figure', 'tile');
    const v = el('video'); v.muted = true; v.loop = true; v.playsInline = true; v.preload = 'none';
    v.src = media(`${stem}.mp4`); v.poster = media(`${stem}.jpg`); v.setAttribute('aria-label', `${name} rollout`);
    f.append(v, el('figcaption', null, `<span class="name">${name}</span><span class="src ${src === 'BiHOI' ? 'bihoi' : src === 'comparison' ? 'comparison' : ''}">${src}</span>`));
    if (!reduce) lazyPlay(v);
    return f;
  };
  const half = Math.ceil(C.gallery.length / 2);
  [[$('#strip-a'), C.gallery.slice(0, half)], [$('#strip-b'), C.gallery.slice(half)]].forEach(([strip, items], k) => {
    strip.style.setProperty('--dur', `${60 + k * 12}s`);
    for (let copy = 0; copy < 2; copy++) {
      const track = el('div', 'strip-track');
      if (copy) track.setAttribute('aria-hidden', 'true');
      items.forEach(it => track.append(tile(it)));
      strip.append(track);
    }
  });

  /* ------------------------------------------------ dataset */
  const ds = $('#dataset-stats');
  C.dataset.stats.forEach(([v, k, unit]) => ds.append(el('div', null, `<dt>${k}</dt><dd>${v}${unit ? `<small>${unit}</small>` : ''}</dd>`)));
  const bars = (host, rows, max, isOurs, digits = 2) => {
    host.replaceChildren(...rows.map(([name, v]) => {
      const b = el('div', 'bar' + (isOurs(name) ? ' ours' : ''),
        `<span>${name}</span><span class="track"><span class="fill"></span></span><span class="v">${v.toFixed(digits)}%</span>`);
      b.dataset.w = `${(v / max) * 100}%`;
      return b;
    }));
    return () => $$('.fill', host).forEach((f, i) => { f.style.width = host.children[i].dataset.w; });
  };
  const contact = $('#contact-bars');
  const growContact = bars(contact, C.dataset.contact, 70, n => n.startsWith('BiHOI'));
  onView(contact, v => v && growContact(), 0.3);

  /* ------------------------------------------------ synced case viewers */
  function viewer(root, spec) {
    root.style.setProperty('--ar', spec.aspect);
    const head = el('div', 'v-head', `<span class="v-label">Choose a case</span>`);
    const tabs = el('div', 'tabs'); tabs.setAttribute('role', 'group');
    head.append(tabs);
    const grid = el('div', 'v-grid');
    const vids = spec.columns.map(([key, label, ours]) => {
      const cell = el('figure', 'v-cell' + (ours ? ' ours' : ''));
      const cap = el('figcaption', null, `<span>${label}</span>`);
      const full = el('button', 'full', '⛶'); full.type = 'button'; full.setAttribute('aria-label', `Fullscreen ${label}`);
      cap.append(full);
      const frame = el('div', 'v-frame');
      const v = el('video'); v.muted = true; v.playsInline = true; v.preload = 'metadata'; v.setAttribute('aria-label', label);
      frame.append(v); cell.append(cap, frame); grid.append(cell);
      full.addEventListener('click', () => (v.requestFullscreen ? v.requestFullscreen() : v.webkitEnterFullscreen?.()));
      return { key, v };
    });
    const tr = el('div', 'transport');
    const play = el('button', 't-play', '▶'); play.type = 'button'; play.setAttribute('aria-label', 'Play');
    const time = el('span', 't-time', '0:00 / 0:00');
    const seek = el('input', 't-seek'); seek.type = 'range'; seek.min = 0; seek.max = 1000; seek.value = 0; seek.setAttribute('aria-label', 'Seek');
    const speeds = el('div', 't-speed');
    [0.5, 1, 2].forEach(r => { const b = el('button', null, `${r}×`); b.type = 'button'; b.dataset.r = r; b.setAttribute('aria-pressed', r === 1); speeds.append(b); });
    tr.append(play, time, seek, speeds);
    root.append(head, grid, tr);

    let idx = -1, t = 0, dur = 0, rate = 1, playing = false, last = 0, userPaused = false, inView = false, raf = 0;
    const END = 0.04;
    const ui = () => {
      time.textContent = `${fmtTime(t)} / ${fmtTime(dur)}`;
      seek.value = dur ? Math.round((t / dur) * 1000) : 0;
      play.textContent = playing ? '❚❚' : '▶'; play.setAttribute('aria-label', playing ? 'Pause' : 'Play');
    };
    const sync = (hard) => vids.forEach(({ v }) => {
      if (!v.duration) return;
      const end = v.duration - END, target = Math.min(t, end);
      if (playing && t < end) {
        v.playbackRate = rate;
        if (v.paused) v.play().catch(() => {});
        if (hard || Math.abs(v.currentTime - target) > 0.3) v.currentTime = target;
      } else {
        if (!v.paused) v.pause();
        if (hard || Math.abs(v.currentTime - target) > 0.08) v.currentTime = target;
      }
    });
    const tick = now => {
      if (!playing) return;
      t += ((now - last) / 1000) * rate; last = now;
      if (dur && t >= dur) { t = 0; sync(true); }
      else sync(false);
      ui(); raf = requestAnimationFrame(tick);
    };
    const start = () => { if (playing || !dur) return; playing = true; last = performance.now(); sync(true); ui(); raf = requestAnimationFrame(tick); };
    const stop = () => { playing = false; cancelAnimationFrame(raf); vids.forEach(({ v }) => v.pause()); ui(); };
    const load = i => {
      if (i === idx) return;
      idx = i; stop(); t = 0; dur = 0; ui();
      $$('button', tabs).forEach((b, k) => b.setAttribute('aria-pressed', k === i));
      const stem = spec.cases[i][0];
      const ready = vids.map(({ key, v }) => new Promise(res => {
        v.addEventListener('loadedmetadata', res, { once: true }); v.addEventListener('error', res, { once: true });
        v.poster = media(`${stem}-${key}.jpg`); v.src = media(`${stem}-${key}.mp4`); v.load();
      }));
      Promise.all(ready).then(() => {
        if (i !== idx) return;
        dur = Math.max(...vids.map(({ v }) => v.duration || 0)); ui();
        if (inView && !userPaused && !reduce) start();
      });
    };
    spec.cases.forEach(([, name], i) => {
      const b = el('button', null, name); b.type = 'button'; b.setAttribute('aria-pressed', 'false');
      b.addEventListener('click', () => { userPaused = false; load(i); });
      tabs.append(b);
    });
    play.addEventListener('click', () => { if (playing) { userPaused = true; stop(); } else { userPaused = false; start(); } });
    seek.addEventListener('input', () => { t = (seek.value / 1000) * dur; sync(true); ui(); });
    speeds.addEventListener('click', e => {
      const b = e.target.closest('button'); if (!b) return;
      rate = +b.dataset.r; $$('button', speeds).forEach(x => x.setAttribute('aria-pressed', x === b));
      vids.forEach(({ v }) => { v.playbackRate = rate; });
    });
    onView(root, vis => { inView = vis; if (vis && !userPaused && !reduce) start(); else if (!vis) stop(); }, 0.35, false);
    load(0);
  }
  $$('[data-viewer]').forEach(root => viewer(root, C.viewers[root.dataset.viewer]));

  /* ------------------------------------------------ pipeline dialog */
  const dlg = $('#pipeline-dialog');
  $('#pipeline-open').addEventListener('click', () => (dlg.showModal ? dlg.showModal() : window.open('assets/media/pipeline.png')));
  $('#pipeline-close').addEventListener('click', () => dlg.close());
  dlg.addEventListener('click', e => { if (e.target === dlg) dlg.close(); });

  /* ------------------------------------------------ results */
  const R = C.results;
  $('#n-traj').textContent = R.trajectory.clips; $('#n-obj').textContent = R.object.clips;
  const digitsHost = $('#big-digits'), srHost = $('#sr-bars'), tbody = $('#results-table tbody');
  const PREC = [2, 4, 2, 4, 4, 2, 2], LOWER = [true, true, true, true, true, false, false];
  const OURS = n => n.startsWith('DONIC');
  const drawDigits = v => {
    const s = v.toFixed(2);
    digitsHost.replaceChildren(...[...s].map(ch => el('span', 'digit' + (ch === '.' ? ' sep' : ''), ch)), el('span', 'digit unit', '%'));
    digitsHost.setAttribute('aria-label', `${s} percent`);
  };
  let anim = 0;
  const countTo = target => {
    cancelAnimationFrame(anim);
    if (reduce) return drawDigits(target);
    const t0 = performance.now();
    const step = now => { const k = ease((now - t0) / 900); drawDigits(target * k); if (k < 1) anim = requestAnimationFrame(step); };
    anim = requestAnimationFrame(step);
  };
  let growSR = () => {};
  const showSplit = (key, animate) => {
    const sp = R[key], rows = sp.rows;
    const ours = rows.find(r => r[0] === 'DONIC');
    const base = rows.filter(r => /-bi$/.test(r[0])).sort((a, b) => b[6] - a[6])[0];
    $('#big-label').textContent = `DONIC success rate · ${sp.label} · ${sp.clips} clips`;
    $('#big-delta').innerHTML = `<b>+${(ours[6] - base[6]).toFixed(2)} pts</b> over ${base[0]}, the stronger bimanual baseline (${base[6].toFixed(2)}%).`;
    if (animate) countTo(ours[6]); else drawDigits(ours[6]);
    growSR = bars(srHost, rows.map(r => [r[0], r[6]]), 100, OURS);
    if (animate || !hasIO) requestAnimationFrame(() => requestAnimationFrame(growSR));
    const best = PREC.map((_, c) => {
      const col = rows.map(r => r[c + 1]);
      return LOWER[c] ? Math.min(...col) : Math.max(...col);
    });
    tbody.replaceChildren(...rows.map((r, i) => {
      const tr = el('tr', (OURS(r[0]) ? 'ours' : '') + (i === 2 ? ' sep' : ''));
      tr.append(el('td', null, r[0]));
      r.slice(1).forEach((v, c) => tr.append(el('td', v === best[c] ? 'best' : '', v.toFixed(PREC[c]))));
      return tr;
    }));
  };
  $$('.split-tabs button').forEach(b => b.addEventListener('click', () => {
    $$('.split-tabs button').forEach(x => x.setAttribute('aria-pressed', x === b));
    showSplit(b.dataset.split, true);
  }));
  showSplit('trajectory', false);
  onView($('.scoreboard'), v => { if (v) { countTo(R.trajectory.rows.find(r => r[0] === 'DONIC')[6]); growSR(); } }, 0.4);

  /* ------------------------------------------------ bibtex */
  $('#bibtex').textContent = site.bibtex;
  $('#copy-bib').addEventListener('click', async e => {
    const b = e.currentTarget;
    try { await navigator.clipboard.writeText(site.bibtex); b.textContent = 'Copied'; }
    catch { b.textContent = 'Select and copy'; }
    setTimeout(() => { b.textContent = 'Copy'; }, 1800);
  });

  /* ------------------------------------------------ reveal on scroll */
  if (hasIO && !reduce) {
    const targets = $$('.section h2, .section .lede, .film, .note, .chart, .stats, .viewer, .pipeline, .cards article, .scoreboard, .table-scroll, .bib');
    const io = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); } }), { threshold: 0.12 });
    targets.forEach(t => { t.classList.add('reveal'); io.observe(t); });
  }
})();
