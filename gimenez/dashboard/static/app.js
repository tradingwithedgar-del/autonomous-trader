/* Gimenez dashboard. Plain JS + TradingView Lightweight Charts (v5). Read-only. */
(function () {
  const LW = window.LightweightCharts;
  const $main = document.getElementById('main');
  let charts = [];
  let timer = null;

  const esc = (s) => String(s == null ? '' : s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const num = (x, d = 2) => (x == null || !isFinite(x) ? '-' : Number(x).toFixed(d));
  const R = (x) => (x == null || !isFinite(x) ? '-' : `<span class="${x >= 0 ? 'pos' : 'neg'}">${x >= 0 ? '+' : ''}${Number(x).toFixed(2)}R</span>`);
  const pct = (x, d = 1) => (x == null || !isFinite(x) ? '-' : `${(x * 100).toFixed(d)}%`);
  const money = (x) => (x == null ? '-' : Number(x).toLocaleString(undefined, { maximumFractionDigits: 2, minimumFractionDigits: 2 }));
  const ts = (iso) => Math.floor(Date.parse(iso) / 1000);
  const when = (iso) => (iso ? new Date(iso).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }) : '-');
  const get = async (u) => { const r = await fetch(u, { credentials: 'same-origin' }); if (!r.ok) throw new Error(`${u}: ${r.status}`); return u.includes('/api/why') ? r.text() : r.json(); };
  const stagePill = (s) => `<span class="pill st-${esc(s)}">${esc(s)}</span>`;

  function clearCharts() { charts.forEach((c) => c.remove()); charts = []; }
  function baseChart(el, extra) {
    const c = LW.createChart(el, Object.assign({
      autoSize: true,
      layout: { background: { type: 'solid', color: css('--surface') }, textColor: css('--ink-2'), fontSize: 11,
        fontFamily: 'system-ui, -apple-system, Segoe UI, sans-serif', panes: { separatorColor: css('--grid') } },
      grid: { vertLines: { color: css('--grid') }, horzLines: { color: css('--grid') } },
      rightPriceScale: { borderColor: css('--axis') },
      timeScale: { borderColor: css('--axis'), timeVisible: true, secondsVisible: false },
      crosshair: { mode: 0 },
    }, extra || {}));
    charts.push(c);
    return c;
  }

  const VERDICT_COLOR = { 'no edge yet': '--v-none', promising: '--v-promising', proven: '--v-proven', failing: '--v-failing' };

  // ---------------------------------------------------------------- overview
  async function overview() {
    const [o, eq] = await Promise.all([get('/api/overview'), get('/api/equity')]);
    const sc = o.scorecard;
    document.getElementById('mode').textContent = o.mode === 'live' ? 'LIVE' : 'demo';
    const tiles = [
      ['Expectancy', R(sc.expectancy_r), `per real trade (${sc.trades})`],
      ['Last 30 trades', R(sc.rolling30_r), 'rolling expectancy'],
      ['Profit factor', num(sc.profit_factor), 'gross win / gross loss'],
      ['Win rate', pct(sc.win_rate, 0), ''],
      ['Max drawdown', pct(sc.max_drawdown_pct), `now ${pct(sc.drawdown_pct)} (halt at ${pct(o.limits.halt, 0)})`],
      ['Live vs backtest', R(sc.live_vs_backtest_r), 'per trade; negative = overfit'],
      ['95% CI', sc.ci95 ? `${num(sc.ci95[0])} / ${num(sc.ci95[1])}` : '-', 'expectancy range (R)'],
      ['Equity', money(sc.equity), `peak ${money(sc.peak)}`],
      ['Ideas tested', (o.ideas_tested || 0).toLocaleString(), 'every one counts against luck'],
      ['Strategies', Object.entries(o.stages).map(([k, v]) => `${v} ${k}`).join(', ') || 'none yet', ''],
    ];
    const vcol = css(VERDICT_COLOR[sc.verdict] || '--v-none');
    $main.innerHTML = `
      ${sc.halted ? `<div class="card" style="border-left:6px solid var(--down)"><b>HALTED:</b> ${esc(sc.halted)}<br>Run <code>gimenez resume</code> on the server after reviewing.</div>` : ''}
      ${o.stop_file ? `<div class="card" style="border-left:6px solid var(--warn)">Stopped by you (<code>gimenez stop</code>): no new real trades.</div>` : ''}
      <div class="card verdict" style="border-left-color:${vcol}">
        <h3>Honest verdict</h3>
        <div class="label" style="color:${vcol}">${esc(sc.verdict)}</div>
        <ul>${sc.evidence.map((e) => `<li>${esc(e)}</li>`).join('')}</ul>
        <p class="sub" style="margin:8px 0 0">${esc(sc.termination)}</p>
      </div>
      <div class="tiles">${tiles.map(([k, v, d]) => `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div><div class="d">${d}</div></div>`).join('')}</div>
      <div class="card" style="margin-top:12px"><h2>Equity and drawdown</h2><div class="sub">Account equity (line) and drawdown from peak (bars). Hard limits: ${pct(o.limits.max_risk_per_trade, 0)} per trade, ${pct(o.limits.max_open_risk, 0)} open, ${pct(o.limits.daily_stop, 0)} daily stop, ${pct(o.limits.halt, 0)} halt.</div><div id="eq" class="chart"></div></div>
      <div class="card"><h2>Open real trades</h2>${tradeTable(o.open, true)}</div>`;
    setAlive(o.last_seen);
    if (eq.length) {
      const c = baseChart(document.getElementById('eq'));
      const s1 = c.addSeries(LW.LineSeries, { color: css('--accent'), lineWidth: 2, title: 'equity', priceLineVisible: false });
      s1.setData(dedupe(eq.map((p) => ({ time: ts(p.ts), value: p.equity }))));
      const s2 = c.addSeries(LW.LineSeries, { color: css('--muted'), lineWidth: 1, lineStyle: 2, title: 'balance', priceLineVisible: false });
      s2.setData(dedupe(eq.map((p) => ({ time: ts(p.ts), value: p.balance }))));
      const dd = c.addSeries(LW.HistogramSeries, { color: css('--down'), priceLineVisible: false, title: 'drawdown %', priceFormat: { type: 'percent' } }, 1);
      dd.setData(dedupe(eq.map((p) => ({ time: ts(p.ts), value: -100 * p.drawdown }))));
      c.panes()[1] && c.panes()[1].setStretchFactor(0.35);
      c.timeScale().fitContent();
    } else {
      document.getElementById('eq').innerHTML = '<div class="empty">No equity data yet.</div>';
    }
  }
  const dedupe = (pts) => { const out = []; let last = -1; for (const p of pts) { if (p.time > last) { out.push(p); last = p.time; } } return out; };

  function setAlive(iso) {
    const el = document.getElementById('alive'), dot = document.getElementById('alivedot');
    if (!iso) { el.textContent = 'trader not seen yet'; return; }
    const age = (Date.now() - Date.parse(iso)) / 1000;
    el.textContent = age < 180 ? 'trader running' : `trader last seen ${when(iso)}`;
    dot.style.background = age < 180 ? css('--up') : css('--down');
  }

  function tradeTable(rows, open) {
    if (!rows.length) return `<div class="empty">${open ? 'No open trades.' : 'No trades yet.'}</div>`;
    return `<div class="tw"><table><tr><th>#</th><th>When</th><th>Market</th><th>Side</th>${open ? '' : '<th>Result</th>'}<th>Entry</th><th>Stop</th><th>Target</th><th>Risk</th>${open ? '' : '<th>Exit</th><th>Post-mortem</th>'}</tr>
      ${rows.map((t) => `<tr class="click" onclick="location.hash='trade/${t.id}'"><td>${t.id}${t.shadow ? ' <span class="muted">(virtual)</span>' : ''}</td><td>${when(t.opened_at)}</td><td>${esc(t.symbol)} ${esc(t.tf)}</td>
        <td class="${t.side === 'buy' ? 'pos' : 'neg'}">${t.side}</td>${open ? '' : `<td>${t.status === 'open' ? 'open' : R(t.r)}</td>`}
        <td>${num(t.entry, 5)}</td><td>${num(t.stop_now || t.stop, 5)}</td><td>${t.target ? num(t.target, 5) : '-'}</td><td>${t.shadow ? '-' : pct(t.risk_pct, 2)}</td>
        ${open ? '' : `<td>${esc(t.exit_reason || '')}</td><td class="wrap">${(t.tags || []).map((x) => `<span class="tag">${esc(x)}</span>`).join('')}</td>`}</tr>`).join('')}
    </table></div>`;
  }

  // ---------------------------------------------------------------- strategies
  async function strategies() {
    const rows = await get('/api/strategies');
    if (!rows.length) { $main.innerHTML = '<div class="card"><h2>Strategy league</h2><div class="empty">Nothing has passed validation yet. That is normal at the start: most ideas are luck, and the filter is strict on purpose.</div></div>'; return; }
    $main.innerHTML = `<div class="card"><h2>Strategy league</h2><div class="sub">Everything Gimenez invented that passed the validation gauntlet. "Backtest" = out-of-sample expectation (walk-forward + holdout). A big negative gap between live and backtest means overfitting.</div>
      <div class="tw"><table><tr><th>Stage</th><th>Strategy</th><th>Backtest</th><th>Real</th><th>Shadow</th><th>Gap</th><th>Robustness</th></tr>
      ${rows.map((s, i) => `<tr class="click" onclick="document.getElementById('sd${i}').style.display=document.getElementById('sd${i}').style.display==='none'?'':'none'">
        <td>${stagePill(s.stage)}</td><td class="wrap"><b>${esc(s.name)}</b></td>
        <td>${R(s.backtest.mean)} <span class="muted">n=${s.backtest.n ?? '-'}</span></td>
        <td>${s.real.n ? `${R(s.real.mean)} <span class="muted">n=${s.real.n}</span>` : '-'}</td>
        <td>${s.shadow.n ? `${R(s.shadow.mean)} <span class="muted">n=${s.shadow.n}</span>` : '-'}</td>
        <td>${R(s.gap_r)}</td><td><span class="muted">DSR ${num(s.dsr)} · plateau ${pct(s.plateau, 0)}</span></td></tr>
        <tr id="sd${i}" style="display:none"><td colspan="7" class="wrap">
          <p><b>Rules:</b> ${esc(s.description)}</p>
          <p><b>Holdout:</b> ${s.holdout ? `${s.holdout.n} trades, ${num(s.holdout.mean, 3)}R, PF ${num(s.holdout.pf)}, p=${num(s.holdout.p, 4)} (look #${s.holdout.looks})` : '-'} ·
             <b>Walk-forward:</b> ${s.walk_forward ? `${s.walk_forward.n} trades, ${num(s.walk_forward.mean, 3)}R, efficiency ${pct(s.walk_forward.efficiency, 0)}` : '-'} ·
             <b>Similar markets:</b> ${(s.peers || []).map((p) => `${esc(p[0])} ${num(p[1])}R`).join(', ') || 'none available'}</p>
          ${s.paused.length ? `<p><b>Paused in:</b> ${s.paused.map(esc).join(', ')}</p>` : ''}
          <pre class="muted">${esc(s.notes || '')}</pre></td></tr>`).join('')}
      </table></div></div>`;
  }

  // ---------------------------------------------------------------- trades
  async function trades(kind) {
    kind = kind || 'real';
    const rows = await get(`/api/trades?kind=${kind}&limit=300`);
    $main.innerHTML = `<div class="card"><div class="backbar"><h2 style="margin:0">Trades</h2>
      ${['real', 'shadow', 'all'].map((k) => `<button class="link" onclick="location.hash='trades/${k}'">${k === kind ? `<b>${k}</b>` : k}</button>`).join(' · ')}</div>
      <div class="sub">Tap a trade to replay it: what it saw, why, the odds, what happened.</div>${tradeTable(rows, false)}</div>`;
  }

  async function tradeDetail(id) {
    const d = await get(`/api/trade/${id}`);
    const t = d.trade, dec = d.decision || {}, odds = dec.odds || {}, f = t.features || {}, pm = t.postmortem || {};
    $main.innerHTML = `<div class="backbar"><button class="link" onclick="history.back()">&larr; back</button></div>
      <div class="card"><h2>#${t.id} ${t.shadow ? '(virtual) ' : ''}${t.side.toUpperCase()} ${esc(t.symbol)} ${esc(t.tf)} ${t.status === 'closed' ? R(t.r) : '<span class="pill">open</span>'}</h2>
      <div class="sub">${esc(d.strategy.name || t.strategy_id)} · ${stagePill(t.stage)}</div>
      <div id="replay" class="chart tall"></div>
      <div class="grid2" style="margin-top:10px">
        <div class="box"><h3>What it saw</h3>
          <p>${esc(d.strategy.description || '')}</p>
          <p>Volatility: <b>${esc(f.vol_regime)}</b> (ATR ${num(f.atr, 5)}) · Session: <b>${esc(f.session)}</b> · Trend: <b>${esc(f.trend)}</b></p>
          <p>Spread at entry: ${num(f.spread, 5)} (${pct(f.spread_vs_risk)} of the risk)</p></div>
        <div class="box"><h3>Why it took it</h3><p>${esc(dec.reason || '')}</p>
          <p>Entry ${num(t.entry, 5)} · stop ${num(t.stop, 5)} · target ${t.target ? num(t.target, 5) : 'none (trailing)'}</p>
          ${t.shadow ? '' : `<p>Risk ${pct(t.risk_pct, 2)} (${money(t.risk_amount)}) · size ${t.qty}</p>`}</div>
        <div class="box"><h3>Expected odds</h3>
          <p>Win rate ${pct(odds.expected_win_rate, 0)} · ${R(odds.expected_r)} per trade · PF ${num(odds.expected_pf)}</p>
          <p class="muted">from ${odds.based_on_trades ?? '-'} out-of-sample backtest trades${odds.regime_expectancy != null ? `; in this market condition ${num(odds.regime_expectancy)}R` : ''}</p>
          <p>Live so far: ${odds.live_trades || 0} real (${R(odds.live_r)}), ${odds.shadow_trades || 0} virtual (${R(odds.shadow_r)})</p></div>
        <div class="box"><h3>What happened</h3>
          ${t.status === 'closed' ? `<p>${esc(t.exit_reason)} at ${num(t.exit_price, 5)} after ${t.bars_held} bars: ${R(t.r)}${t.shadow ? '' : ` (${money(t.pnl)})`}${t.estimated ? ' <span class="muted">(exit estimated)</span>' : ''}</p>
          <p>Best ${R(t.mfe_r)} / worst ${R(-(t.mae_r || 0))} along the way</p>
          ${(pm.tags || []).map((x) => `<span class="tag">${esc(x)}</span>`).join('')}
          <ul style="margin:6px 0 0;padding-left:18px">${(pm.notes || []).map((n) => `<li>${esc(n)}</li>`).join('')}</ul>` : '<p>Still open.</p>'}</div>
      </div></div>`;
    drawReplay(document.getElementById('replay'), dec.snapshot, d.after, t);
  }

  function drawReplay(el, snap, after, t) {
    if (!snap || !snap.t) { el.innerHTML = '<div class="empty">No chart snapshot stored for this one.</div>'; return; }
    const c = baseChart(el);
    const T = snap.t.map(ts), bars = [];
    for (let i = 0; i < T.length; i++) if (snap.open[i] != null) bars.push({ time: T[i], open: snap.open[i], high: snap.high[i], low: snap.low[i], close: snap.close[i] });
    if (after && after.t) for (let i = 0; i < after.t.length; i++) bars.push({ time: ts(after.t[i]), open: after.open[i], high: after.high[i], low: after.low[i], close: after.close[i] });
    const cs = c.addSeries(LW.CandlestickSeries, { upColor: css('--up'), downColor: css('--down'), wickUpColor: css('--up'), wickDownColor: css('--down'), borderVisible: false, priceLineVisible: false, lastValueVisible: false });
    cs.setData(dedupe(bars));
    const palette = ['#2a6fdb', '#d97706', '#7c3aed', '#0891b2', '#db2777'];
    Object.entries(snap.overlays || {}).forEach(([name, vals], k) => {
      const s = c.addSeries(LW.LineSeries, { color: palette[k % palette.length], lineWidth: 1, title: name, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false });
      s.setData(T.map((x, i) => (vals[i] == null ? { time: x } : { time: x, value: vals[i] })));
    });
    const lv = (price, color, title, style) => price != null && cs.createPriceLine({ price, color, lineWidth: 1, lineStyle: style, title, axisLabelVisible: true });
    lv(t.entry, css('--accent'), 'entry', 0);
    lv(t.stop, css('--down'), 'stop', 2);
    if (t.target) lv(t.target, css('--up'), 'target', 2);
    if (t.stop_now && t.stop_now !== t.stop) lv(t.stop_now, css('--warn'), 'stop now', 1);
    const m = [];
    const sig = T[T.length - 1];
    m.push({ time: sig, position: t.side === 'buy' ? 'belowBar' : 'aboveBar', shape: t.side === 'buy' ? 'arrowUp' : 'arrowDown', color: css('--accent'), text: `signal ${t.side}` });
    if (t.closed_at) {
      const xt = ts(t.closed_at), all = bars.map((b) => b.time);
      let snapT = all[0]; for (const x of all) if (x <= xt) snapT = x;
      m.push({ time: snapT, position: 'inBar', shape: 'square', color: (t.r || 0) >= 0 ? css('--up') : css('--down'), text: `exit ${t.r >= 0 ? '+' : ''}${num(t.r)}R` });
    }
    LW.createSeriesMarkers(cs, m.sort((a, b) => a.time - b.time));
    c.timeScale().setVisibleLogicalRange({ from: Math.max(0, T.length - 90), to: Math.min(bars.length + 5, T.length + Math.max(40, (t.bars_held || 0) + 15)) });
  }

  // ---------------------------------------------------------------- decisions
  async function decisions(action) {
    const rows = await get(`/api/decisions?limit=300${action ? `&action=${action}` : ''}`);
    $main.innerHTML = `<div class="card"><div class="backbar"><h2 style="margin:0">Setups found</h2>
      ${['', 'taken', 'shadow', 'passed'].map((k) => `<button class="link" onclick="location.hash='decisions/${k}'">${k === (action || '') ? `<b>${k || 'all'}</b>` : (k || 'all')}</button>`).join(' · ')}</div>
      <div class="sub">Every signal any strategy produced, and what Gimenez did with it. Tap one to see the chart it saw.</div>
      ${rows.length ? `<div class="tw"><table><tr><th>When</th><th>Market</th><th>Side</th><th>Action</th><th>Reason</th><th>Strategy</th></tr>
      ${rows.map((d) => `<tr class="click" onclick="location.hash='${d.trade_id ? `trade/${d.trade_id}` : `decision/${d.id}`}'"><td>${when(d.ts)}</td><td>${esc(d.symbol)} ${esc(d.tf)}</td>
        <td class="${d.side === 'buy' ? 'pos' : 'neg'}">${d.side}</td><td><span class="pill">${esc(d.action)}</span></td><td class="wrap">${esc(d.reason)}</td><td class="wrap muted">${esc(d.strategy)}</td></tr>`).join('')}
      </table></div>` : '<div class="empty">No setups yet.</div>'}</div>`;
  }

  async function decisionDetail(id) {
    const d = await get(`/api/decision/${id}`);
    const x = d.decision, lv = (x.snapshot || {}).levels || {};
    $main.innerHTML = `<div class="backbar"><button class="link" onclick="history.back()">&larr; back</button></div>
      <div class="card"><h2>${esc(x.action)}: ${x.side} ${esc(x.symbol)} ${esc(x.tf)}</h2><div class="sub">${when(x.ts)} · ${esc(d.strategy.name || '')}</div>
      <div id="replay" class="chart tall"></div><div class="box" style="margin-top:10px"><p><b>Reason:</b> ${esc(x.reason)}</p><p>${esc(d.strategy.description || '')}</p></div></div>`;
    drawReplay(document.getElementById('replay'), x.snapshot, null, { side: x.side, entry: lv.entry, stop: lv.stop, target: lv.target });
  }

  // ---------------------------------------------------------------- research
  async function research() {
    const r = await get('/api/research');
    const t = r.totals;
    const hints = Object.entries(r.hints || {});
    $main.innerHTML = `<div class="card"><h2>Discovery funnel</h2><div class="sub">Thousands of ideas are tried; most that look good are luck. Only the best few per run may look at the untouched holdout, and every look makes the next one stricter.</div>
      <div class="tiles">${[['Ideas tested', t.ideas], ['Best candidates checked', t.candidates], ['Holdout looks', t.looks], ['Passed everything', t.passed],
        ['In shadow', r.stages.shadow || 0], ['Real money', (r.stages.probation || 0) + (r.stages.active || 0) + (r.stages.proven || 0)], ['Retired', r.stages.retired || 0]]
        .map(([k, v]) => `<div class="tile"><div class="k">${k}</div><div class="v">${(v || 0).toLocaleString()}</div></div>`).join('')}</div></div>
      ${hints.length ? `<div class="card"><h2>Lessons fed back into research</h2>${hints.map(([k, v]) => `<p><b>${esc(k)}</b>: start with stops >= ${v.sl_atr_min} ATR (${esc(v.evidence)} of losses were stopped too tight)</p>`).join('')}</div>` : ''}
      <div class="card"><h2>Recent research runs</h2>${r.runs.length ? `<div class="tw"><table><tr><th>When</th><th>Market</th><th>Bars</th><th>Ideas</th><th>Holdout looks</th><th>Passed</th><th>Where the best ideas failed</th></tr>
      ${r.runs.map((x) => `<tr><td>${when(x.ts)}</td><td>${esc(x.symbol)} ${esc(x.tf)}</td><td>${x.bars}</td><td>${x.trials}</td><td>${x.holdout_looks}</td><td>${x.passed}</td>
        <td class="wrap muted">${((x.summary || {}).top || []).map((c) => `${esc(c.name)}: ${esc((c.why || [])[0] || c.stage)}`).join('<br>')}</td></tr>`).join('')}</table></div>` : '<div class="empty">No research runs yet (it needs downloaded history first).</div>'}</div>
      <div class="card"><h2>Price history on disk</h2><div class="tw"><table><tr><th>Market</th><th>TF</th><th>Bars</th></tr>${r.history.map((h) => `<tr><td>${esc(h.symbol)}</td><td>${esc(h.tf)}</td><td>${h.bars.toLocaleString()}</td></tr>`).join('')}</table></div></div>`;
  }

  // ---------------------------------------------------------------- markets
  async function markets() {
    const w = await get('/api/watchlist');
    $main.innerHTML = `<div class="card"><h2>Markets</h2><div class="sub">Every instrument on the account, scored on cost (spread vs typical hourly range), trading hours and activity. ${w.screening_left ? `Screening in progress: ${w.screening_left} left.` : `Last screen: ${when(w.last_screen)}.`}</div>
      ${w.rows.length ? `<div class="tw"><table><tr><th></th><th>Market</th><th>Class</th><th>Score</th><th>Spread / ATR</th><th>Hours/day</th><th>Note</th></tr>
      ${w.rows.map((r) => `<tr><td>${r.chosen ? '★' : ''}</td><td>${esc(r.symbol)}</td><td>${esc(r.asset_class)}</td><td>${num(r.score)}</td><td>${pct(r.cost_ratio)}</td><td>${num(r.hours_open, 0)}</td><td class="wrap muted">${esc(r.reason)}</td></tr>`).join('')}</table></div>` : '<div class="empty">Not screened yet.</div>'}</div>`;
  }

  async function why() {
    const [txt, ev] = await Promise.all([get('/api/why?hours=24'), get('/api/events?limit=150')]);
    $main.innerHTML = `<div class="card"><h2>Last 24 hours</h2><pre>${esc(txt)}</pre></div>
      <div class="card"><h2>Event log</h2><ul class="events">${ev.map((e) => `<li><span class="kind">${esc(e.kind)}</span><span class="muted">${when(e.ts)}</span> ${esc(e.message)}</li>`).join('')}</ul></div>`;
  }

  // ---------------------------------------------------------------- router
  async function route() {
    clearCharts();
    clearTimeout(timer);
    const [tab, arg] = (location.hash.replace('#', '') || 'overview').split('/');
    document.querySelectorAll('nav button').forEach((b) => b.classList.toggle('on', b.dataset.tab === tab || (tab === 'trade' && b.dataset.tab === 'trades') || (tab === 'decision' && b.dataset.tab === 'decisions')));
    try {
      if (tab === 'overview') await overview();
      else if (tab === 'strategies') await strategies();
      else if (tab === 'trades') await trades(arg);
      else if (tab === 'trade') await tradeDetail(arg);
      else if (tab === 'decisions') await decisions(arg);
      else if (tab === 'decision') await decisionDetail(arg);
      else if (tab === 'research') await research();
      else if (tab === 'markets') await markets();
      else if (tab === 'why') await why();
    } catch (e) {
      $main.innerHTML = `<div class="card"><div class="empty">Could not load: ${esc(e.message)}</div></div>`;
    }
    if (tab === 'overview') timer = setTimeout(route, 60000);
  }
  document.getElementById('nav').addEventListener('click', (e) => { const b = e.target.closest('button'); if (b) location.hash = b.dataset.tab; });
  window.addEventListener('hashchange', route);
  route();
})();
