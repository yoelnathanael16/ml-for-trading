/* ──────────────────────────────────────────────────────────
   Quant ML Research — Static Dashboard
   Reads pre-computed JSON from docs/data/ and renders the
   8-tab dashboard with Chart.js 4.
   ────────────────────────────────────────────────────────── */

"use strict";

// ── Chart.js global defaults (dark theme) ────────────────
Chart.defaults.color = "#8b9bb4";
Chart.defaults.borderColor = "#2d3d54";
Chart.defaults.font.family = "'Segoe UI', system-ui, sans-serif";
Chart.defaults.font.size = 12;

// Detect whether we're running from the file system (file://) vs a server
const DATA_BASE = (() => {
  if (location.protocol === "file:") {
    // When opening index.html directly — data/ is relative to the HTML file
    return "data/";
  }
  return "data/";
})();

// ── Global state ─────────────────────────────────────────
let manifest = null;
let currentTicker = null;
let currentData = null;
let portfolioData = null;
let statusData = null;
const charts = {};  // name → Chart instance (for destroy-before-recreate)

// ── Helpers ───────────────────────────────────────────────
const pct = (v, decimals = 2) =>
  v == null ? "N/A" : `${(v * 100).toFixed(decimals)}%`;
const pctDirect = (v, decimals = 2) =>
  v == null ? "N/A" : `${v.toFixed(decimals)}%`;
const fmtNum = (v, decimals = 2) =>
  v == null ? "N/A" : Number(v).toFixed(decimals);
const fmtDollar = (v) =>
  v == null ? "N/A" : `$${Number(v).toFixed(2)}`;

function colorCell(td, value) {
  if (value == null) return;
  if (value > 0) td.classList.add("positive");
  else if (value < 0) td.classList.add("negative");
}

function destroyChart(name) {
  if (charts[name]) { charts[name].destroy(); delete charts[name]; }
}

function showError(msg) {
  const el = document.getElementById("error-banner");
  el.textContent = msg;
  el.classList.remove("hidden");
}

async function fetchJSON(path) {
  const resp = await fetch(path);
  if (!resp.ok) throw new Error(`HTTP ${resp.status} for ${path}`);
  return resp.json();
}

// ── Initialise ────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", async () => {
  // Tab navigation
  document.querySelectorAll(".nav-item").forEach(link => {
    link.addEventListener("click", (e) => {
      e.preventDefault();
      switchTab(link.dataset.tab);
    });
  });

  // SHAP model switcher
  document.getElementById("shap-model-select").addEventListener("change", () => {
    if (currentData) renderTab8(currentData);
  });

  try {
    manifest = await fetchJSON(DATA_BASE + "manifest.json");
    populateTickerSelect(manifest.tickers);

    // Load status + portfolio in parallel
    [statusData, portfolioData] = await Promise.all([
      fetchJSON(DATA_BASE + "status.json").catch(() => null),
      fetchJSON(DATA_BASE + "portfolio.json").catch(() => null),
    ]);

    document.getElementById("build-meta").textContent =
      "Built: " + (manifest.build_time ? manifest.build_time.slice(0, 16).replace("T", " ") : "unknown");

    // Trigger first ticker load
    const sel = document.getElementById("ticker-select");
    sel.addEventListener("change", () => loadTicker(sel.value));
    await loadTicker(sel.value);
  } catch (err) {
    showError("Failed to load manifest: " + err.message);
    console.error(err);
  }
});

function populateTickerSelect(tickers) {
  const sel = document.getElementById("ticker-select");
  sel.innerHTML = "";
  tickers.forEach(t => {
    const opt = document.createElement("option");
    opt.value = t;
    opt.textContent = t;
    sel.appendChild(opt);
  });
}

async function loadTicker(ticker) {
  currentTicker = ticker;
  try {
    currentData = await fetchJSON(DATA_BASE + ticker + ".json");

    // Render status banner
    if (statusData && statusData[ticker]) {
      renderStatusBanner(statusData[ticker]);
    }

    // Render the currently active tab
    const activeTab = document.querySelector(".nav-item.active")?.dataset.tab || "tab1";
    renderTab(activeTab);
  } catch (err) {
    showError("Failed to load data for " + ticker + ": " + err.message);
    console.error(err);
  }
}

function switchTab(tabId) {
  document.querySelectorAll(".tab-section").forEach(s => {
    s.classList.remove("active");
    s.classList.add("hidden");
  });
  const el = document.getElementById(tabId);
  if (el) { el.classList.remove("hidden"); el.classList.add("active"); }

  document.querySelectorAll(".nav-item").forEach(l => l.classList.remove("active"));
  const link = document.querySelector(`[data-tab="${tabId}"]`);
  if (link) link.classList.add("active");

  renderTab(tabId);
}

function renderTab(tabId) {
  if (!currentData) return;
  switch (tabId) {
    case "tab1": renderTab1(currentData); break;
    case "tab2": renderTab2(currentData); break;
    case "tab3": renderTab3(currentData); break;
    case "tab4": renderTab4(portfolioData); break;
    case "tab5": renderTab5(currentData); break;
    case "tab6": renderTab6(currentData); break;
    case "tab7": renderTab7(currentData); break;
    case "tab8": renderTab8(currentData); break;
  }
}

// ── Status banner ─────────────────────────────────────────
function renderStatusBanner(s) {
  document.getElementById("stat-refresh").textContent =
    s.last_refresh ? s.last_refresh.slice(0, 16).replace("T", " ") : "Never";
  document.getElementById("stat-retrain").textContent =
    s.last_retrain ? s.last_retrain.slice(0, 16).replace("T", " ") : "Never";
  const anomCard = document.getElementById("stat-anomaly-card");
  const anomVal = document.getElementById("stat-anomaly");
  if (s.anomaly_flag === true) {
    anomVal.textContent = "⚠️ ANOMALY";
    anomCard.style.borderColor = "#e74c3c";
  } else if (s.anomaly_flag === false) {
    anomVal.textContent = "✅ Normal";
    anomCard.style.borderColor = "#2ecc71";
  } else {
    anomVal.textContent = "—";
  }
  document.getElementById("status-banner").classList.remove("hidden");
}

// ── TAB 1: Benchmarks ─────────────────────────────────────
function renderTab1(data) {
  document.getElementById("tab1-title").textContent =
    `Model Benchmarking Results: ${data.ticker}`;

  const t1 = data.tab1_benchmarks || {};
  const errEl = document.getElementById("tab1-error");

  if (t1.error || !t1.rows || t1.rows.length === 0) {
    errEl.textContent = t1.error || "Benchmarking results not available.";
    errEl.classList.remove("hidden");
    return;
  }
  errEl.classList.add("hidden");

  // Table
  const tbody = document.querySelector("#bench-table tbody");
  tbody.innerHTML = "";
  t1.rows.forEach(row => {
    const tr = document.createElement("tr");
    const cells = [
      { v: row.model, raw: null },
      { v: row.accuracy, raw: null },
      { v: pct(row.total_return_base), raw: row.total_return_base },
      { v: fmtNum(row.sharpe_base), raw: row.sharpe_base },
      { v: pct(row.max_drawdown_base), raw: row.max_drawdown_base },
      { v: pct(row.total_return_adv), raw: row.total_return_adv },
      { v: fmtNum(row.sharpe_adv), raw: row.sharpe_adv },
      { v: pct(row.max_drawdown_adv), raw: row.max_drawdown_adv },
    ];
    cells.forEach(({ v, raw }) => {
      const td = document.createElement("td");
      td.textContent = v;
      if (raw != null) colorCell(td, raw);
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });

  // Charts
  const labels = t1.rows.map(r => r.model);
  const retBase = t1.rows.map(r => r.total_return_base != null ? (r.total_return_base * 100) : null);
  const retAdv  = t1.rows.map(r => r.total_return_adv != null ? (r.total_return_adv * 100) : null);
  const shrBase = t1.rows.map(r => r.sharpe_base);
  const shrAdv  = t1.rows.map(r => r.sharpe_adv);

  _makeGroupedBar("chart-returns", labels, [
    { label: "Base (%)", data: retBase, backgroundColor: "rgba(99,179,237,0.8)" },
    { label: "Adv (%)", data: retAdv, backgroundColor: "rgba(0,210,211,0.8)" },
  ]);

  _makeGroupedBar("chart-sharpe", labels, [
    { label: "Base", data: shrBase, backgroundColor: "rgba(183,148,246,0.8)" },
    { label: "Adv", data: shrAdv, backgroundColor: "rgba(246,173,85,0.8)" },
  ]);
}

// ── TAB 2: GMM Regimes ────────────────────────────────────
function renderTab2(data) {
  const t2 = data.tab2_gmm || {};
  const errEl = document.getElementById("tab2-error");

  if (t2.error || !t2.price_series || t2.price_series.length === 0) {
    errEl.textContent = t2.error || "GMM model data not available.";
    errEl.classList.remove("hidden");
    return;
  }
  errEl.classList.add("hidden");

  // Regime scatter — split into 3 datasets by regime colour
  const regimeColors = { 0: "#2ecc71", 1: "#f1c40f", 2: "#e74c3c" };
  const regimeLabels = { 0: "Low Vol", 1: "Moderate Vol", 2: "High Vol" };

  const byRegime = { 0: [], 1: [], 2: [] };
  t2.price_series.forEach(p => {
    if (p.regime != null && p.close != null) {
      byRegime[p.regime].push({ x: p.date, y: p.close });
    }
  });

  destroyChart("chart-gmm");
  const ctx = document.getElementById("chart-gmm").getContext("2d");
  charts["chart-gmm"] = new Chart(ctx, {
    type: "scatter",
    data: {
      datasets: [0, 1, 2].map(r => ({
        label: regimeLabels[r],
        data: byRegime[r],
        pointBackgroundColor: regimeColors[r],
        pointRadius: 2,
        showLine: false,
      })),
    },
    options: {
      animation: false,
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        x: { type: "category", ticks: { maxTicksLimit: 10, maxRotation: 0 }, grid: { color: "#2d3d54" } },
        y: { title: { display: true, text: "Price ($)" }, grid: { color: "#2d3d54" } },
      },
    },
  });

  // Regime stats table
  const tbody = document.querySelector("#regime-stats-table tbody");
  tbody.innerHTML = "";
  const names = { 0: "Low Vol", 1: "Med Vol", 2: "High Vol" };
  (t2.regime_stats || []).forEach(s => {
    const tr = document.createElement("tr");
    [names[s.regime_id] || s.regime_id, s.total_days,
     pctDirect(s.mean_daily_return_pct, 3), pctDirect(s.daily_vol_pct, 3)].forEach(v => {
      const td = document.createElement("td");
      td.textContent = v;
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
}

// ── TAB 3: Trading Simulator ──────────────────────────────
function renderTab3(data) {
  const t3 = data.tab3_simulator || {};
  const errEl = document.getElementById("tab3-error");

  if (t3.error && !t3.metrics) {
    errEl.textContent = t3.error;
    errEl.classList.remove("hidden");
    return;
  }
  errEl.classList.add("hidden");

  // Params
  const paramRow = document.getElementById("tab3-params");
  paramRow.innerHTML = "";
  if (t3.params) {
    const tags = [
      ["Model", t3.model || "—"],
      ["Sizing", t3.params.sizing_method],
      ["Stop Loss", `${t3.params.stop_loss_pct}%`],
      ["Profit Target", `${t3.params.profit_target_pct}%`],
      ["Trailing Stop", `${t3.params.trailing_stop_pct}%`],
      ["Time Barrier", `${t3.params.time_barrier}d`],
      ["Trend Filter", t3.params.trend_filter ? "ON" : "OFF"],
    ];
    tags.forEach(([k, v]) => {
      const span = document.createElement("div");
      span.className = "param-tag";
      span.innerHTML = `${k}: <span>${v}</span>`;
      paramRow.appendChild(span);
    });
  }

  // Metrics
  const metricsRow = document.getElementById("tab3-metrics");
  metricsRow.innerHTML = "";
  const m = t3.metrics || {};
  [
    ["Total Return", pct(m.total_return)],
    ["Sharpe Ratio", fmtNum(m.sharpe_ratio)],
    ["Max Drawdown", pct(m.max_drawdown)],
    ["Total Trades", m.num_trades != null ? m.num_trades : "—"],
  ].forEach(([label, val]) => {
    metricsRow.innerHTML += `<div class="metric-card"><div class="metric-val">${val}</div><div class="metric-label">${label}</div></div>`;
  });
  metricsRow.className = "metric-row";

  // Equity curve chart
  if (t3.equity_curve && t3.equity_curve.length > 0) {
    const labels = t3.equity_curve.map(p => p.date);
    const eqData = t3.equity_curve.map(p => p.value);
    const bhData = (t3.bh_curve || []).map(p => p.value);

    destroyChart("chart-equity");
    const ctx = document.getElementById("chart-equity").getContext("2d");
    charts["chart-equity"] = new Chart(ctx, {
      type: "line",
      data: {
        labels,
        datasets: [
          {
            label: "Buy & Hold",
            data: bhData,
            borderColor: "#8b9bb4",
            borderDash: [5, 5],
            borderWidth: 1.5,
            pointRadius: 0,
            tension: 0.1,
          },
          {
            label: "Strategy",
            data: eqData,
            borderColor: "#00d2d3",
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.1,
            fill: false,
          },
        ],
      },
      options: {
        animation: false,
        responsive: true,
        plugins: { legend: { position: "top" } },
        scales: {
          x: { ticks: { maxTicksLimit: 10, maxRotation: 0 }, grid: { color: "#2d3d54" } },
          y: { title: { display: true, text: "Normalised Value" }, grid: { color: "#2d3d54" } },
        },
      },
    });
  }

  // Trades table
  const tbody = document.querySelector("#trades-table tbody");
  tbody.innerHTML = "";
  if (t3.trades && t3.trades.length > 0) {
    t3.trades.forEach(t => {
      const tr = document.createElement("tr");
      [
        t.entry_date, t.exit_date, t.position,
        fmtDollar(t.entry_price), fmtDollar(t.exit_price),
        pctDirect(t.size_pct, 1), t.reason,
        pctDirect(t.pnl_pct, 2),
      ].forEach((v, i) => {
        const td = document.createElement("td");
        td.textContent = v;
        if (i === 7 && t.pnl_pct != null) colorCell(td, t.pnl_pct);
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
  } else {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 8;
    td.textContent = "No trades recorded for this period.";
    td.style.textAlign = "center";
    td.style.color = "#8b9bb4";
    tr.appendChild(td);
    tbody.appendChild(tr);
  }
}

// ── TAB 4: Portfolio Allocation ───────────────────────────
function renderTab4(port) {
  const errEl = document.getElementById("tab4-error");
  if (!port || port.error) {
    errEl.textContent = (port && port.error) || "Portfolio data not available.";
    errEl.classList.remove("hidden");
    return;
  }
  errEl.classList.add("hidden");

  const tickers = port.tickers || [];
  const weights = port.weights || {};
  const methods = ["EW", "RP", "MVO", "HRP"];

  // Weights table
  const tbody = document.querySelector("#portfolio-weights-table tbody");
  tbody.innerHTML = "";
  tickers.forEach(t => {
    const tr = document.createElement("tr");
    const tdT = document.createElement("td");
    tdT.textContent = t;
    tdT.style.fontWeight = "600";
    tr.appendChild(tdT);
    methods.forEach(m => {
      const td = document.createElement("td");
      const w = weights[m] && weights[m][t];
      td.textContent = w != null ? pct(w) : "—";
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });

  // Portfolio bar chart
  const colors = ["rgba(99,179,237,0.8)", "rgba(0,210,211,0.8)", "rgba(246,173,85,0.8)", "rgba(183,148,246,0.8)"];
  _makeGroupedBar("chart-portfolio", tickers, methods.map((m, i) => ({
    label: m,
    data: tickers.map(t => weights[m] && weights[m][t] != null ? weights[m][t] * 100 : null),
    backgroundColor: colors[i],
  })));

  // Performance table
  const perfTbody = document.querySelector("#portfolio-perf-table tbody");
  perfTbody.innerHTML = "";
  (port.perf_metrics || []).forEach(p => {
    const tr = document.createElement("tr");
    [
      p.method,
      pctDirect(p.expected_annual_return_pct),
      pctDirect(p.expected_annual_vol_pct),
      fmtNum(p.sharpe_ratio),
    ].forEach((v, i) => {
      const td = document.createElement("td");
      td.textContent = v;
      if (i === 1 && p.expected_annual_return_pct != null) colorCell(td, p.expected_annual_return_pct);
      tr.appendChild(td);
    });
    perfTbody.appendChild(tr);
  });
}

// ── TAB 5: HMM Regime ─────────────────────────────────────
function renderTab5(data) {
  const t5 = data.tab5_hmm || {};
  const errEl = document.getElementById("tab5-error");

  if (!t5.regime_gmm && !t5.regime_hmm) {
    errEl.textContent = "HMM regime data unavailable.";
    errEl.classList.remove("hidden");
    return;
  }
  errEl.classList.add("hidden");

  document.getElementById("hmm-gmm-regime").textContent = t5.regime_gmm || "—";
  document.getElementById("hmm-hmm-regime").textContent = t5.regime_hmm || "—";

  // Transition matrix
  const wrap = document.getElementById("hmm-transition-wrap");
  wrap.innerHTML = "";
  const tm = t5.hmm_transition_matrix;
  if (tm && tm.length > 0) {
    const rowLabels = ["Bull→", "Sideways→", "Bear→"];
    const colLabels = ["→Bull", "→Sideways", "→Bear"];
    const table = document.createElement("table");
    // Header row
    const thead = document.createElement("thead");
    const htr = document.createElement("tr");
    htr.innerHTML = "<th></th>";
    colLabels.forEach(c => { htr.innerHTML += `<th>${c}</th>`; });
    thead.appendChild(htr);
    table.appendChild(thead);
    // Body
    const tbody = document.createElement("tbody");
    tm.forEach((row, i) => {
      const tr = document.createElement("tr");
      const th = document.createElement("th");
      th.textContent = rowLabels[i] || i;
      tr.appendChild(th);
      row.forEach(v => {
        const td = document.createElement("td");
        const pv = v != null ? Math.round(parseFloat(v) * 100) : 0;
        td.textContent = v != null ? parseFloat(v).toFixed(2) : "—";
        // Blue shade from 0% to 100%
        const alpha = pv / 100;
        td.style.background = `rgba(0,210,211,${alpha * 0.5})`;
        td.style.color = alpha > 0.5 ? "#fff" : "#8b9bb4";
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    wrap.appendChild(table);
  } else {
    wrap.textContent = "Transition matrix not available.";
    wrap.style.color = "#8b9bb4";
  }
}

// ── TAB 6: Risk & Volatility ──────────────────────────────
function renderTab6(data) {
  const t6 = data.tab6_risk || {};
  document.getElementById("tab6-error").classList.add("hidden");

  // Volatility
  document.getElementById("risk-garch").textContent =
    t6.garch_vol != null ? pct(t6.garch_vol) : "N/A";
  document.getElementById("risk-roll").textContent =
    t6.rolling_vol != null ? pct(t6.rolling_vol) : "N/A";

  const sigBox = document.getElementById("risk-vol-signal");
  if (t6.garch_vol != null && t6.rolling_vol != null) {
    if (t6.garch_vol > t6.rolling_vol * 1.1) {
      sigBox.textContent = "⚠️ GARCH forecasts HIGHER volatility than recent history — consider reducing position size.";
      sigBox.style.borderColor = "#f1c40f";
    } else if (t6.garch_vol < t6.rolling_vol * 0.9) {
      sigBox.textContent = "✅ GARCH forecasts LOWER volatility — market may be calming.";
      sigBox.style.borderColor = "#2ecc71";
    } else {
      sigBox.textContent = "GARCH forecast is in line with recent rolling volatility.";
    }
    sigBox.classList.remove("hidden");
  }

  // CVaR / VaR
  document.getElementById("risk-cvar95").textContent = t6.cvar_95 != null ? pct(t6.cvar_95) : "N/A";
  document.getElementById("risk-cvar99").textContent = t6.cvar_99 != null ? pct(t6.cvar_99) : "N/A";
  document.getElementById("risk-var95").textContent  = t6.var_95  != null ? pct(t6.var_95)  : "N/A";
  document.getElementById("risk-var99").textContent  = t6.var_99  != null ? pct(t6.var_99)  : "N/A";

  const scale = t6.position_scale;
  if (scale != null) {
    const dir = scale < 1 ? "Reduce" : "Increase";
    document.getElementById("risk-scale").textContent =
      `${fmtNum(scale)}× (${dir} by ${Math.abs((1 - scale) * 100).toFixed(0)}%)`;
  }

  // Anomaly
  const anomBox = document.getElementById("anomaly-box");
  if (t6.anomaly_flag === true) {
    anomBox.textContent = `⚠️ MARKET ANOMALY DETECTED — Score: ${fmtNum(t6.anomaly_score, 4)}. Exercise caution.`;
    anomBox.style.borderColor = "#e74c3c";
    anomBox.style.color = "#e74c3c";
  } else if (t6.anomaly_flag === false) {
    anomBox.textContent = `✅ Market conditions appear normal — Anomaly score: ${fmtNum(t6.anomaly_score, 4)}`;
    anomBox.style.borderColor = "#2ecc71";
    anomBox.style.color = "#2ecc71";
  } else {
    anomBox.textContent = "Anomaly data unavailable.";
  }
}

// ── TAB 7: Mean Reversion ─────────────────────────────────
function renderTab7(data) {
  const t7 = data.tab7_mr || {};
  document.getElementById("tab7-error").classList.add("hidden");

  document.getElementById("mr-zscore").textContent =
    t7.zscore != null ? fmtNum(t7.zscore, 3) : "N/A";

  if (t7.halflife == null || t7.halflife > 252) {
    document.getElementById("mr-halflife").textContent = "Non-mean-reverting";
  } else {
    document.getElementById("mr-halflife").textContent = `${fmtNum(t7.halflife, 1)} days`;
  }

  const signalMap = { 1: "📈 Long (MR)", "-1": "📉 Short (MR)", 0: "➡️ Neutral" };
  document.getElementById("mr-signal").textContent =
    t7.mr_signal != null ? (signalMap[String(t7.mr_signal)] || "Neutral") : "N/A";

  const statusBox = document.getElementById("mr-status-box");
  if (t7.is_mean_reverting === true) {
    statusBox.textContent = "✅ Mean Reverting: YES";
    statusBox.style.borderColor = "#2ecc71";
  } else if (t7.is_mean_reverting === false) {
    statusBox.textContent = "❌ Mean Reverting: NO (trending)";
    statusBox.style.borderColor = "#f1c40f";
  } else {
    statusBox.textContent = "Mean reversion status unknown.";
  }

  // Z-score time series
  if (t7.zscore_series && t7.zscore_series.length > 0) {
    const labels = t7.zscore_series.map(p => p.date);
    const values = t7.zscore_series.map(p => p.zscore);
    const n = labels.length;

    destroyChart("chart-zscore");
    const ctx = document.getElementById("chart-zscore").getContext("2d");
    charts["chart-zscore"] = new Chart(ctx, {
      type: "line",
      data: {
        labels,
        datasets: [
          { label: "Z-Score", data: values, borderColor: "#00d2d3", borderWidth: 1.5, pointRadius: 0, tension: 0.2 },
          { label: "+2σ", data: Array(n).fill(2), borderColor: "#e74c3c", borderDash: [5,3], borderWidth: 1, pointRadius: 0 },
          { label: "-2σ", data: Array(n).fill(-2), borderColor: "#2ecc71", borderDash: [5,3], borderWidth: 1, pointRadius: 0 },
          { label: "+0.5σ", data: Array(n).fill(0.5), borderColor: "#f1c40f", borderDash: [3,3], borderWidth: 1, pointRadius: 0 },
          { label: "-0.5σ", data: Array(n).fill(-0.5), borderColor: "#f1c40f", borderDash: [3,3], borderWidth: 1, pointRadius: 0 },
        ],
      },
      options: {
        animation: false,
        responsive: true,
        plugins: { legend: { position: "top" } },
        scales: {
          x: { ticks: { maxTicksLimit: 10, maxRotation: 0 }, grid: { color: "#2d3d54" } },
          y: { title: { display: true, text: "Z-Score" }, grid: { color: "#2d3d54" } },
        },
      },
    });
  }
}

// ── TAB 8: Explainability ─────────────────────────────────
function renderTab8(data) {
  const t8 = data.tab8_shap || {};
  const errEl = document.getElementById("tab8-error");
  const selectedModel = document.getElementById("shap-model-select").value;
  const importances = t8[selectedModel];

  if (!importances || Object.keys(importances).length === 0) {
    errEl.textContent = `No SHAP importances available for ${selectedModel}.`;
    errEl.classList.remove("hidden");
    destroyChart("chart-shap");
    document.querySelector("#shap-top5-table tbody").innerHTML = "";
    return;
  }
  errEl.classList.add("hidden");

  // Sort by absolute importance, take top 15
  const entries = Object.entries(importances)
    .map(([f, v]) => [f, v != null ? Math.abs(v) : 0])
    .sort((a, b) => a[1] - b[1])
    .slice(-15);

  const labels = entries.map(e => e[0]);
  const values = entries.map(e => e[1]);
  const barColors = values.map(v => v >= 0 ? "#00d2d3" : "#e74c3c");

  destroyChart("chart-shap");
  const ctx = document.getElementById("chart-shap").getContext("2d");
  charts["chart-shap"] = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [{
        label: "Mean |SHAP Value|",
        data: values,
        backgroundColor: barColors,
        borderRadius: 4,
      }],
    },
    options: {
      indexAxis: "y",
      animation: false,
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        x: { title: { display: true, text: "Mean |SHAP Value|" }, grid: { color: "#2d3d54" } },
        y: { grid: { color: "#2d3d54" } },
      },
    },
  });

  // Top 5 table
  const top5 = Object.entries(importances)
    .filter(([, v]) => v != null)
    .map(([f, v]) => [f, Math.abs(v)])
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5);

  const tbody = document.querySelector("#shap-top5-table tbody");
  tbody.innerHTML = "";
  top5.forEach(([feat, val], i) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${i + 1}</td><td>${feat}</td><td>${fmtNum(val, 4)}</td>`;
    tbody.appendChild(tr);
  });
}

// ── Chart helper: grouped bar ─────────────────────────────
function _makeGroupedBar(canvasId, labels, datasets) {
  destroyChart(canvasId);
  const ctx = document.getElementById(canvasId).getContext("2d");
  charts[canvasId] = new Chart(ctx, {
    type: "bar",
    data: { labels, datasets },
    options: {
      animation: false,
      responsive: true,
      plugins: { legend: { position: "top" } },
      scales: {
        x: { grid: { color: "#2d3d54" } },
        y: { grid: { color: "#2d3d54" } },
      },
    },
  });
}
