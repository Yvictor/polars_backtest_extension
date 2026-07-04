# ruff: noqa: E501  (the template string embeds CSS/JS with long lines)
"""Self-contained HTML template for the interactive backtest report.

The template has no external assets (no CDN, no plotly): charts are inline SVG
rendered by vanilla JS, so the output file works offline and in notebook
iframes. Placeholders: __TITLE__, __PAYLOAD__, __VERSION__.
"""

TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
.viz-root {
  --page:        #f9f9f7;
  --surface-1:   #fcfcfb;
  --ink:         #0b0b0b;
  --ink-2:       #52514e;
  --muted:       #898781;
  --grid:        #e1e0d9;
  --axis:        #c3c2b7;
  --border:      rgba(11,11,11,0.10);
  --s1:          #2a78d6;  /* strategy */
  --s2:          #1baf7a;  /* benchmark */
  --neg:         #e34948;  /* drawdown / losses */
  --div-mid:     #f0efec;
  --div-pos:     #1c5cab;
  --div-neg:     #b73634;
  --tip-bg:      #1a1a19;
  --tip-ink:     #ffffff;
}
@media (prefers-color-scheme: dark) {
  .viz-root {
    --page:        #0d0d0d;
    --surface-1:   #1a1a19;
    --ink:         #ffffff;
    --ink-2:       #c3c2b7;
    --muted:       #898781;
    --grid:        #2c2c2a;
    --axis:        #383835;
    --border:      rgba(255,255,255,0.10);
    --s1:          #3987e5;
    --s2:          #199e70;
    --neg:         #e66767;
    --div-mid:     #383835;
    --div-pos:     #6da7ec;
    --div-neg:     #e66767;
    --tip-bg:      #fcfcfb;
    --tip-ink:     #0b0b0b;
  }
}
* { box-sizing: border-box; }
body.viz-root {
  margin: 0; background: var(--page); color: var(--ink);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  font-size: 14px; line-height: 1.45;
}
#app { max-width: 1160px; margin: 0 auto; padding: 24px 20px 48px; }
header { display: flex; flex-wrap: wrap; align-items: baseline; gap: 8px 16px; margin-bottom: 16px; }
h1 { font-size: 20px; margin: 0; font-weight: 650; }
h2 { font-size: 14px; margin: 0 0 8px; font-weight: 600; color: var(--ink-2); }
.sub { color: var(--muted); font-size: 13px; }
#chips { display: flex; flex-wrap: wrap; gap: 6px; margin-left: auto; }
.chip { border: 1px solid var(--border); background: var(--surface-1); color: var(--ink-2);
        border-radius: 999px; padding: 2px 10px; font-size: 12px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(128px, 1fr)); gap: 10px; margin-bottom: 14px; }
.tile { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 10px 12px; }
.tile .k { color: var(--muted); font-size: 12px; }
.tile .v { font-size: 20px; font-weight: 650; margin-top: 2px; }
.tile .v.pos { color: var(--s1); }
.tile .v.neg { color: var(--neg); }
.card { background: var(--surface-1); border: 1px solid var(--border); border-radius: 12px;
        padding: 14px 16px 10px; margin-bottom: 14px; }
.card-head { display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px; }
.controls { display: flex; gap: 8px; }
.seg { display: inline-flex; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
.seg button { border: 0; background: transparent; color: var(--ink-2); padding: 3px 10px; font-size: 12px;
              cursor: pointer; font-family: inherit; }
.seg button.on { background: var(--s1); color: #fff; }
.chart { position: relative; width: 100%; }
.chart svg { display: block; width: 100%; }
.chart.tall { height: 340px; } .chart.short { height: 170px; } .chart.mid { height: 260px; }
.row2 { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
@media (max-width: 800px) { .row2 { grid-template-columns: 1fr; } }
#tooltip { position: fixed; pointer-events: none; z-index: 10; display: none;
           background: var(--tip-bg); color: var(--tip-ink); border-radius: 8px;
           padding: 7px 10px; font-size: 12px; max-width: 280px;
           box-shadow: 0 4px 16px rgba(0,0,0,0.25); }
#tooltip .t { color: color-mix(in srgb, var(--tip-ink) 62%, transparent); margin-bottom: 2px; }
#tooltip .r { display: flex; gap: 10px; justify-content: space-between; }
#tooltip .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 5px; }
.hm { overflow-x: auto; }
.hm table { border-collapse: separate; border-spacing: 2px; width: 100%; }
.hm th { font-size: 11px; color: var(--muted); font-weight: 500; padding: 2px 4px; text-align: center; }
.hm td { font-size: 11px; text-align: center; padding: 4px 2px; border-radius: 4px; min-width: 44px;
         font-variant-numeric: tabular-nums; }
.hm td.y { color: var(--ink-2); font-weight: 600; background: transparent; }
.hm td.tot { font-weight: 650; }
table.stats { width: 100%; border-collapse: collapse; }
table.stats td { padding: 5px 8px; border-bottom: 1px solid var(--grid); font-variant-numeric: tabular-nums; }
table.stats td:first-child { color: var(--ink-2); }
table.stats td:last-child { text-align: right; font-weight: 550; }
.cols3 { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 0 28px; }
footer { color: var(--muted); font-size: 12px; margin-top: 20px; text-align: center; }
svg text { font-family: inherit; }
</style>
</head>
<body class="viz-root">
<div id="app">
  <header>
    <div>
      <h1 id="rpt-title"></h1>
      <div id="rpt-range" class="sub"></div>
    </div>
    <div id="chips"></div>
  </header>
  <section class="tiles" id="tiles"></section>
  <section class="card">
    <div class="card-head">
      <h2>Equity curve</h2>
      <div class="controls">
        <div class="seg" id="range-seg"></div>
        <div class="seg" id="scale-seg"></div>
      </div>
    </div>
    <div id="equity" class="chart tall"></div>
  </section>
  <section class="card">
    <h2>Drawdown</h2>
    <div id="drawdown" class="chart short"></div>
  </section>
  <section class="card" id="heatmap-card">
    <h2>Monthly returns (%)</h2>
    <div id="heatmap" class="hm"></div>
  </section>
  <div class="row2" id="trades-row">
    <section class="card">
      <h2>Trade return distribution</h2>
      <div id="hist" class="chart mid"></div>
    </section>
    <section class="card">
      <h2>MAE vs trade return</h2>
      <div id="scatter" class="chart mid"></div>
    </section>
  </div>
  <section class="card">
    <h2>Statistics</h2>
    <div class="cols3" id="stats-cols"></div>
  </section>
  <footer>generated by polars-backtest __VERSION__</footer>
</div>
<div id="tooltip"></div>
<script>
"use strict";
const P = __PAYLOAD__;

/* ---------- helpers ---------- */
const $ = (id) => document.getElementById(id);
const SVGNS = "http://www.w3.org/2000/svg";
function el(name, attrs, parent) {
  const e = document.createElementNS(SVGNS, name);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}
function css(name) { return getComputedStyle(document.body).getPropertyValue(name).trim(); }
function fmtPct(v, d = 1) { return v == null ? "–" : (v * 100).toFixed(d) + "%"; }
function fmtSignPct(v, d = 1) { return v == null ? "–" : (v >= 0 ? "+" : "") + (v * 100).toFixed(d) + "%"; }
function fmtNum(v, d = 2) { return v == null ? "–" : (+v).toFixed(d); }
function parseDate(s) { return new Date(s + "T00:00:00Z").getTime(); }
function fmtDate(ms) { return new Date(ms).toISOString().slice(0, 10); }
function bisect(arr, x) {
  let lo = 0, hi = arr.length - 1;
  while (lo < hi) { const m = (lo + hi) >> 1; if (arr[m] < x) lo = m + 1; else hi = m; }
  if (lo > 0 && Math.abs(arr[lo - 1] - x) < Math.abs(arr[lo] - x)) lo--;
  return lo;
}
function niceTicks(lo, hi, n = 5) {
  if (!(hi > lo)) return [lo];
  const span = hi - lo, step0 = Math.pow(10, Math.floor(Math.log10(span / n)));
  let step = step0;
  for (const m of [1, 2, 2.5, 5, 10]) { if (span / (step0 * m) <= n) { step = step0 * m; break; } }
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-12; v += step) out.push(+v.toFixed(12));
  return out;
}
/* candidate multiples for log-scale ticks: 1-1.5-2-3-5-7 per decade */
function logTicks(lo, hi) {
  const out = [];
  for (let dec = Math.floor(Math.log10(lo)); dec <= Math.ceil(Math.log10(hi)); dec++)
    for (const m of [1, 1.5, 2, 3, 5, 7]) {
      const v = m * Math.pow(10, dec);
      if (v >= lo * 0.999 && v <= hi * 1.001) out.push(v);
    }
  while (out.length > 8) out.splice(1, 1), out.length % 2 && out.splice(out.length - 2, 1);
  return out.length ? out : [lo, hi];
}
function dateTicks(t0, t1) {
  const out = [], d0 = new Date(t0), d1 = new Date(t1);
  const months = (d1.getUTCFullYear() - d0.getUTCFullYear()) * 12 + d1.getUTCMonth() - d0.getUTCMonth();
  const step = months > 60 ? 12 : months > 24 ? 6 : months > 10 ? 3 : 1;
  const d = new Date(Date.UTC(d0.getUTCFullYear(), d0.getUTCMonth() + 1, 1));
  while (d.getTime() <= t1) {
    if (d.getUTCMonth() % step === 0) out.push(d.getTime());
    d.setUTCMonth(d.getUTCMonth() + 1);
  }
  return out;
}
function fmtTick(ms, span) {
  const d = new Date(ms);
  if (span > 3.2e10) return d.getUTCMonth() === 0 ? "" + d.getUTCFullYear()
    : d.getUTCFullYear() + "-" + String(d.getUTCMonth() + 1).padStart(2, "0");
  return d.getUTCFullYear() + "-" + String(d.getUTCMonth() + 1).padStart(2, "0");
}
const tipEl = $("tooltip");
function showTip(html, x, y) {
  tipEl.innerHTML = html; tipEl.style.display = "block";
  const r = tipEl.getBoundingClientRect();
  let px = x + 14, py = y + 14;
  if (px + r.width > innerWidth - 8) px = x - r.width - 14;
  if (py + r.height > innerHeight - 8) py = y - r.height - 14;
  tipEl.style.left = px + "px"; tipEl.style.top = py + "px";
}
function hideTip() { tipEl.style.display = "none"; }
function mixToward(hexFrom, hexTo, t) {
  const a = hexFrom.match(/\w\w/g).map(h => parseInt(h, 16));
  const b = hexTo.match(/\w\w/g).map(h => parseInt(h, 16));
  return "rgb(" + a.map((v, i) => Math.round(v + (b[i] - v) * t)).join(",") + ")";
}

/* ---------- state ---------- */
const state = { range: "all", scale: "linear" };
const DATES = P.daily.dates.map(parseDate);
const LAST = DATES[DATES.length - 1];
const RANGES = [["all", "All"], ["3y", "3Y"], ["1y", "1Y"], ["ytd", "YTD"], ["6m", "6M"]];
function rangeStart() {
  const d = new Date(LAST);
  switch (state.range) {
    case "3y": d.setUTCFullYear(d.getUTCFullYear() - 3); return d.getTime();
    case "1y": d.setUTCFullYear(d.getUTCFullYear() - 1); return d.getTime();
    case "6m": d.setUTCMonth(d.getUTCMonth() - 6); return d.getTime();
    case "ytd": return Date.UTC(d.getUTCFullYear(), 0, 1);
    default: return DATES[0];
  }
}
function visibleIdx() {
  const t0 = rangeStart();
  let i0 = DATES.findIndex(t => t >= t0);
  if (i0 < 0) i0 = 0;
  return [i0, DATES.length - 1];
}

/* ---------- header / tiles / chips ---------- */
function renderHeader() {
  $("rpt-title").textContent = P.title;
  $("rpt-range").textContent = P.daily.dates[0] + " → " + P.daily.dates[P.daily.dates.length - 1]
    + "  ·  " + P.daily.dates.length + " trading days";
  $("chips").innerHTML = P.params.map(p => '<span class="chip">' + p + "</span>").join("");
}
function renderTiles() {
  const s = P.stats, t = P.trade_summary || {};
  const items = [
    ["Total return", fmtSignPct(s.total_return), s.total_return],
    ["CAGR", fmtSignPct(s.cagr), s.cagr],
    ["Max drawdown", fmtPct(s.max_drawdown), s.max_drawdown],
    ["Sharpe", fmtNum(s.daily_sharpe), null],
    ["Sortino", fmtNum(s.daily_sortino), null],
    ["Calmar", fmtNum(s.calmar), null],
    ["Win ratio", fmtPct(s.win_ratio, 0), null],
    ["Trades", t.n == null ? "–" : String(t.n), null],
  ];
  $("tiles").innerHTML = items.map(([k, v, sign]) => {
    const cls = sign == null ? "" : sign >= 0 ? " pos" : " neg";
    return '<div class="tile"><div class="k">' + k + '</div><div class="v' + cls + '">' + v + "</div></div>";
  }).join("");
}

/* ---------- generic frame ---------- */
function frame(container) {
  container.innerHTML = "";
  const W = container.clientWidth, H = container.clientHeight;
  const m = { l: 56, r: 84, t: 12, b: 26 };
  const svg = el("svg", { viewBox: "0 0 " + W + " " + H, width: W, height: H }, null);
  container.appendChild(svg);
  return { svg, W, H, m, iw: W - m.l - m.r, ih: H - m.t - m.b };
}
function drawXAxis(f, x, t0, t1) {
  for (const t of dateTicks(t0, t1)) {
    el("line", { x1: x(t), x2: x(t), y1: f.m.t, y2: f.m.t + f.ih, stroke: css("--grid"), "stroke-width": 1 }, f.svg);
    const lbl = fmtTick(t, t1 - t0);
    if (lbl) el("text", { x: x(t), y: f.H - 8, "text-anchor": "middle", fill: css("--muted"), "font-size": 11 }, f.svg).textContent = lbl;
  }
  el("line", { x1: f.m.l, x2: f.m.l + f.iw, y1: f.m.t + f.ih, y2: f.m.t + f.ih, stroke: css("--axis") }, f.svg);
}

/* ---------- equity curve ---------- */
function renderEquity() {
  const c = $("equity"), f = frame(c);
  const [i0, i1] = visibleIdx();
  const xs = DATES.slice(i0, i1 + 1);
  const rebase = P.daily.creturn[i0];
  const ys = P.daily.creturn.slice(i0, i1 + 1).map(v => v / rebase);
  let bs = null;
  if (P.daily.benchmark) {
    const b0 = P.daily.benchmark.slice(i0, i1 + 1);
    const base = b0.find(v => v != null);
    if (base != null) bs = b0.map(v => (v == null ? null : v / base));
  }
  const log = state.scale === "log";
  const tr = log ? Math.log : (v) => v;
  const all = ys.concat(bs ? bs.filter(v => v != null) : []);
  let lo = Math.min(...all), hi = Math.max(...all);
  if (hi === lo) { hi += 0.01; lo -= 0.01; }
  const pad = (tr(hi) - tr(lo)) * 0.06 || 0.01;
  const y0 = tr(lo) - pad, y1 = tr(hi) + pad;
  const t0 = xs[0], t1 = xs[xs.length - 1];
  const x = (t) => f.m.l + (t - t0) / (t1 - t0 || 1) * f.iw;
  const y = (v) => f.m.t + (1 - (tr(v) - y0) / (y1 - y0)) * f.ih;

  const ticks = log ? logTicks(lo, hi) : niceTicks(lo, hi, 5);
  for (const tv of ticks) {
    el("line", { x1: f.m.l, x2: f.m.l + f.iw, y1: y(tv), y2: y(tv), stroke: css("--grid") }, f.svg);
    el("text", { x: f.m.l - 8, y: y(tv) + 4, "text-anchor": "end", fill: css("--muted"), "font-size": 11 }, f.svg)
      .textContent = fmtSignPct(tv - 1, Math.abs(tv - 1) < 0.1 ? 1 : 0);
  }
  drawXAxis(f, x, t0, t1);

  const path = (vals) => {
    let d = "", pen = false;
    for (let i = 0; i < xs.length; i++) {
      const v = vals[i];
      if (v == null) { pen = false; continue; }
      d += (pen ? "L" : "M") + x(xs[i]).toFixed(1) + "," + y(v).toFixed(1);
      pen = true;
    }
    return d;
  };
  if (bs) {
    el("path", { d: path(bs), fill: "none", stroke: css("--s2"), "stroke-width": 2 }, f.svg);
    const lastB = bs.filter(v => v != null).slice(-1)[0];
    el("text", { x: f.m.l + f.iw + 6, y: y(lastB) + 4, fill: css("--s2"), "font-size": 11, "font-weight": 600 }, f.svg)
      .textContent = "benchmark";
  }
  el("path", { d: path(ys), fill: "none", stroke: css("--s1"), "stroke-width": 2 }, f.svg);
  el("text", { x: f.m.l + f.iw + 6, y: y(ys[ys.length - 1]) + 4, fill: css("--s1"), "font-size": 11, "font-weight": 600 }, f.svg)
    .textContent = "strategy";

  attachCrosshair(c, f, xs, x, (i, px, py) => {
    const dot = (col) => '<span class="dot" style="background:' + col + '"></span>';
    let html = '<div class="t">' + fmtDate(xs[i]) + "</div>"
      + '<div class="r"><span>' + dot(css("--s1")) + "strategy</span><b>" + fmtSignPct(ys[i] - 1) + "</b></div>";
    if (bs && bs[i] != null)
      html += '<div class="r"><span>' + dot(css("--s2")) + "benchmark</span><b>" + fmtSignPct(bs[i] - 1) + "</b></div>";
    showTip(html, px, py);
    return [y(ys[i])].concat(bs && bs[i] != null ? [y(bs[i])] : []);
  }, [css("--s1"), css("--s2")]);
}

/* ---------- drawdown ---------- */
function renderDrawdown() {
  const c = $("drawdown"), f = frame(c);
  const [i0, i1] = visibleIdx();
  const xs = DATES.slice(i0, i1 + 1);
  /* recompute drawdown inside the visible window so it matches the rebased curve */
  const cr = P.daily.creturn.slice(i0, i1 + 1);
  let peak = -Infinity;
  const dd = cr.map(v => { peak = Math.max(peak, v); return v / peak - 1; });
  const lo = Math.min(...dd, -0.001);
  const t0 = xs[0], t1 = xs[xs.length - 1];
  const x = (t) => f.m.l + (t - t0) / (t1 - t0 || 1) * f.iw;
  const y = (v) => f.m.t + (v / lo) * f.ih;
  for (const tv of niceTicks(lo, 0, 3)) {
    el("line", { x1: f.m.l, x2: f.m.l + f.iw, y1: y(tv), y2: y(tv), stroke: css("--grid") }, f.svg);
    el("text", { x: f.m.l - 8, y: y(tv) + 4, "text-anchor": "end", fill: css("--muted"), "font-size": 11 }, f.svg)
      .textContent = fmtPct(tv, 0);
  }
  drawXAxis(f, x, t0, t1);
  let d = "M" + x(xs[0]).toFixed(1) + "," + y(0).toFixed(1);
  for (let i = 0; i < xs.length; i++) d += "L" + x(xs[i]).toFixed(1) + "," + y(dd[i]).toFixed(1);
  d += "L" + x(xs[xs.length - 1]).toFixed(1) + "," + y(0).toFixed(1) + "Z";
  el("path", { d, fill: css("--neg"), "fill-opacity": 0.18, stroke: css("--neg"), "stroke-width": 1.5 }, f.svg);
  attachCrosshair(c, f, xs, x, (i, px, py) => {
    showTip('<div class="t">' + fmtDate(xs[i]) + '</div><div class="r"><span>drawdown</span><b>'
      + fmtPct(dd[i]) + "</b></div>", px, py);
    return [y(dd[i])];
  }, [css("--neg")]);
}

function attachCrosshair(container, f, xs, x, onMove, dotColors) {
  const layer = el("g", {}, f.svg);
  const rect = el("rect", { x: f.m.l, y: f.m.t, width: f.iw, height: f.ih, fill: "transparent" }, f.svg);
  rect.addEventListener("mousemove", (ev) => {
    const bb = f.svg.getBoundingClientRect();
    const t = (ev.clientX - bb.left - f.m.l) / f.iw * (xs[xs.length - 1] - xs[0]) + xs[0];
    const i = bisect(xs, t);
    layer.innerHTML = "";
    el("line", { x1: x(xs[i]), x2: x(xs[i]), y1: f.m.t, y2: f.m.t + f.ih, stroke: css("--axis"), "stroke-dasharray": "3,3" }, layer);
    const ys = onMove(i, ev.clientX, ev.clientY);
    ys.forEach((py, k) => el("circle", {
      cx: x(xs[i]), cy: py, r: 4, fill: dotColors[k] || dotColors[0],
      stroke: css("--surface-1"), "stroke-width": 2,
    }, layer));
  });
  rect.addEventListener("mouseleave", () => { layer.innerHTML = ""; hideTip(); });
}

/* ---------- monthly heatmap ---------- */
function renderHeatmap() {
  if (!P.return_table || !P.return_table.length) { $("heatmap-card").style.display = "none"; return; }
  const rows = P.return_table;
  const maxAbs = Math.max(0.01, ...rows.flatMap(r => r.months.filter(v => v != null).map(Math.abs)));
  const cell = (v, extraCls) => {
    if (v == null) return '<td class="' + (extraCls || "") + '"></td>';
    const t = Math.min(1, Math.abs(v) / maxAbs);
    const bg = mixToward(css("--div-mid").replace(/\s/g, ""), (v >= 0 ? css("--div-pos") : css("--div-neg")).replace(/\s/g, ""), t);
    const ink = t > 0.55 ? "#ffffff" : css("--ink");
    return '<td class="' + (extraCls || "") + '" style="background:' + bg + ";color:" + ink + '" data-v="' + v + '">'
      + (v * 100).toFixed(1) + "</td>";
  };
  let html = "<table><thead><tr><th></th>";
  for (let m = 1; m <= 12; m++) html += "<th>" + m + "</th>";
  html += "<th>Year</th></tr></thead><tbody>";
  for (const r of rows) {
    const yr = r.months.reduce((acc, v) => (v == null ? acc : acc * (1 + v)), 1) - 1;
    html += '<tr><td class="y">' + r.year + "</td>" + r.months.map(v => cell(v)).join("") + cell(yr, "tot") + "</tr>";
  }
  $("heatmap").innerHTML = html + "</tbody></table>";
  $("heatmap").querySelectorAll("td[data-v]").forEach(td => {
    td.addEventListener("mousemove", (ev) => showTip(fmtSignPct(+td.dataset.v, 2), ev.clientX, ev.clientY));
    td.addEventListener("mouseleave", hideTip);
  });
}

/* ---------- trade charts ---------- */
function renderHist() {
  const T = P.trades;
  if (!T || !T.ret.length) { $("trades-row").style.display = "none"; return; }
  const c = $("hist"), f = frame(c);
  f.m.r = 16; f.iw = f.W - f.m.l - f.m.r;
  const rets = T.ret.filter(v => v != null);
  const lo = Math.min(...rets), hi = Math.max(...rets);
  const nb = Math.min(40, Math.max(10, Math.round(Math.sqrt(rets.length) * 1.5)));
  const w = (hi - lo) / nb || 1e-9;
  const bins = Array.from({ length: nb }, () => 0);
  for (const v of rets) bins[Math.min(nb - 1, Math.floor((v - lo) / w))]++;
  const ymax = Math.max(...bins);
  const x = (v) => f.m.l + (v - lo) / (hi - lo || 1) * f.iw;
  const y = (n) => f.m.t + (1 - n / ymax) * f.ih;
  for (const tv of niceTicks(0, ymax, 4)) {
    el("line", { x1: f.m.l, x2: f.m.l + f.iw, y1: y(tv), y2: y(tv), stroke: css("--grid") }, f.svg);
    el("text", { x: f.m.l - 8, y: y(tv) + 4, "text-anchor": "end", fill: css("--muted"), "font-size": 11 }, f.svg).textContent = tv;
  }
  for (const tv of niceTicks(lo, hi, 6))
    el("text", { x: x(tv), y: f.H - 8, "text-anchor": "middle", fill: css("--muted"), "font-size": 11 }, f.svg)
      .textContent = fmtPct(tv, 0);
  el("line", { x1: f.m.l, x2: f.m.l + f.iw, y1: f.m.t + f.ih, y2: f.m.t + f.ih, stroke: css("--axis") }, f.svg);
  if (lo < 0 && hi > 0)
    el("line", { x1: x(0), x2: x(0), y1: f.m.t, y2: f.m.t + f.ih, stroke: css("--axis"), "stroke-dasharray": "3,3" }, f.svg);
  bins.forEach((n, i) => {
    if (!n) return;
    const vlo = lo + i * w;
    const bar = el("rect", {
      x: x(vlo) + 1, y: y(n), width: Math.max(1, x(vlo + w) - x(vlo) - 2), height: f.m.t + f.ih - y(n),
      rx: 2, fill: vlo + w / 2 >= 0 ? css("--div-pos") : css("--div-neg"), "fill-opacity": 0.85,
    }, f.svg);
    bar.addEventListener("mousemove", (ev) => showTip(
      "<b>" + n + "</b> trades in " + fmtPct(vlo, 1) + " … " + fmtPct(vlo + w, 1), ev.clientX, ev.clientY));
    bar.addEventListener("mouseleave", hideTip);
  });
}
function renderScatter() {
  const T = P.trades;
  if (!T || !T.ret.length) return;
  const c = $("scatter"), f = frame(c);
  f.m.r = 16; f.iw = f.W - f.m.l - f.m.r;
  const pts = [];
  for (let i = 0; i < T.ret.length; i++)
    if (T.ret[i] != null && T.mae[i] != null) pts.push(i);
  const xlo = Math.min(0, ...pts.map(i => T.mae[i])), xhi = Math.max(0.001, ...pts.map(i => T.mae[i]));
  const ylo = Math.min(...pts.map(i => T.ret[i])), yhi = Math.max(...pts.map(i => T.ret[i]));
  const x = (v) => f.m.l + (v - xlo) / (xhi - xlo || 1) * f.iw;
  const y = (v) => f.m.t + (1 - (v - ylo) / (yhi - ylo || 1)) * f.ih;
  for (const tv of niceTicks(ylo, yhi, 5)) {
    el("line", { x1: f.m.l, x2: f.m.l + f.iw, y1: y(tv), y2: y(tv), stroke: css("--grid") }, f.svg);
    el("text", { x: f.m.l - 8, y: y(tv) + 4, "text-anchor": "end", fill: css("--muted"), "font-size": 11 }, f.svg)
      .textContent = fmtPct(tv, 0);
  }
  for (const tv of niceTicks(xlo, xhi, 6))
    el("text", { x: x(tv), y: f.H - 8, "text-anchor": "middle", fill: css("--muted"), "font-size": 11 }, f.svg)
      .textContent = fmtPct(tv, 0);
  el("line", { x1: f.m.l, x2: f.m.l + f.iw, y1: f.m.t + f.ih, y2: f.m.t + f.ih, stroke: css("--axis") }, f.svg);
  if (ylo < 0 && yhi > 0)
    el("line", { x1: f.m.l, x2: f.m.l + f.iw, y1: y(0), y2: y(0), stroke: css("--axis"), "stroke-dasharray": "3,3" }, f.svg);
  for (const i of pts) {
    const dot = el("circle", {
      cx: x(T.mae[i]), cy: y(T.ret[i]), r: 3.5,
      fill: css("--s1"), "fill-opacity": 0.5, stroke: css("--surface-1"), "stroke-width": 0.5,
    }, f.svg);
    dot.addEventListener("mousemove", (ev) => showTip(
      '<div class="t">' + T.stock[i] + " · " + T.entry[i] + " → " + (T.exit[i] || "open") + "</div>"
      + '<div class="r"><span>return</span><b>' + fmtSignPct(T.ret[i]) + "</b></div>"
      + '<div class="r"><span>MAE</span><b>' + fmtPct(T.mae[i]) + "</b></div>"
      + '<div class="r"><span>held</span><b>' + T.pdays[i] + "d</b></div>", ev.clientX, ev.clientY));
    dot.addEventListener("mouseleave", hideTip);
  }
  el("text", { x: f.m.l + f.iw, y: f.H - 8, "text-anchor": "end", fill: css("--muted"), "font-size": 10 }, f.svg)
    .textContent = T.sampled ? "sampled " + T.ret.length + " of " + T.total + " trades" : "";
}

/* ---------- stats tables ---------- */
function renderStats() {
  const groups = P.stat_groups;
  $("stats-cols").innerHTML = groups.map(g =>
    '<table class="stats">' + g.map(([k, v]) => "<tr><td>" + k + "</td><td>" + v + "</td></tr>").join("") + "</table>"
  ).join("");
}

/* ---------- controls ---------- */
function renderControls() {
  $("range-seg").innerHTML = RANGES.map(([k, lbl]) =>
    '<button data-k="' + k + '"' + (state.range === k ? ' class="on"' : "") + ">" + lbl + "</button>").join("");
  $("scale-seg").innerHTML = [["linear", "Linear"], ["log", "Log"]].map(([k, lbl]) =>
    '<button data-k="' + k + '"' + (state.scale === k ? ' class="on"' : "") + ">" + lbl + "</button>").join("");
  $("range-seg").querySelectorAll("button").forEach(b =>
    b.addEventListener("click", () => { state.range = b.dataset.k; renderControls(); renderEquity(); renderDrawdown(); }));
  $("scale-seg").querySelectorAll("button").forEach(b =>
    b.addEventListener("click", () => { state.scale = b.dataset.k; renderControls(); renderEquity(); }));
}

function renderAll() {
  renderHeader(); renderTiles(); renderControls();
  renderEquity(); renderDrawdown(); renderHeatmap(); renderHist(); renderScatter(); renderStats();
}
renderAll();
let rsTimer = null;
addEventListener("resize", () => { clearTimeout(rsTimer); rsTimer = setTimeout(() => { renderEquity(); renderDrawdown(); renderHist(); renderScatter(); }, 150); });
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", renderAll);
</script>
</body>
</html>
"""
