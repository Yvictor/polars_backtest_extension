# ruff: noqa: E501  (the template string embeds CSS/JS with long lines)
"""Self-contained HTML template for the interactive backtest report.

The template has no external assets (no CDN, no plotly): charts are inline SVG
rendered by vanilla JS, so the output file works offline and in notebook
iframes. Placeholders: __TITLE__, __PAYLOAD__, __VERSION__.

Layout follows specs/VIZ_V2_FINLAB_PARITY_SPEC.md: hero KPI bar, five quality
score chips (獲利/風險/報酬比/勝率/流動性) with a metric tile row, and
per-dimension panel tabs (歷史績效/月報酬/年度比較/交易/虧損歷史/報酬分布/流動性).
"""

TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-Hant">
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
  --up:          #006300;  /* positive-return text */
  --dn:          #b73634;  /* negative-return text */
  --good:        #0ca30c;  /* status ramp (score >= 70) */
  --warn:        #fab219;  /* status ramp (40-69) */
  --bad:         #d03b3b;  /* status ramp (< 40) */
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
    --up:          #0ca30c;
    --dn:          #e66767;
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
.info { color: var(--muted); font-size: 11px; cursor: help; font-weight: 400; }
/* hero KPI bar */
.hero { display: grid; grid-template-columns: 1.5fr 1fr 1fr 1fr; gap: 10px; margin-bottom: 12px; }
@media (max-width: 720px) { .hero { grid-template-columns: 1fr 1fr; } }
.tile { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 10px 12px; }
.tile .k { color: var(--muted); font-size: 12px; }
.tile .v { font-size: 22px; font-weight: 650; margin-top: 2px; }
.tile .v.xl { font-size: 34px; }
.tile .v.up { color: var(--up); } .tile .v.dn { color: var(--dn); }
.tile .c { color: var(--muted); font-size: 11px; margin-top: 2px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; }
/* score chips */
.qchips { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 12px; }
.qchip { display: flex; align-items: center; gap: 8px; border: 1px solid var(--border);
         background: var(--surface-1); border-radius: 12px; padding: 6px 14px 6px 8px;
         cursor: pointer; font: inherit; color: var(--ink); }
.qchip.on { border-color: var(--s1); box-shadow: 0 0 0 1px var(--s1); }
.qchip .ql { font-size: 13px; font-weight: 600; text-align: left; line-height: 1.25; }
.qchip .qna { font-size: 11px; color: var(--muted); font-weight: 400; }
.ring.good { stroke: var(--good); } .ring.warn { stroke: var(--warn); } .ring.bad { stroke: var(--bad); }
/* metric tile row */
.mrow { display: grid; grid-template-columns: repeat(auto-fit, minmax(158px, 1fr)); gap: 10px; margin-bottom: 14px; }
.mtile { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 8px 12px; }
.mtile .mk { color: var(--ink-2); font-size: 12px; display: flex; align-items: center; gap: 6px; }
.mtile .mv { font-size: 18px; font-weight: 650; margin-top: 2px; }
.mtile .mc { color: var(--muted); font-size: 11px; margin-top: 1px; }
.dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; flex: none; }
.d-ok { background: var(--good); } .d-no { background: var(--bad); } .d-na { background: var(--axis); }
/* tabs + panels */
.tabbar { display: flex; gap: 4px; border-bottom: 1px solid var(--grid); margin-bottom: 14px; flex-wrap: wrap; }
.tabbar button { border: 0; background: transparent; color: var(--ink-2); font: inherit; font-size: 13px;
                 padding: 7px 14px; cursor: pointer; border-bottom: 2px solid transparent; margin-bottom: -1px; }
.tabbar button.on { color: var(--ink); font-weight: 600; border-bottom-color: var(--s1); }
.panel { display: none; } .panel.on { display: block; }
.card { background: var(--surface-1); border: 1px solid var(--border); border-radius: 12px;
        padding: 14px 16px 12px; margin-bottom: 14px; }
.card-head { display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px; margin-bottom: 4px; }
.controls { display: flex; gap: 8px; }
.seg { display: inline-flex; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
.seg button { border: 0; background: transparent; color: var(--ink-2); padding: 3px 10px; font-size: 12px;
              cursor: pointer; font-family: inherit; }
.seg button.on { background: var(--s1); color: #fff; }
.legend { display: flex; gap: 14px; font-size: 12px; color: var(--ink-2); align-items: center; }
.legend .dot { margin-right: 5px; }
.kpis { display: flex; gap: 16px; font-size: 12.5px; color: var(--ink-2); flex-wrap: wrap; }
.kpis b { color: var(--ink); font-variant-numeric: tabular-nums; }
.chart { position: relative; width: 100%; }
.chart svg { display: block; width: 100%; }
.chart.tall { height: 330px; } .chart.short { height: 160px; } .chart.mid { height: 260px; }
.row2 { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
@media (max-width: 800px) { .row2 { grid-template-columns: 1fr; } }
#tooltip { position: fixed; pointer-events: none; z-index: 10; display: none;
           background: var(--tip-bg); color: var(--tip-ink); border-radius: 8px;
           padding: 7px 10px; font-size: 12px; max-width: 300px;
           box-shadow: 0 4px 16px rgba(0,0,0,0.25); }
#tooltip .t { color: color-mix(in srgb, var(--tip-ink) 62%, transparent); margin-bottom: 2px; }
#tooltip .r { display: flex; gap: 10px; justify-content: space-between; }
#tooltip .dot { margin-right: 5px; }
/* yearly chips */
.ychips { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
.ychip { border: 1px solid var(--border); border-radius: 8px; padding: 2px 8px; font-size: 12px;
         cursor: pointer; font-family: inherit; color: var(--ink-2); background: transparent; }
.ychip b { font-variant-numeric: tabular-nums; font-weight: 600; }
.ychip.up b { color: var(--up); } .ychip.dn b { color: var(--dn); }
.ychip.on { border-color: var(--s1); box-shadow: 0 0 0 1px var(--s1); color: var(--ink); }
/* heatmap */
.hm { overflow-x: auto; }
.hm table { border-collapse: separate; border-spacing: 2px; width: 100%; }
.hm th { font-size: 11px; color: var(--muted); font-weight: 500; padding: 2px 4px; text-align: center; }
.hm td { font-size: 11px; text-align: center; padding: 4px 2px; border-radius: 4px; min-width: 44px;
         font-variant-numeric: tabular-nums; }
.hm td.y { color: var(--ink-2); font-weight: 600; background: transparent; }
.hm td.tot { font-weight: 650; }
/* trade table */
table.tt { width: 100%; border-collapse: collapse; font-size: 12.5px; }
table.tt th { text-align: left; color: var(--muted); font-weight: 500; font-size: 12px;
              padding: 6px 8px; border-bottom: 1px solid var(--axis); white-space: nowrap; }
table.tt th.sortable { cursor: pointer; }
table.tt td { padding: 6px 8px; border-bottom: 1px solid var(--grid); vertical-align: top;
              font-variant-numeric: tabular-nums; }
.tsub { color: var(--muted); font-size: 11px; }
.badge { display: inline-block; border-radius: 6px; padding: 1px 7px; font-size: 12px; font-weight: 600;
         font-variant-numeric: tabular-nums; }
.badge.up { color: var(--up); background: color-mix(in srgb, var(--up) 12%, transparent); }
.badge.dn { color: var(--dn); background: color-mix(in srgb, var(--dn) 12%, transparent); }
.flag { display: inline-block; font-size: 11px; border: 1px solid color-mix(in srgb, var(--dn) 45%, transparent);
        color: var(--dn); border-radius: 5px; padding: 0 5px; margin-right: 4px; white-space: nowrap; }
.pager { display: flex; gap: 8px; align-items: center; justify-content: flex-end;
         color: var(--muted); font-size: 12px; margin-top: 8px; }
.pager button { border: 1px solid var(--border); background: transparent; color: var(--ink-2);
                border-radius: 6px; padding: 2px 10px; cursor: pointer; font: inherit; font-size: 12px; }
.pager button:disabled { opacity: 0.4; cursor: default; }
/* drawdown rank list */
.ddrow { display: grid; grid-template-columns: 72px 1fr 72px 84px; align-items: center; gap: 8px;
         width: 100%; border: 0; background: transparent; padding: 4px 6px; cursor: pointer;
         font: inherit; color: var(--ink); border-radius: 6px; text-align: left; font-size: 12.5px; }
.ddrow:hover { background: color-mix(in srgb, var(--ink) 4%, transparent); }
.ddrow.on { background: color-mix(in srgb, var(--s1) 10%, transparent); }
.ddrow .ddbar i { display: block; height: 10px; border-radius: 3px; }
.ddrow .ddv { text-align: right; font-variant-numeric: tabular-nums; font-weight: 600; }
.ddrow .ddd { color: var(--muted); font-size: 11px; text-align: right; font-variant-numeric: tabular-nums; }
/* small stat tables */
table.stats { width: 100%; border-collapse: collapse; }
table.stats th { text-align: left; color: var(--muted); font-weight: 500; font-size: 12px; padding: 5px 8px;
                 border-bottom: 1px solid var(--axis); }
table.stats td { padding: 5px 8px; border-bottom: 1px solid var(--grid); font-variant-numeric: tabular-nums; }
table.stats td:first-child { color: var(--ink-2); }
table.stats.kv td:last-child { text-align: right; font-weight: 550; }
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
  <section class="hero" id="hero"></section>
  <section class="qchips" id="qchips"></section>
  <section class="mrow" id="mrow"></section>
  <nav class="tabbar" id="tabbar"></nav>

  <section class="panel" id="panel-perf">
    <div class="card">
      <div class="card-head">
        <h2>歷史績效 <span class="info" data-tip="策略與大盤的累積報酬曲線，可切換線性/對數與時間區間；下方為同區間的策略回檔">&#9432;</span></h2>
        <div class="legend" id="eq-legend"></div>
        <div class="controls">
          <div class="seg" id="range-seg"></div>
          <div class="seg" id="scale-seg"></div>
        </div>
      </div>
      <div id="equity" class="chart tall"></div>
      <div id="drawdown" class="chart short"></div>
      <div id="ychips" class="ychips"></div>
    </div>
  </section>

  <section class="panel" id="panel-monthly">
    <div class="card">
      <div class="card-head">
        <h2>月報酬 (%) <span class="info" data-tip="每月報酬熱力圖，顏色深度代表漲跌幅；最右欄為年度合計">&#9432;</span></h2>
        <div class="kpis" id="mstats"></div>
      </div>
      <div id="heatmap" class="hm"></div>
    </div>
  </section>

  <section class="panel" id="panel-yearly">
    <div class="card">
      <div class="card-head">
        <h2>年度比較 <span class="info" data-tip="策略與大盤的逐年報酬（每年重設基準），比較每年相對表現">&#9432;</span></h2>
        <div class="legend" id="yr-legend"></div>
        <div class="kpis" id="yr-stats"></div>
      </div>
      <div id="yearly-chart" class="chart mid"></div>
    </div>
  </section>

  <section class="panel" id="panel-trades">
    <div class="card">
      <div class="card-head">
        <h2>交易明細 <span class="info" data-tip="全部交易紀錄：報酬、進出場價、持倉比重、MAE/GMFE 與漲跌停旗標；點欄位標題可排序">&#9432;</span></h2>
        <div class="kpis" id="tr-stats"></div>
      </div>
      <div id="ttable-wrap" style="overflow-x:auto"></div>
      <div class="pager" id="tpager"></div>
    </div>
  </section>

  <section class="panel" id="panel-dd">
    <div class="card">
      <div class="card-head">
        <h2>虧損歷史 <span class="info" data-tip="全期間回檔深度（策略 vs 大盤）；點下方排名可在圖上標示該回檔區間">&#9432;</span></h2>
        <div class="legend" id="dd-legend"></div>
        <div class="controls"><div class="seg" id="dd-seg"></div></div>
      </div>
      <div class="kpis" id="dd-caption" style="margin-bottom:6px"></div>
      <div id="dd-chart" class="chart mid"></div>
      <h2 style="margin-top:12px">跌幅排名 <span class="info" data-tip="歷史回檔事件由深至淺排序：低谷年月、最大跌幅與總天數（高點到回復）">&#9432;</span></h2>
      <div id="dd-rank"></div>
    </div>
  </section>

  <section class="panel" id="panel-dist">
    <div class="row2">
      <div class="card">
        <h2>交易報酬分布 <span class="info" data-tip="每筆交易報酬的直方圖：綠為獲利、紅為虧損">&#9432;</span></h2>
        <div id="hist" class="chart mid"></div>
        <div class="sub" id="dist-note" style="margin-top:6px"></div>
      </div>
      <div class="card">
        <h2>報酬與最大不利偏移 (MAE) <span class="info" data-tip="x 軸為交易期間最大浮虧 (MAE)、y 軸為最終報酬；左上代表曾大幅浮虧仍獲利">&#9432;</span></h2>
        <div id="scatter" class="chart mid"></div>
      </div>
    </div>
    <div class="card">
      <h2>模擬停損 <span class="info" data-tip="以各筆交易的 MAE 模擬固定停損：MAE 低於停損線的交易以停損價出場（忽略費用）">&#9432;</span></h2>
      <div id="stops"></div>
    </div>
  </section>

  <section class="panel" id="panel-liq">
    <div class="card">
      <h2>流動性 <span class="info" data-tip="胃納量與漲跌停成交風險：買在漲停可能買不到、賣在跌停可能賣不掉">&#9432;</span></h2>
      <div class="tiles" id="liq-tiles" style="margin-bottom:12px"></div>
      <div id="liq-list"></div>
    </div>
  </section>

  <section class="card" id="stats-card">
    <h2>統計摘要</h2>
    <div class="cols3" id="stats-cols"></div>
  </section>
  <footer id="rpt-footer"></footer>
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
function fmtMetric(v, fmt) {
  if (v == null) return "–";
  switch (fmt) {
    case "pct": return fmtSignPct(v, 1);
    case "num": return (+v).toFixed(2);
    case "int": return (Math.round(v * 10) / 10) + " 檔";
    case "days": return (Math.round(v * 10) / 10) + " 天";
    case "wan": return Math.round(v / 1e4).toLocaleString("en-US") + " 萬";
  }
  return String(v);
}
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
function legendHtml(items) {
  return items.map(([c, l]) => '<span><span class="dot" style="background:' + c + '"></span>' + l + "</span>").join("");
}

/* ---------- zh-TW descriptions (spec §3) ---------- */
const DESC = {
  annualReturn: "年度回報：策略的年化報酬率 (CAGR)",
  alpha: "Alpha：相對於基準、經風險調整後的超額表現（需設定大盤）",
  beta: "Beta：策略對市場變動的敏感性（需設定大盤）",
  avgNStock: "平均持有：投資組合中持有的平均股票數",
  maxNStock: "最多持有：投資組合中持有的最大股票數",
  maxDrawdown: "最大回檔：從高點到低谷的最大百分比下降",
  avgDrawdown: "平均回檔幅度：各回檔事件的平均跌幅",
  avgDrawdownDays: "平均回檔時間：各回檔事件的平均天數",
  valueAtRisk: "Value at Risk：月報酬 5% 分位數，給定信心水準下的預期最大損失",
  cvalueAtRisk: "Conditional VaR：低於 VaR 的月報酬平均，尾部事件後的預期損失",
  volatility: "波動性：策略日報酬的年化標準差",
  sharpeRatio: "夏普值：風險調整後表現（無風險利率 2%）",
  sortinoRatio: "Sortino Ratio：只計下行波動的風險調整表現",
  calmarRatio: "Calmar Ratio：年度回報與最大回檔的比率",
  profitFactor: "Profit Factor：總獲利與總虧損的比率",
  tailRatio: "Tail Ratio：右尾（贏）與左尾（輸）的比率",
  winRate: "逐筆交易勝率：獲利交易的比例",
  m12WinRate: "12個月勝大盤：滾動 12 個月贏過大盤的比例（需設定大盤）",
  expectancy: "期望值：每筆交易的平均報酬",
  mae: "最大不利偏移：交易獲利前的平均最大浮虧",
  mfe: "最大有利偏移：交易轉虧前的平均最大浮盈",
  capacity: "胃納量：不影響市場價格可部署的最大資金（需 trading_value 欄位）",
  buyHigh: "買在漲停：進場即漲停的比例，過高代表可能買不到",
  sellLow: "賣在跌停：出場即跌停的比例，過高代表可能賣不掉",
  disposalStockRatio: "處置股：交易處置股的比例（需外部旗標資料）",
  warningStockRatio: "警示股：交易警示股的比例（需外部旗標資料）",
  fullDeliveryStockRatio: "全額交割股：交易全額交割股的比例（需外部旗標資料）",
};

/* ---------- state ---------- */
const state = {
  dim: "profitability", tab: "perf", range: "all", scale: "linear",
  tsort: "entry", tdir: -1, tpage: 0, ddSide: "strat", ddSel: -1,
};
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
  let t0, t1 = LAST;
  if (state.range.startsWith("y:")) {
    const y = +state.range.slice(2);
    t0 = Date.UTC(y, 0, 1); t1 = Date.UTC(y, 11, 31);
  } else t0 = rangeStart();
  let i0 = DATES.findIndex(t => t >= t0);
  if (i0 < 0) i0 = 0;
  let i1 = DATES.length - 1;
  while (i1 > i0 && DATES[i1] > t1) i1--;
  return [i0, i1];
}
/* per-year returns from a cumulative series (rebased at each year boundary) */
function yearlyReturns(vals) {
  const byYear = new Map();
  for (let i = 0; i < DATES.length; i++) {
    const y = new Date(DATES[i]).getUTCFullYear();
    if (!byYear.has(y)) byYear.set(y, [i, i]); else byYear.get(y)[1] = i;
  }
  const rows = [];
  let prevLast = null;
  for (const y of [...byYear.keys()].sort()) {
    const [a, b] = byYear.get(y);
    let lv = null;
    for (let i = b; i >= a; i--) if (vals[i] != null) { lv = vals[i]; break; }
    let base = prevLast;
    if (base == null) for (let i = a; i <= b; i++) if (vals[i] != null) { base = vals[i]; break; }
    rows.push({ y, ret: lv != null && base ? lv / base - 1 : null });
    if (lv != null) prevLast = lv;
  }
  return rows;
}
function drawdownOf(vals) {
  let peak = null;
  return vals.map(v => {
    if (v == null) return null;
    peak = peak == null ? v : Math.max(peak, v);
    return v / peak - 1;
  });
}

/* ---------- header / hero / quality ---------- */
function renderHeader() {
  $("rpt-title").textContent = P.title;
  $("rpt-range").textContent = P.daily.dates[0] + " → " + P.daily.dates[P.daily.dates.length - 1]
    + "  ·  " + P.daily.dates.length + " trading days";
  $("chips").innerHTML = (P.params || []).map(p => '<span class="chip">' + p + "</span>").join("");
  $("rpt-footer").textContent = "generated by polars-backtest __VERSION__ · schema " + (P.schema || "?")
    + " · 品質檢核採 corrected 模式（無資料項不列入計分）· 無風險利率 2%/年";
}
function renderHero() {
  const m = P.metrics || {}, s = P.stats || {}, t = P.trade_summary || {};
  const ar = m.annualReturn != null ? m.annualReturn : s.cagr;
  const md = m.maxDrawdown != null ? m.maxDrawdown : s.max_drawdown;
  const sh = m.sharpeRatio != null ? m.sharpeRatio : s.daily_sharpe;
  const wr = m.winRate != null ? m.winRate : t.win_rate;
  const items = [
    ["年度回報", fmtSignPct(ar, 1), "xl " + (ar != null && ar < 0 ? "dn" : "up"), DESC.annualReturn],
    ["最大回檔", fmtPct(md, 1), "dn", DESC.maxDrawdown],
    ["夏普值", fmtNum(sh, 2), "", DESC.sharpeRatio],
    ["逐筆交易勝率", fmtPct(wr, 1), "", DESC.winRate],
  ];
  $("hero").innerHTML = items.map(([k, v, cls, tip]) =>
    '<div class="tile" data-tip="' + tip + '"><div class="k">' + k + ' <span class="info">&#9432;</span></div>'
    + '<div class="v ' + cls + '">' + v + "</div></div>").join("");
}
const DIMS = ["profitability", "risk", "ratio", "winrate", "liquidity"];
function renderQuality() {
  const q = P.quality || {};
  const C = 2 * Math.PI * 17;
  $("qchips").innerHTML = DIMS.map(d => {
    const info = q[d] || {}, s = info.score;
    const cls = s == null ? "na" : s >= 70 ? "good" : s >= 40 ? "warn" : "bad";
    let ring = '<circle cx="22" cy="22" r="17" fill="none" stroke="var(--grid)" stroke-width="4"></circle>';
    if (s != null && s > 0)
      ring += '<circle cx="22" cy="22" r="17" fill="none" class="ring ' + cls + '" stroke-width="4" stroke-linecap="round"'
        + ' stroke-dasharray="' + (C * s / 100).toFixed(1) + " " + C.toFixed(1) + '" transform="rotate(-90 22 22)"></circle>';
    return '<button class="qchip' + (state.dim === d ? " on" : "") + '" data-dim="' + d + '">'
      + '<svg viewBox="0 0 44 44" width="44" height="44">' + ring
      + '<text x="22" y="26" text-anchor="middle" font-size="13" font-weight="650" fill="var(--ink)">' + (s == null ? "–" : s) + "</text></svg>"
      + '<span class="ql">' + (info.label || d) + (s == null ? '<br><span class="qna">無資料</span>' : "") + "</span></button>";
  }).join("");
  $("qchips").querySelectorAll(".qchip").forEach(b => b.addEventListener("click", () => {
    state.dim = b.dataset.dim;
    state.tab = availableTabs()[0];
    renderQuality(); renderTabs(); showPanel();
  }));
  renderMetricRow();
}
function renderMetricRow() {
  const q = P.quality || {};
  const checks = (q[state.dim] && q[state.dim].checks) || [];
  $("mrow").innerHTML = checks.map(c => {
    const d = c.pass == null ? "na" : c.pass ? "ok" : "no";
    return '<div class="mtile" data-tip="' + (DESC[c.key] || c.label) + '">'
      + '<div class="mk"><span class="dot d-' + d + '"></span>' + c.label + ' <span class="info">&#9432;</span></div>'
      + '<div class="mv">' + fmtMetric(c.value, c.fmt) + "</div>"
      + '<div class="mc">' + (c.value == null ? "無資料 · " : "") + (c.caption || "") + "</div></div>";
  }).join("");
}

/* ---------- tabs ---------- */
const DIM_TABS = {
  profitability: ["perf", "monthly", "yearly", "trades"],
  risk: ["dd", "trades"],
  ratio: ["perf", "dist"],
  winrate: ["dist", "trades"],
  liquidity: ["liq", "trades"],
};
const TAB_LABELS = { perf: "歷史績效", monthly: "月報酬", yearly: "年度比較", trades: "交易",
                     dd: "虧損歷史", dist: "報酬分布", liq: "流動性" };
function tabAvailable(t) {
  const hasTrades = P.trades && P.trades.ret && P.trades.ret.length;
  if (t === "monthly") return P.return_table && P.return_table.length;
  if (t === "trades" || t === "dist") return !!hasTrades;
  return true;
}
function availableTabs() { return DIM_TABS[state.dim].filter(tabAvailable); }
function renderTabs() {
  const tabs = availableTabs();
  if (!tabs.includes(state.tab)) state.tab = tabs[0];
  $("tabbar").innerHTML = tabs.map(t =>
    '<button data-t="' + t + '"' + (state.tab === t ? ' class="on"' : "") + ">" + TAB_LABELS[t] + "</button>").join("");
  $("tabbar").querySelectorAll("button").forEach(b => b.addEventListener("click", () => {
    state.tab = b.dataset.t; renderTabs(); showPanel();
  }));
}
const PANEL_RENDER = {
  perf: renderPerfPanel, monthly: renderMonthlyPanel, yearly: renderYearlyPanel,
  trades: renderTradesPanel, dd: renderDDPanel, dist: renderDistPanel, liq: renderLiqPanel,
};
function showPanel() {
  document.querySelectorAll(".panel").forEach(p => p.classList.remove("on"));
  const panel = $("panel-" + state.tab);
  if (panel) panel.classList.add("on");
  (PANEL_RENDER[state.tab] || (() => {}))();
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

/* ---------- 歷史績效 panel ---------- */
function renderPerfPanel() { renderControls(); renderEquity(); renderDrawdown(); renderYearChips(); }
function renderControls() {
  $("range-seg").innerHTML = RANGES.map(([k, lbl]) =>
    '<button data-k="' + k + '"' + (state.range === k ? ' class="on"' : "") + ">" + lbl + "</button>").join("");
  $("scale-seg").innerHTML = [["linear", "Linear"], ["log", "Log"]].map(([k, lbl]) =>
    '<button data-k="' + k + '"' + (state.scale === k ? ' class="on"' : "") + ">" + lbl + "</button>").join("");
  $("range-seg").querySelectorAll("button").forEach(b =>
    b.addEventListener("click", () => { state.range = b.dataset.k; renderPerfPanel(); }));
  $("scale-seg").querySelectorAll("button").forEach(b =>
    b.addEventListener("click", () => { state.scale = b.dataset.k; renderControls(); renderEquity(); }));
  $("eq-legend").innerHTML = P.daily.benchmark
    ? legendHtml([[css("--s1"), "策略"], [css("--s2"), "大盤"]]) : "";
}
function renderEquity() {
  const c = $("equity"), f = frame(c);
  const [i0, i1] = visibleIdx();
  const xs = DATES.slice(i0, i1 + 1);
  if (xs.length < 2) { return; }
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
      .textContent = "大盤";
  }
  el("path", { d: path(ys), fill: "none", stroke: css("--s1"), "stroke-width": 2 }, f.svg);
  el("text", { x: f.m.l + f.iw + 6, y: y(ys[ys.length - 1]) + 4, fill: css("--s1"), "font-size": 11, "font-weight": 600 }, f.svg)
    .textContent = "策略";

  attachCrosshair(c, f, xs, x, (i, px, py) => {
    const dot = (col) => '<span class="dot" style="background:' + col + '"></span>';
    let html = '<div class="t">' + fmtDate(xs[i]) + "</div>"
      + '<div class="r"><span>' + dot(css("--s1")) + "策略</span><b>" + fmtSignPct(ys[i] - 1) + "</b></div>";
    if (bs && bs[i] != null)
      html += '<div class="r"><span>' + dot(css("--s2")) + "大盤</span><b>" + fmtSignPct(bs[i] - 1) + "</b></div>";
    showTip(html, px, py);
    return [y(ys[i])].concat(bs && bs[i] != null ? [y(bs[i])] : []);
  }, [css("--s1"), css("--s2")]);
}
function renderDrawdown() {
  const c = $("drawdown"), f = frame(c);
  const [i0, i1] = visibleIdx();
  const xs = DATES.slice(i0, i1 + 1);
  if (xs.length < 2) { return; }
  /* recompute drawdown inside the visible window so it matches the rebased curve */
  const dd = drawdownOf(P.daily.creturn.slice(i0, i1 + 1));
  const lo = Math.min(...dd.filter(v => v != null), -0.001);
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
  el("text", { x: f.m.l + f.iw + 6, y: f.m.t + 12, fill: css("--muted"), "font-size": 11 }, f.svg).textContent = "回檔";
  attachCrosshair(c, f, xs, x, (i, px, py) => {
    showTip('<div class="t">' + fmtDate(xs[i]) + '</div><div class="r"><span>回檔</span><b>'
      + fmtPct(dd[i]) + "</b></div>", px, py);
    return [y(dd[i])];
  }, [css("--neg")]);
}
function renderYearChips() {
  const rows = yearlyReturns(P.daily.creturn);
  const brows = P.daily.benchmark ? yearlyReturns(P.daily.benchmark) : null;
  const total = P.daily.creturn[P.daily.creturn.length - 1] / P.daily.creturn[0] - 1;
  const chip = (key, label, ret, tip) =>
    '<button class="ychip ' + (ret != null && ret < 0 ? "dn" : "up") + (state.range === key ? " on" : "")
    + '" data-r="' + key + '"' + (tip ? ' data-tip="' + tip + '"' : "") + ">" + label + " <b>" + fmtSignPct(ret, 1) + "</b></button>";
  $("ychips").innerHTML = chip("all", "全部", total)
    + rows.map((r, i) => chip("y:" + r.y, r.y, r.ret,
        brows && brows[i] && brows[i].ret != null ? "大盤 " + fmtSignPct(brows[i].ret, 1) : null)).join("");
  $("ychips").querySelectorAll(".ychip").forEach(b => b.addEventListener("click", () => {
    state.range = b.dataset.r === "all" ? "all" : b.dataset.r;
    renderPerfPanel();
  }));
}

/* ---------- 月報酬 panel ---------- */
function renderMonthlyPanel() {
  if (!P.return_table || !P.return_table.length) return;
  const cells = P.return_table.flatMap(r => r.months.filter(v => v != null));
  if (cells.length) {
    const avg = cells.reduce((a, v) => a + v, 0) / cells.length;
    const win = cells.filter(v => v > 0).length / cells.length;
    $("mstats").innerHTML = "<span>平均月報酬 <b>" + fmtSignPct(avg, 2) + "</b></span>"
      + "<span>月勝率 <b>" + fmtPct(win, 1) + "</b></span>";
  }
  renderHeatmap();
}
function renderHeatmap() {
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
  html += "<th>年度</th></tr></thead><tbody>";
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

/* ---------- 年度比較 panel ---------- */
function renderYearlyPanel() {
  const ys = yearlyReturns(P.daily.creturn);
  const bs = P.daily.benchmark ? yearlyReturns(P.daily.benchmark) : null;
  $("yr-legend").innerHTML = legendHtml(
    bs ? [[css("--s1"), "策略"], [css("--s2"), "大盤"]] : [[css("--s1"), "策略"]]);
  if (bs) {
    const pairs = ys.map((r, i) => [r.ret, bs[i] ? bs[i].ret : null]).filter(([a, b]) => a != null && b != null);
    const wins = pairs.filter(([a, b]) => a > b).length;
    const excess = pairs.length ? pairs.reduce((s, [a, b]) => s + (a - b), 0) / pairs.length : null;
    $("yr-stats").innerHTML = "<span>贏大盤 <b>" + wins + " / " + pairs.length + " 年</b></span>"
      + "<span>平均超額報酬 <b>" + fmtSignPct(excess, 1) + "</b></span>";
  } else $("yr-stats").innerHTML = '<span class="sub">未設定大盤，僅顯示策略年報酬</span>';

  const c = $("yearly-chart"), f = frame(c);
  f.m.r = 16; f.iw = f.W - f.m.l - f.m.r;
  const vals = ys.map(r => r.ret).concat(bs ? bs.map(r => r.ret) : []).filter(v => v != null);
  if (!vals.length) return;
  let lo = Math.min(0, ...vals), hi = Math.max(0, ...vals);
  if (hi === lo) hi += 0.01;
  const padv = (hi - lo) * 0.08;
  lo -= lo < 0 ? padv : 0; hi += padv;
  const ticks = niceTicks(lo, hi, 5);
  if (lo < 0 && ticks.length > 1 && ticks[0] > lo + 1e-9) {
    ticks.unshift(ticks[0] - (ticks[1] - ticks[0]));
    lo = ticks[0];
  }
  const y = (v) => f.m.t + (1 - (v - lo) / (hi - lo)) * f.ih;
  for (const tv of ticks) {
    el("line", { x1: f.m.l, x2: f.m.l + f.iw, y1: y(tv), y2: y(tv), stroke: css("--grid") }, f.svg);
    el("text", { x: f.m.l - 8, y: y(tv) + 4, "text-anchor": "end", fill: css("--muted"), "font-size": 11 }, f.svg)
      .textContent = fmtSignPct(tv, 0);
  }
  el("line", { x1: f.m.l, x2: f.m.l + f.iw, y1: y(0), y2: y(0), stroke: css("--axis") }, f.svg);
  const n = ys.length, bw = f.iw / n, gw = Math.min(bw * 0.6, 64);
  const single = !bs;
  const barw = single ? gw : (gw - 2) / 2;
  ys.forEach((r, i) => {
    const cx = f.m.l + bw * (i + 0.5);
    const bars = single ? [[r.ret, css("--s1"), "策略"]]
      : [[r.ret, css("--s1"), "策略"], [bs[i] ? bs[i].ret : null, css("--s2"), "大盤"]];
    bars.forEach(([v, col, name], k) => {
      if (v == null) return;
      const bx = single ? cx - barw / 2 : cx - gw / 2 + k * (barw + 2);
      const yv = y(v), y0 = y(0);
      const rect = el("rect", {
        x: bx.toFixed(1), y: Math.min(yv, y0).toFixed(1), width: barw.toFixed(1),
        height: Math.max(1, Math.abs(y0 - yv)).toFixed(1), rx: 3, fill: col, "fill-opacity": 0.9,
      }, f.svg);
      rect.addEventListener("mousemove", (ev) => {
        let html = '<div class="t">' + r.y + "</div>"
          + '<div class="r"><span>' + name + "</span><b>" + fmtSignPct(v, 1) + "</b></div>";
        if (!single && bs[i] && bs[i].ret != null && r.ret != null && name === "策略")
          html += '<div class="r"><span>超額</span><b>' + fmtSignPct(r.ret - bs[i].ret, 1) + "</b></div>";
        showTip(html, ev.clientX, ev.clientY);
      });
      rect.addEventListener("mouseleave", hideTip);
    });
    el("text", { x: cx, y: f.H - 8, "text-anchor": "middle", fill: css("--muted"), "font-size": 11 }, f.svg)
      .textContent = r.y;
  });
}

/* ---------- 交易 panel ---------- */
function renderTradesPanel() {
  const T = P.trades;
  if (!T) return;
  const n = T.ret.length;
  $("tr-stats").innerHTML = "<span>共 <b>" + T.total + "</b> 筆</span>"
    + (T.sampled ? '<span class="sub">顯示抽樣 ' + n + " / " + T.total + " 筆</span>" : "");
  const idx = [...Array(n).keys()];
  const dir = state.tdir;
  const cmp = {
    ret: (a, b) => ((T.ret[a] == null ? -1e9 : T.ret[a]) - (T.ret[b] == null ? -1e9 : T.ret[b])),
    entry: (a, b) => (T.entry[a] || "").localeCompare(T.entry[b] || ""),
    exit: (a, b) => (T.exit[a] || "9999-99-99").localeCompare(T.exit[b] || "9999-99-99"),
  }[state.tsort] || ((a, b) => a - b);
  idx.sort((a, b) => dir * cmp(a, b));
  const per = 50, pages = Math.max(1, Math.ceil(n / per));
  if (state.tpage >= pages) state.tpage = pages - 1;
  const page = idx.slice(state.tpage * per, state.tpage * per + per);
  const arrow = (k) => state.tsort === k ? (dir > 0 ? " ▲" : " ▼") : "";
  let html = '<table class="tt"><thead><tr>'
    + "<th>代號 / 名稱</th>"
    + '<th class="sortable" data-s="ret">報酬' + arrow("ret") + "</th>"
    + '<th class="sortable" data-s="entry">進場' + arrow("entry") + "</th>"
    + '<th class="sortable" data-s="exit">出場' + arrow("exit") + "</th>"
    + "<th>持倉</th><th>MAE / GMFE</th><th>旗標</th></tr></thead><tbody>";
  for (const i of page) {
    const name = T.name && T.name[i] ? ' <span class="tsub">' + T.name[i] + "</span>" : "";
    const ret = T.ret[i] == null ? '<span class="tsub">–</span>'
      : '<span class="badge ' + (T.ret[i] >= 0 ? "up" : "dn") + '">' + (T.ret[i] >= 0 ? "▲ " : "▼ ") + fmtSignPct(T.ret[i], 1) + "</span>";
    const px = (v) => v == null ? "" : '<div class="tsub">$' + v + "</div>";
    const exitCell = T.exit[i] ? T.exit[i] + px(T.exit_px[i]) : '<span class="tsub">持有中</span>';
    let flags = "";
    if (T.lim_entry && T.lim_entry[i]) flags += '<span class="flag">漲停進</span>';
    if (T.lim_exit && T.lim_exit[i]) flags += '<span class="flag">跌停出</span>';
    html += "<tr><td>" + T.stock[i] + name + "</td><td>" + ret + "</td>"
      + "<td>" + (T.entry[i] || "–") + px(T.entry_px[i]) + "</td><td>" + exitCell + "</td>"
      + "<td>" + fmtPct(T.pos[i], 1) + "</td>"
      + '<td><span class="tsub">MAE ' + fmtPct(T.mae[i], 1) + " · GMFE " + fmtPct(T.gmfe[i], 1) + "</span></td>"
      + "<td>" + flags + "</td></tr>";
  }
  $("ttable-wrap").innerHTML = html + "</tbody></table>";
  $("ttable-wrap").querySelectorAll("th.sortable").forEach(th => th.addEventListener("click", () => {
    const k = th.dataset.s;
    if (state.tsort === k) state.tdir = -state.tdir; else { state.tsort = k; state.tdir = -1; }
    state.tpage = 0; renderTradesPanel();
  }));
  $("tpager").innerHTML = pages > 1
    ? '<button id="pg-prev"' + (state.tpage === 0 ? " disabled" : "") + ">‹ 上一頁</button>"
      + "<span>第 " + (state.tpage + 1) + " / " + pages + " 頁</span>"
      + '<button id="pg-next"' + (state.tpage >= pages - 1 ? " disabled" : "") + ">下一頁 ›</button>"
    : "";
  const prev = $("pg-prev"), next = $("pg-next");
  if (prev) prev.addEventListener("click", () => { state.tpage--; renderTradesPanel(); });
  if (next) next.addEventListener("click", () => { state.tpage++; renderTradesPanel(); });
}

/* ---------- 虧損歷史 panel ---------- */
function ddEpisodes() {
  return (state.ddSide === "bench" ? P.benchmark_dd_episodes : P.dd_episodes) || [];
}
function epLabel(ep) {
  return ep.trough ? ep.trough.slice(0, 4) + " " + ep.trough.slice(5, 7) + "M" : "–";
}
function renderDDPanel() {
  const hasBench = !!(P.benchmark_dd_episodes && P.benchmark_dd_episodes.length);
  $("dd-seg").innerHTML = hasBench
    ? [["strat", "策略"], ["bench", "大盤"]].map(([k, lbl]) =>
        '<button data-k="' + k + '"' + (state.ddSide === k ? ' class="on"' : "") + ">" + lbl + "</button>").join("")
    : "";
  $("dd-seg").querySelectorAll("button").forEach(b => b.addEventListener("click", () => {
    state.ddSide = b.dataset.k; state.ddSel = -1; renderDDPanel();
  }));
  $("dd-legend").innerHTML = legendHtml(
    P.daily.benchmark ? [[css("--neg"), "策略"], [css("--muted"), "大盤"]] : [[css("--neg"), "策略"]]);
  const eps = ddEpisodes();
  const sel = state.ddSel >= 0 ? eps[state.ddSel] : null;
  $("dd-caption").innerHTML = sel
    ? "<span>回檔幅度 <b>" + epLabel(sel) + "</b></span><span><b>" + fmtPct(sel.depth, 1) + "</b></span>"
      + "<span><b>" + (sel.days == null ? "–" : sel.days + " 天") + "</b>" + (sel.end ? "" : "（尚未回復）") + "</span>"
    : '<span class="sub">點下方排名可標示回檔區間</span>';
  renderDDChart(sel);
  const maxDepth = Math.max(0.001, ...eps.map(e => Math.abs(e.depth)));
  const barCol = state.ddSide === "bench" ? css("--muted") : css("--neg");
  $("dd-rank").innerHTML = eps.length ? eps.map((e, i) =>
    '<button class="ddrow' + (state.ddSel === i ? " on" : "") + '" data-i="' + i + '">'
    + '<span>' + epLabel(e) + "</span>"
    + '<span class="ddbar"><i style="width:' + (Math.abs(e.depth) / maxDepth * 100).toFixed(1) + "%;background:" + barCol + ';opacity:.75"></i></span>'
    + '<span class="ddv">' + fmtPct(e.depth, 1) + "</span>"
    + '<span class="ddd">' + (e.days == null ? "–" : e.days + " 天") + (e.end ? "" : " · 進行中") + "</span></button>"
  ).join("") : '<div class="sub">無回檔事件</div>';
  $("dd-rank").querySelectorAll(".ddrow").forEach(b => b.addEventListener("click", () => {
    const i = +b.dataset.i;
    state.ddSel = state.ddSel === i ? -1 : i;
    renderDDPanel();
  }));
}
function renderDDChart(sel) {
  const c = $("dd-chart"), f = frame(c);
  f.m.r = 16; f.iw = f.W - f.m.l - f.m.r;
  const dds = drawdownOf(P.daily.creturn);
  const ddb = P.daily.benchmark ? drawdownOf(P.daily.benchmark) : null;
  const allv = dds.concat(ddb ? ddb.filter(v => v != null) : []).filter(v => v != null);
  const lo = Math.min(...allv, -0.001);
  const t0 = DATES[0], t1 = LAST;
  const x = (t) => f.m.l + (t - t0) / (t1 - t0 || 1) * f.iw;
  const y = (v) => f.m.t + (v / lo) * f.ih;
  for (const tv of niceTicks(lo, 0, 4)) {
    el("line", { x1: f.m.l, x2: f.m.l + f.iw, y1: y(tv), y2: y(tv), stroke: css("--grid") }, f.svg);
    el("text", { x: f.m.l - 8, y: y(tv) + 4, "text-anchor": "end", fill: css("--muted"), "font-size": 11 }, f.svg)
      .textContent = fmtPct(tv, 0);
  }
  drawXAxis(f, x, t0, t1);
  if (sel) {
    const xs0 = x(parseDate(sel.start)), xs1 = x(sel.end ? parseDate(sel.end) : LAST);
    el("rect", { x: xs0.toFixed(1), y: f.m.t, width: Math.max(2, xs1 - xs0).toFixed(1), height: f.ih,
                 fill: css("--s1"), "fill-opacity": 0.12 }, f.svg);
  }
  const path = (vals) => {
    let d = "", pen = false;
    for (let i = 0; i < DATES.length; i++) {
      const v = vals[i];
      if (v == null) { pen = false; continue; }
      d += (pen ? "L" : "M") + x(DATES[i]).toFixed(1) + "," + y(v).toFixed(1);
      pen = true;
    }
    return d;
  };
  if (ddb) el("path", { d: path(ddb), fill: "none", stroke: css("--muted"), "stroke-width": 1.5 }, f.svg);
  el("path", { d: path(dds), fill: "none", stroke: css("--neg"), "stroke-width": 1.8 }, f.svg);
  const fillD = path(dds) + "L" + x(LAST).toFixed(1) + "," + y(0).toFixed(1) + "L" + x(DATES[0]).toFixed(1) + "," + y(0).toFixed(1) + "Z";
  el("path", { d: fillD, fill: css("--neg"), "fill-opacity": 0.1, stroke: "none" }, f.svg);
  attachCrosshair(c, f, DATES, x, (i, px, py) => {
    const dot = (col) => '<span class="dot" style="background:' + col + '"></span>';
    let html = '<div class="t">' + fmtDate(DATES[i]) + "</div>"
      + '<div class="r"><span>' + dot(css("--neg")) + "策略</span><b>" + fmtPct(dds[i]) + "</b></div>";
    if (ddb && ddb[i] != null)
      html += '<div class="r"><span>' + dot(css("--muted")) + "大盤</span><b>" + fmtPct(ddb[i]) + "</b></div>";
    showTip(html, px, py);
    return [y(dds[i])].concat(ddb && ddb[i] != null ? [y(ddb[i])] : []);
  }, [css("--neg"), css("--muted")]);
}

/* ---------- 報酬分布 panel ---------- */
function renderDistPanel() { renderHist(); renderScatter(); renderStops(); }
function renderHist() {
  const T = P.trades;
  if (!T || !T.ret.length) return;
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
      "<b>" + n + "</b> 筆交易介於 " + fmtPct(vlo, 1) + " … " + fmtPct(vlo + w, 1), ev.clientX, ev.clientY));
    bar.addEventListener("mouseleave", hideTip);
  });
  const sorted = [...rets].sort((a, b) => a - b);
  const q05 = sorted[Math.max(0, Math.floor(sorted.length * 0.05) - (sorted.length * 0.05 % 1 === 0 ? 1 : 0))];
  $("dist-note").textContent = q05 < 0
    ? "有 5% 的機率，單筆交易將有 " + fmtPct(-q05, 1) + " 以上的虧損"
    : "95% 的交易報酬高於 " + fmtSignPct(q05, 1);
}
function renderScatter() {
  const T = P.trades;
  if (!T || !T.ret.length) return;
  const c = $("scatter"), f = frame(c);
  f.m.r = 16; f.iw = f.W - f.m.l - f.m.r;
  const pts = [];
  for (let i = 0; i < T.ret.length; i++)
    if (T.ret[i] != null && T.mae[i] != null) pts.push(i);
  if (!pts.length) return;
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
      '<div class="t">' + T.stock[i] + " · " + T.entry[i] + " → " + (T.exit[i] || "持有中") + "</div>"
      + '<div class="r"><span>報酬</span><b>' + fmtSignPct(T.ret[i]) + "</b></div>"
      + '<div class="r"><span>MAE</span><b>' + fmtPct(T.mae[i]) + "</b></div>"
      + '<div class="r"><span>持有</span><b>' + T.pdays[i] + " 天</b></div>", ev.clientX, ev.clientY));
    dot.addEventListener("mouseleave", hideTip);
  }
  el("text", { x: f.m.l + f.iw, y: f.H - 8, "text-anchor": "end", fill: css("--muted"), "font-size": 10 }, f.svg)
    .textContent = T.sampled ? "抽樣 " + T.ret.length + " / " + T.total + " 筆" : "";
}
function renderStops() {
  const T = P.trades;
  const idx = [];
  for (let i = 0; i < T.ret.length; i++) if (T.ret[i] != null) idx.push(i);
  if (!idx.length) { $("stops").innerHTML = '<div class="sub">無已平倉交易</div>'; return; }
  const n = idx.length;
  const wr0 = idx.filter(i => T.ret[i] > 0).length / n;
  const mean0 = idx.reduce((s, i) => s + T.ret[i], 0) / n;
  let rows = "";
  for (const s of [0.05, 0.10, 0.20]) {
    const hit = idx.filter(i => T.mae[i] != null && T.mae[i] <= -s);
    const clipped = idx.map(i => (T.mae[i] != null && T.mae[i] <= -s) ? -s : T.ret[i]);
    const wr1 = clipped.filter(v => v > 0).length / n;
    const mean1 = clipped.reduce((a, v) => a + v, 0) / n;
    rows += "<tr><td>停損 " + (s * 100).toFixed(0) + "%</td>"
      + "<td>" + hit.length + " 筆 (" + fmtPct(hit.length / n, 1) + ")</td>"
      + "<td>" + fmtPct(wr0, 1) + " → " + fmtPct(wr1, 1) + "</td>"
      + "<td>" + fmtSignPct(mean0, 2) + " → " + fmtSignPct(mean1, 2)
      + " (" + fmtSignPct(mean1 - mean0, 2) + ")</td></tr>";
  }
  $("stops").innerHTML = '<table class="stats"><thead><tr><th>停損設定</th><th>觸發交易</th>'
    + "<th>勝率變化</th><th>平均報酬變化</th></tr></thead><tbody>" + rows + "</tbody></table>"
    + '<div class="sub" style="margin-top:6px">以 MAE 近似：MAE 低於停損線的交易視為以停損價出場（忽略費用與滑價）</div>';
}

/* ---------- 流動性 panel ---------- */
function renderLiqPanel() {
  const m = P.metrics || {}, ts = P.trade_summary || {};
  const cap = m.capacity;
  const bh = m.buyHigh != null ? m.buyHigh : ts.buy_high_ratio;
  const sl = m.sellLow != null ? m.sellLow : ts.sell_low_ratio;
  const tile = (label, value, sub, cls, tip) =>
    '<div class="tile" data-tip="' + tip + '"><div class="k">' + label + ' <span class="info">&#9432;</span></div>'
    + '<div class="v' + (cls ? " " + cls : "") + '">' + value + "</div>"
    + '<div class="c">' + sub + "</div></div>";
  const dotTxt = (ok) => ok == null ? "無資料" : ok ? "✓ 通過" : "✗ 未達標";
  $("liq-tiles").innerHTML =
    tile("胃納量", cap == null ? "–" : fmtMetric(cap, "wan"),
      dotTxt(cap == null ? null : cap > 500000) + " · 需 > 50 萬"
      + (cap == null ? "（需 trading_value 欄位）" : ""), "", DESC.capacity)
    + tile("買在漲停", bh == null ? "–" : fmtPct(bh, 1),
      dotTxt(bh == null ? null : bh < 0.05) + " · 需 < 5%"
      + (ts.buy_high_n != null ? " · " + ts.buy_high_n + " 筆" : ""), bh != null && bh >= 0.05 ? "dn" : "", DESC.buyHigh)
    + tile("賣在跌停", sl == null ? "–" : fmtPct(sl, 1),
      dotTxt(sl == null ? null : sl < 0.05) + " · 需 < 5%"
      + (ts.sell_low_n != null ? " · " + ts.sell_low_n + " 筆" : ""), sl != null && sl >= 0.05 ? "dn" : "", DESC.sellLow);
  /* evidence list: the specific trades that hit limit prices */
  const T = P.trades;
  let html = "<h2 style='margin-top:8px'>漲跌停成交明細 <span class='info' data-tip='進場當日即漲停（可能買不到）或出場當日即跌停（可能賣不掉）的交易'>&#9432;</span></h2>";
  if (!T || !T.ret.length) html += '<div class="sub">無交易資料</div>';
  else {
    const flagsKnown = (T.lim_entry || []).some(v => v != null) || (T.lim_exit || []).some(v => v != null);
    const hits = [];
    for (let i = 0; i < T.ret.length; i++)
      if ((T.lim_entry && T.lim_entry[i]) || (T.lim_exit && T.lim_exit[i])) hits.push(i);
    if (!flagsKnown)
      html += '<div class="sub">無漲跌停判定資料（input_df 需含 limit_up / limit_down 欄位）</div>';
    else if (!hits.length)
      html += '<div class="sub">✓ 無任何交易發生在漲停買進或跌停賣出</div>';
    else {
      const shown = hits.slice(0, 30);
      html += '<table class="tt"><thead><tr><th>代號 / 名稱</th><th>進場</th><th>出場</th><th>報酬</th><th>旗標</th></tr></thead><tbody>';
      for (const i of shown) {
        const name = T.name && T.name[i] ? ' <span class="tsub">' + T.name[i] + "</span>" : "";
        let flags = "";
        if (T.lim_entry && T.lim_entry[i]) flags += '<span class="flag">漲停進</span>';
        if (T.lim_exit && T.lim_exit[i]) flags += '<span class="flag">跌停出</span>';
        html += "<tr><td>" + T.stock[i] + name + "</td><td>" + (T.entry[i] || "–") + "</td>"
          + "<td>" + (T.exit[i] || '<span class="tsub">持有中</span>') + "</td>"
          + "<td>" + (T.ret[i] == null ? "–" : fmtSignPct(T.ret[i], 1)) + "</td><td>" + flags + "</td></tr>";
      }
      html += "</tbody></table>";
      if (hits.length > shown.length)
        html += '<div class="sub" style="margin-top:6px">僅顯示前 ' + shown.length + " 筆，共 " + hits.length + " 筆</div>";
    }
  }
  $("liq-list").innerHTML = html;
}

/* ---------- stats tables ---------- */
function renderStats() {
  if (!P.stat_groups) { $("stats-card").style.display = "none"; return; }
  $("stats-cols").innerHTML = P.stat_groups.map(g =>
    '<table class="stats kv">' + g.map(([k, v]) => "<tr><td>" + k + "</td><td>" + v + "</td></tr>").join("") + "</table>"
  ).join("");
}

/* ---------- info tooltips (ⓘ) ---------- */
$("app").addEventListener("mouseover", (ev) => {
  const n = ev.target.closest("[data-tip]");
  if (n && n.dataset.tip) showTip('<div class="t">' + n.dataset.tip + "</div>", ev.clientX, ev.clientY);
});
$("app").addEventListener("mouseout", (ev) => {
  if (ev.target.closest("[data-tip]")) hideTip();
});

/* ---------- boot ---------- */
function renderAll() {
  renderHeader(); renderHero(); renderQuality(); renderTabs(); showPanel(); renderStats();
}
renderAll();
let rsTimer = null;
addEventListener("resize", () => {
  clearTimeout(rsTimer);
  rsTimer = setTimeout(() => { (PANEL_RENDER[state.tab] || (() => {}))(); }, 150);
});
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", renderAll);
</script>
</body>
</html>
"""
