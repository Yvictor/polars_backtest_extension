# ruff: noqa: E501  (the template string embeds CSS/JS with long lines)
"""Self-contained HTML template for the interactive backtest report.

The template has no external assets (no CDN, no plotly): charts are inline SVG
rendered by vanilla JS, so the output file works offline and in notebook
iframes. Placeholders: __TITLE__, __PAYLOAD__, __VERSION__.

Layout is a quant-narrative one-page scroll (no tabs): 1 判決 (hero KPIs +
tradability flags + collapsible FinLab quality checklist), 2 績效軌跡 (log
equity, rolling-1Y panel, monthly heatmap, yearly bars), 3 報酬結構 (long/short
split, concentration), 4 回檔與痛苦 (underwater stats + dd episodes), 5 實盤
可行性 (limit-up dependency, capacity, cost structure), 6 交易顯微鏡
(collapsed: distributions, MAE, stops, trade table). Each section opens with a
data-driven zh-TW takeaway sentence.
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
  --good:        #0ca30c;  /* status ramp */
  --warn:        #b98200;  /* amber (light bg readable) */
  --bad:         #d03b3b;
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
    --warn:        #fab219;
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
header { margin-bottom: 8px; }
h1 { font-size: 20px; margin: 0; font-weight: 650; }
h2 { font-size: 14px; margin: 0 0 8px; font-weight: 600; color: var(--ink-2); }
.sub { color: var(--muted); font-size: 12px; }
.chip { border: 1px solid var(--border); background: var(--surface-1); color: var(--ink-2);
        border-radius: 999px; padding: 2px 10px; font-size: 12px; }
.info { color: var(--muted); font-size: 11px; cursor: help; font-weight: 400; }
/* sections */
.sec { margin: 26px 0 8px; }
.sec-title { display: flex; align-items: baseline; gap: 8px; font-size: 16px; font-weight: 700; margin: 0 0 2px; }
.sec-title .no { color: var(--muted); font-weight: 600; font-size: 13px; }
.takeaway { font-size: 13.5px; color: var(--ink-2); margin: 4px 0 12px; padding: 2px 0 2px 10px;
            border-left: 3px solid var(--s1); }
/* hero + tiles */
.hero { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 10px; margin-bottom: 12px; }
.tile { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 10px 12px; }
.tile .k { color: var(--muted); font-size: 12px; }
.tile .v { font-size: 22px; font-weight: 650; margin-top: 2px; }
.tile .v.xl { font-size: 30px; }
.tile .v.up { color: var(--up); } .tile .v.dn { color: var(--dn); } .tile .v.wa { color: var(--warn); }
.tile .c { color: var(--muted); font-size: 11px; margin-top: 2px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; }
/* tradability flag pills */
.flags { display: flex; flex-wrap: wrap; gap: 8px; margin: 0 0 12px; }
.pill { display: inline-flex; align-items: center; gap: 7px; border-radius: 999px; padding: 5px 13px;
        font-size: 12.5px; border: 1px solid var(--border); background: var(--surface-1);
        color: var(--ink-2); cursor: help; }
.pill b { font-weight: 650; color: var(--ink); }
.pill .st { width: 9px; height: 9px; border-radius: 50%; background: var(--axis); flex: none; }
.pill.g { border-color: color-mix(in srgb, var(--good) 45%, transparent); }
.pill.g .st { background: var(--good); }
.pill.a { border-color: color-mix(in srgb, var(--warn) 55%, transparent); }
.pill.a .st { background: var(--warn); }
.pill.r { border-color: color-mix(in srgb, var(--bad) 55%, transparent); }
.pill.r .st { background: var(--bad); }
/* collapsible quality checklist */
details.card > summary { cursor: pointer; }
.qsum { display: inline-flex; align-items: center; gap: 6px; margin-right: 16px; font-size: 12.5px; color: var(--ink-2); }
.qsum b.sc { font-size: 14px; font-variant-numeric: tabular-nums; }
.sc.good { color: var(--good); } .sc.warn { color: var(--warn); } .sc.bad { color: var(--bad); } .sc.na { color: var(--muted); }
.qdots { display: inline-flex; gap: 3px; }
.qdot { width: 7px; height: 7px; border-radius: 50%; display: inline-block; background: var(--axis); }
.qdot.ok { background: var(--good); } .qdot.no { background: var(--bad); }
.qtitle { font-weight: 600; color: var(--ink); margin-right: 14px; font-size: 13px; }
.dimh { margin: 12px 0 6px; font-size: 13px; font-weight: 600; color: var(--ink-2); }
/* metric tiles (quality detail) */
.mrow { display: grid; grid-template-columns: repeat(auto-fit, minmax(158px, 1fr)); gap: 10px; }
.mtile { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 8px 12px; }
.mtile .mk { color: var(--ink-2); font-size: 12px; display: flex; align-items: center; gap: 6px; }
.mtile .mv { font-size: 18px; font-weight: 650; margin-top: 2px; }
.mtile .mc { color: var(--muted); font-size: 11px; margin-top: 1px; }
.dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; flex: none; }
.d-ok { background: var(--good); } .d-no { background: var(--bad); } .d-na { background: var(--axis); }
/* cards */
.card { background: var(--surface-1); border: 1px solid var(--border); border-radius: 12px;
        padding: 14px 16px 12px; margin-bottom: 14px; }
.card .card { background: transparent; }
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
.row2.one { grid-template-columns: 1fr; }
@media (max-width: 800px) { .row2 { grid-template-columns: 1fr; } }
#tooltip { position: fixed; pointer-events: none; z-index: 10; display: none;
           background: var(--tip-bg); color: var(--tip-ink); border-radius: 8px;
           padding: 7px 10px; font-size: 12px; max-width: 320px;
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
.hm td { font-size: 11px; text-align: center; padding: 4px 2px; border-radius: 4px; min-width: 34px;
         font-variant-numeric: tabular-nums; }
.hm td.y { color: var(--ink-2); font-weight: 600; background: transparent; }
.hm td.tot { font-weight: 650; }
/* long/short split */
table.ls { width: 100%; border-collapse: collapse; font-size: 12.5px; }
table.ls th { text-align: right; color: var(--muted); font-weight: 500; font-size: 12px; padding: 5px 8px;
              border-bottom: 1px solid var(--axis); }
table.ls th:first-child, table.ls td:first-child { text-align: left; }
table.ls td { padding: 5px 8px; text-align: right; border-bottom: 1px solid var(--grid);
              font-variant-numeric: tabular-nums; }
.cbar { display: flex; height: 14px; border-radius: 7px; overflow: hidden; background: var(--div-mid); margin-top: 10px; }
.cbar i { display: block; height: 100%; }
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
.flag.wa { border-color: color-mix(in srgb, var(--warn) 55%, transparent); color: var(--warn); }
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
#micro > summary { font-size: 15px; font-weight: 650; }
footer { color: var(--muted); font-size: 12px; margin-top: 28px; text-align: center; }
#chips { display: inline-flex; flex-wrap: wrap; gap: 6px; justify-content: center; margin-bottom: 6px; }
svg text { font-family: inherit; }
</style>
</head>
<body class="viz-root">
<div id="app">
  <header>
    <h1 id="rpt-title"></h1>
    <div id="rpt-range" class="sub"></div>
  </header>

  <section class="sec" id="sec-verdict">
    <div class="sec-title"><span class="no">1</span>判決 <span class="info" data-tip="策略總評：核心績效數字、可交易性旗標與 FinLab 品質檢核">&#9432;</span></div>
    <div class="takeaway" id="tk-verdict"></div>
    <div class="hero" id="hero"></div>
    <div class="flags" id="flags"></div>
    <details class="card" id="qcheck">
      <summary><span id="qsummary"></span></summary>
      <div id="qdetail"></div>
    </details>
  </section>

  <section class="sec" id="sec-perf">
    <div class="sec-title"><span class="no">2</span>績效軌跡</div>
    <div class="takeaway" id="tk-perf"></div>
    <div class="card">
      <div class="card-head">
        <h2>歷史績效 <span class="info" data-tip="策略與大盤的累積報酬曲線（預設對數刻度），可切換線性/對數與時間區間；下方為同區間的策略回檔">&#9432;</span></h2>
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
    <div class="card" id="rolling-card">
      <h2>滾動 1 年表現 <span class="info" data-tip="以 252 個交易日視窗滾動計算的年化報酬與夏普值；夏普若逐年走低即為 Alpha 衰減訊號">&#9432;</span></h2>
      <div id="roll-ret" class="chart short"></div>
      <div id="roll-sharpe" class="chart short" style="margin-top:6px"></div>
    </div>
    <div class="row2">
      <div class="card" id="monthly-card">
        <div class="card-head">
          <h2>月報酬 (%) <span class="info" data-tip="每月報酬熱力圖，顏色深度代表漲跌幅；最右欄為年度合計">&#9432;</span></h2>
          <div class="kpis" id="mstats"></div>
        </div>
        <div id="heatmap" class="hm"></div>
      </div>
      <div class="card">
        <div class="card-head">
          <h2>年度比較 <span class="info" data-tip="策略與大盤的逐年報酬（每年重設基準），比較每年相對表現">&#9432;</span></h2>
          <div class="legend" id="yr-legend"></div>
        </div>
        <div class="kpis" id="yr-stats" style="margin-bottom:4px"></div>
        <div id="yearly-chart" class="chart mid"></div>
      </div>
    </div>
  </section>

  <section class="sec" id="sec-structure">
    <div class="sec-title"><span class="no">3</span>報酬結構</div>
    <div class="takeaway" id="tk-structure"></div>
    <div class="row2" id="sec3-row">
      <div class="card" id="ls-card">
        <h2>多空拆解 <span class="info" data-tip="多單與空單的筆數、勝率、平均報酬與貢獻占比（貢獻 = 報酬 × |持倉比重|）">&#9432;</span></h2>
        <div id="ls-body"></div>
      </div>
      <div class="card">
        <h2>報酬集中度 <span class="info" data-tip="每筆交易貢獻（報酬 × |持倉比重|）的分布；前 10 筆占比過高代表報酬依賴少數交易">&#9432;</span></h2>
        <div class="tiles" id="conc-tiles" style="margin-bottom:10px"></div>
        <div id="contrib-hist" class="chart mid"></div>
        <div class="sub" id="conc-note" style="margin-top:6px"></div>
      </div>
    </div>
  </section>

  <section class="sec" id="sec-dd">
    <div class="sec-title"><span class="no">4</span>回檔與痛苦</div>
    <div class="takeaway" id="tk-dd"></div>
    <div class="card">
      <div class="tiles" id="uw-tiles" style="margin-bottom:12px"></div>
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

  <section class="sec" id="sec-live">
    <div class="sec-title"><span class="no">5</span>實盤可行性</div>
    <div class="takeaway" id="tk-live"></div>
    <div class="card">
      <h2>漲停依賴 <span class="info" data-tip="進場當日即漲停的交易可能實際買不到；報酬貢獻占比衡量策略有多少損益建立在這些交易上">&#9432;</span></h2>
      <div class="tiles" id="lim-tiles" style="margin-bottom:8px"></div>
      <div class="sub" id="lim-note" style="margin-bottom:8px"></div>
      <div id="fill-scn"></div>
      <div id="liq-list"></div>
    </div>
    <div class="row2">
      <div class="card">
        <h2>胃納量 <span class="info" data-tip="不影響市場價格可部署的最大資金（FinLab 法，需 trading_value 欄位）">&#9432;</span></h2>
        <div class="tiles" id="cap-tiles" style="margin-bottom:8px"></div>
        <div class="sub" id="cap-note"></div>
      </div>
      <div class="card">
        <h2>成本結構 <span class="info" data-tip="年換手率與成本拖累估計；回測報酬已內含一倍成本">&#9432;</span></h2>
        <div class="tiles" id="cost-tiles" style="margin-bottom:8px"></div>
        <div class="sub" id="cost-note"></div>
      </div>
    </div>
  </section>

  <section class="sec" id="sec-micro">
    <details class="card" id="micro">
      <summary><span class="sec-title" style="display:inline-flex"><span class="no">6</span>交易顯微鏡</span> <span class="sub" id="micro-hint"></span></summary>
      <div class="row2" style="margin-top:12px">
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
      <div class="card">
        <div class="card-head">
          <h2>交易明細 <span class="info" data-tip="全部交易紀錄：報酬、進出場價、持倉比重、MAE/GMFE 與漲跌停旗標；點欄位標題可排序">&#9432;</span></h2>
          <div class="kpis" id="tr-stats"></div>
        </div>
        <div id="ttable-wrap" style="overflow-x:auto"></div>
        <div class="pager" id="tpager"></div>
      </div>
      <div class="card" id="stats-card">
        <h2>統計摘要</h2>
        <div class="cols3" id="stats-cols"></div>
      </div>
    </details>
  </section>

  <footer id="rpt-footer"><span id="chips"></span><div id="foot-meta"></div></footer>
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
function meanOf(arr) {
  const v = arr.filter(x => x != null);
  return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null;
}
/* HTML-escape user-derived strings before any innerHTML / tooltip interpolation */
const esc = (s) => String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

/* ---------- zh-TW metric descriptions ---------- */
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

/* ---------- state + shared series ---------- */
const state = {
  range: "all", scale: "log",
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
/* rolling 252-trading-day annualized return + Sharpe (rf 2%/yr), client-side */
const ROLL = (() => {
  const c = P.daily.creturn, n = c.length, W = 252;
  if (n < 300) return null;
  const rets = new Array(n).fill(0);
  let prev = null;
  for (let i = 0; i < n; i++) {
    const v = c[i];
    if (v != null) { if (prev != null && prev > 0) rets[i] = v / prev - 1; prev = v; }
  }
  const rfd = 0.02 / 252;
  const idxs = [], annRet = [], sharpe = [];
  let s = 0, s2 = 0;
  for (let i = 1; i < n; i++) {
    s += rets[i]; s2 += rets[i] * rets[i];
    if (i > W) { s -= rets[i - W]; s2 -= rets[i - W] * rets[i - W]; }
    if (i >= W) {
      const m = s / W, varr = Math.max(0, s2 / W - m * m), sd = Math.sqrt(varr);
      idxs.push(i);
      annRet.push(c[i] != null && c[i - W] != null && c[i - W] > 0 ? c[i] / c[i - W] - 1 : null);
      sharpe.push(sd > 1e-12 ? (m - rfd) / sd * Math.sqrt(252) : null);
    }
  }
  return { idxs, annRet, sharpe };
})();
function fullSharpe() {
  if (P.stats && typeof P.stats.daily_sharpe === "number") return P.stats.daily_sharpe;
  return ROLL ? meanOf(ROLL.sharpe) : null;
}
/* alpha decay: mean rolling-1Y Sharpe over the last 2 years vs full period */
function alphaDecay() {
  if (!ROLL || DATES.length < 756) return null;
  const cut = P.daily.creturn.length - 504;
  const all = [], recent = [];
  for (let k = 0; k < ROLL.idxs.length; k++) {
    const v = ROLL.sharpe[k];
    if (v == null) continue;
    all.push(v);
    if (ROLL.idxs[k] >= cut) recent.push(v);
  }
  if (all.length < 10 || recent.length < 10) return null;
  const full = all.reduce((a, b) => a + b, 0) / all.length;
  const rec = recent.reduce((a, b) => a + b, 0) / recent.length;
  return { full, rec, ratio: full > 1e-9 ? rec / full : null };
}
function heroNumbers() {
  const m = P.metrics || {}, s = P.stats || {};
  const cagr = m.annualReturn != null ? m.annualReturn : s.cagr;
  const md = m.maxDrawdown != null ? m.maxDrawdown : s.max_drawdown;
  const sh = m.sharpeRatio != null ? m.sharpeRatio : s.daily_sharpe;
  const cal = m.calmarRatio != null ? m.calmarRatio : s.calmar;
  return { cagr, md, sh, cal };
}
function benchCagr() {
  const b = P.daily.benchmark;
  if (!b) return null;
  let first = null, firstT = null, last = null, lastT = null;
  for (let i = 0; i < b.length; i++) {
    if (b[i] == null || b[i] <= 0) continue;
    if (first == null) { first = b[i]; firstT = DATES[i]; }
    last = b[i]; lastT = DATES[i];
  }
  if (first == null || lastT === firstT) return null;
  const years = (lastT - firstT) / 86400000 / 365.25;
  return years > 0 ? Math.pow(last / first, 1 / years) - 1 : null;
}
/* underwater statistics from the full daily curve */
const UW = (() => {
  const c = P.daily.creturn, ds = DATES;
  let peak = null, peakT = null, troughT = null, troughV = null, start = null;
  let below = 0, total = 0, maxDays = 0, recSum = 0, recN = 0;
  for (let i = 0; i < c.length; i++) {
    const v = c[i];
    if (v == null) continue;
    total++;
    if (peak == null || v >= peak) {
      if (start != null) {
        const days = (ds[i] - start) / 86400000;
        if (days > maxDays) maxDays = days;
        recSum += (ds[i] - troughT) / 86400000; recN++;
        start = null;
      }
      peak = v; peakT = ds[i];
    } else {
      below++;
      if (start == null) { start = peakT; troughT = ds[i]; troughV = v; }
      else if (v < troughV) { troughV = v; troughT = ds[i]; }
    }
  }
  let ongoing = false;
  if (start != null) {
    ongoing = true;
    const days = (LAST - start) / 86400000;
    if (days > maxDays) maxDays = days;
  }
  return {
    pctBelow: total ? below / total : null,
    maxDays: Math.round(maxDays),
    avgRec: recN ? recSum / recN : null,
    ongoing,
  };
})();

/* ---------- generic chart frame ---------- */
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

/* ---------- header / footer ---------- */
function renderHeader() {
  $("rpt-title").textContent = P.title;
  $("rpt-range").textContent = P.daily.dates[0] + " → " + P.daily.dates[P.daily.dates.length - 1]
    + "  ·  " + P.daily.dates.length + " trading days";
  $("chips").innerHTML = (P.params || []).map(p => '<span class="chip">' + esc(p) + "</span>").join("");
  $("foot-meta").textContent = "generated by polars-backtest __VERSION__ · schema " + (P.schema || "?")
    + " · 品質檢核採 corrected 模式（無資料項不列入計分）· 無風險利率 2%/年";
}

/* ---------- SECTION 1 判決 ---------- */
function renderVerdict() {
  const h = heroNumbers();
  const bc = benchCagr();
  const excess = h.cagr != null && bc != null ? h.cagr - bc : null;
  const items = [
    ["年化報酬 (CAGR)", fmtSignPct(h.cagr, 1), "xl " + (h.cagr != null && h.cagr < 0 ? "dn" : "up"), "", DESC.annualReturn],
    ["最大回檔", fmtPct(h.md, 1), "dn", "", DESC.maxDrawdown],
    ["夏普值", fmtNum(h.sh, 2), "", "", DESC.sharpeRatio],
    ["Calmar", fmtNum(h.cal, 2), "", "", DESC.calmarRatio],
    ["對大盤年化超額", fmtSignPct(excess, 1), excess == null ? "" : excess < 0 ? "dn" : "up",
     excess == null ? "未設定大盤" : "策略 CAGR − 大盤 CAGR (" + fmtSignPct(bc, 1) + ")",
     "年化超額報酬：策略 CAGR 減大盤 CAGR（需設定大盤）"],
  ];
  $("hero").innerHTML = items.map(([k, v, cls, cap, tip]) =>
    '<div class="tile" data-tip="' + tip + '"><div class="k">' + k + ' <span class="info">&#9432;</span></div>'
    + '<div class="v ' + cls + '">' + v + "</div>"
    + (cap ? '<div class="c">' + cap + "</div>" : "") + "</div>").join("");
  const flags = flagList();
  $("flags").innerHTML = flags.map(f =>
    '<span class="pill ' + f.cls + '" data-tip="' + f.tip + '"><span class="st"></span>'
    + f.label + " <b>" + f.val + "</b></span>").join("");
  const parts = [];
  if (h.cagr != null && h.md != null)
    parts.push("年化 " + fmtSignPct(h.cagr, 1) + "、最大回檔 " + fmtPct(h.md, 1)
      + (h.sh != null ? "、夏普 " + fmtNum(h.sh, 2) : ""));
  if (excess != null) parts.push("對大盤年化超額 " + fmtSignPct(excess, 1));
  const cnt = { g: 0, a: 0, r: 0, na: 0 };
  flags.forEach(f => { cnt[f.cls] = (cnt[f.cls] || 0) + 1; });
  parts.push("可交易性 " + cnt.g + " 綠 / " + cnt.a + " 黃 / " + cnt.r + " 紅"
    + (cnt.na ? "（" + cnt.na + " 項無資料）" : ""));
  $("tk-verdict").textContent = parts.join("；") + "。";
  renderQuality();
}
function flagList() {
  const m = P.metrics || {}, ts = P.trade_summary || {};
  const h = heroNumbers();
  const flags = [];
  const cap = m.capacity;
  flags.push({
    label: "資金容量",
    cls: cap == null ? "na" : cap >= 5e7 ? "g" : cap >= 1e7 ? "a" : "r",
    val: cap == null ? "無資料" : fmtMetric(cap, "wan"),
    tip: "規則：胃納量 ≥5000萬 綠 / ≥1000萬 黃 / 更低 紅（FinLab capacity 估計，需 trading_value）",
  });
  const lkc = ts.entry_locked_contrib_ratio;
  const bhc = ts.buy_high_contrib_ratio;
  const bh = m.buyHigh != null ? m.buyHigh : ts.buy_high_ratio;
  if (lkc != null) flags.push({
    label: "漲停依賴", cls: lkc <= 0.05 ? "g" : lkc <= 0.15 ? "a" : "r",
    val: "一字鎖死貢獻 " + fmtPct(lkc, 1),
    tip: "規則：一字鎖死進場（開盤即漲停且全日未打開，完全買不到）的報酬貢獻占比 ≤5% 綠 / ≤15% 黃 / >15% 紅",
  });
  else if (bhc != null) flags.push({
    label: "漲停依賴", cls: bhc <= 0.10 ? "g" : bhc <= 0.25 ? "a" : "r",
    val: "貢獻 " + fmtPct(bhc, 1),
    tip: "規則（後備，無一字鎖死分類）：漲停進場交易的報酬貢獻占比 ≤10% 綠 / ≤25% 黃 / >25% 紅",
  });
  else flags.push({
    label: "漲停依賴", cls: bh == null ? "na" : bh <= 0.05 ? "g" : bh <= 0.10 ? "a" : "r",
    val: bh == null ? "無資料" : "筆數 " + fmtPct(bh, 1),
    tip: "規則（後備，無貢獻資料）：買在漲停筆數比率 ≤5% 綠 / ≤10% 黃 / >10% 紅",
  });
  const cd = ts.cost_drag_annual;
  flags.push({
    label: "成本敏感",
    cls: (cd == null || h.cagr == null || h.cagr <= 0) ? "na" : cd <= 0.2 * h.cagr ? "g" : cd <= 0.5 * h.cagr ? "a" : "r",
    val: cd == null ? "無資料" : (h.cagr == null || h.cagr <= 0) ? "無法評估" : fmtPct(cd, 1) + "/年",
    tip: "規則：年成本拖累（換手率×(2×fee+tax)）≤ CAGR 的 20% 綠 / ≤50% 黃 / 更高 紅",
  });
  const ad = alphaDecay();
  flags.push({
    label: "Alpha衰減",
    cls: ad == null || ad.ratio == null ? "na" : ad.ratio >= 0.7 ? "g" : ad.ratio >= 0.4 ? "a" : "r",
    val: ad == null ? "資料不足" : ad.ratio == null ? "無法評估"
      : "近2年夏普 " + fmtNum(ad.rec, 2) + " / 全期 " + fmtNum(ad.full, 2),
    tip: "規則：近 2 年滾動 1 年夏普平均 ÷ 全期平均 ≥70% 綠 / ≥40% 黃 / 更低 紅（需 ≥3 年資料）",
  });
  const sides = ts.sides || {};
  if (sides.short) {
    const lc = sides.long ? sides.long.contrib : 0;
    const sc = sides.short.contrib;
    const tot = ts.total_contrib != null ? ts.total_contrib : lc + sc;
    let cls, val;
    if (Math.abs(tot) > 1e-12 && sc < -0.10 * Math.abs(tot)) { cls = "r"; val = "空單拖累 " + fmtSignPct(sc, 1); }
    else if (lc > 0 && sc > 0) { cls = "g"; val = "多空皆正貢獻"; }
    else { cls = "a"; val = (sc <= 0 ? "空單" : "多單") + "貢獻 ≤ 0"; }
    flags.push({
      label: "多空平衡", cls, val,
      tip: "規則：多空皆正貢獻 綠 / 一邊貢獻 ≤0 黃 / 空單貢獻低於總損益的 -10% 紅",
    });
  }
  return flags;
}
const DIMS = ["profitability", "risk", "ratio", "winrate", "liquidity"];
function renderQuality() {
  const q = P.quality || {};
  let sum = '<span class="qtitle">品質檢核（FinLab 27 項）</span>';
  for (const d of DIMS) {
    const info = q[d] || {}, sc = info.score;
    const cls = sc == null ? "na" : sc >= 70 ? "good" : sc >= 40 ? "warn" : "bad";
    const dots = (info.checks || []).map(c =>
      '<span class="qdot ' + (c.pass == null ? "na" : c.pass ? "ok" : "no") + '"></span>').join("");
    sum += '<span class="qsum"><b class="sc ' + cls + '">' + (sc == null ? "–" : sc) + "</b>"
      + (info.label || d) + '<span class="qdots">' + dots + "</span></span>";
  }
  sum += '<span class="sub">點擊展開明細</span>';
  $("qsummary").innerHTML = sum;
  $("qdetail").innerHTML = DIMS.map(d => {
    const info = q[d] || {};
    const tiles = (info.checks || []).map(c => {
      const dcls = c.pass == null ? "na" : c.pass ? "ok" : "no";
      return '<div class="mtile" data-tip="' + (DESC[c.key] || c.label) + '">'
        + '<div class="mk"><span class="dot d-' + dcls + '"></span>' + c.label + ' <span class="info">&#9432;</span></div>'
        + '<div class="mv">' + fmtMetric(c.value, c.fmt) + "</div>"
        + '<div class="mc">' + (c.value == null ? "無資料 · " : "") + (c.caption || "") + "</div></div>";
    }).join("");
    return '<div class="dimh">' + (info.label || d)
      + (info.score != null ? " · " + info.score + " 分" : " · 無資料") + "</div>"
      + '<div class="mrow">' + tiles + "</div>";
  }).join("");
}

/* ---------- SECTION 2 績效軌跡 ---------- */
function renderPerfSection() {
  renderControls(); renderEquity(); renderDrawdown(); renderYearChips();
  renderRolling(); renderMonthly(); renderYearly(); renderPerfTakeaway();
}
function renderPerfTakeaway() {
  const c = P.daily.creturn;
  const total = c[c.length - 1] != null && c[0] ? c[c.length - 1] / c[0] - 1 : null;
  const h = heroNumbers();
  const parts = [];
  if (total != null) parts.push("全期累積報酬 " + fmtSignPct(total, 1)
    + (h.cagr != null ? "（年化 " + fmtSignPct(h.cagr, 1) + "）" : ""));
  const ad = alphaDecay();
  if (ad != null && ad.ratio != null)
    parts.push("近兩年滾動夏普平均 " + fmtNum(ad.rec, 2) + "，為全期平均 " + fmtNum(ad.full, 2)
      + " 的 " + fmtPct(ad.ratio, 0));
  else if (ROLL == null) parts.push("資料不足 300 日，未顯示滾動 1 年面板");
  $("tk-perf").textContent = parts.length ? parts.join("；") + "。" : "無足夠日頻資料。";
}
function renderControls() {
  $("range-seg").innerHTML = RANGES.map(([k, lbl]) =>
    '<button data-k="' + k + '"' + (state.range === k ? ' class="on"' : "") + ">" + lbl + "</button>").join("");
  $("scale-seg").innerHTML = [["log", "Log"], ["linear", "Linear"]].map(([k, lbl]) =>
    '<button data-k="' + k + '"' + (state.scale === k ? ' class="on"' : "") + ">" + lbl + "</button>").join("");
  $("range-seg").querySelectorAll("button").forEach(b =>
    b.addEventListener("click", () => { state.range = b.dataset.k; renderControls(); renderEquity(); renderDrawdown(); renderYearChips(); }));
  $("scale-seg").querySelectorAll("button").forEach(b =>
    b.addEventListener("click", () => { state.scale = b.dataset.k; renderControls(); renderEquity(); }));
  $("eq-legend").innerHTML = P.daily.benchmark
    ? legendHtml([[css("--s1"), "策略"], [css("--s2"), "大盤"]]) : "";
}
function renderEquity() {
  const c = $("equity");
  if (!c.clientWidth) return;
  const f = frame(c);
  const [i0, i1] = visibleIdx();
  const xs = DATES.slice(i0, i1 + 1);
  if (xs.length < 2) { return; }
  const win = P.daily.creturn.slice(i0, i1 + 1);
  const rebase = win.find(v => v != null && v !== 0);
  if (rebase == null) { c.innerHTML = '<div class="sub">此區間無資料</div>'; return; }
  const ys = win.map(v => (v == null ? null : v / rebase));
  let bs = null;
  if (P.daily.benchmark) {
    const b0 = P.daily.benchmark.slice(i0, i1 + 1);
    const base = b0.find(v => v != null && v !== 0);
    if (base != null) bs = b0.map(v => (v == null ? null : v / base));
  }
  const all = ys.filter(v => v != null).concat(bs ? bs.filter(v => v != null) : []);
  if (!all.length) { c.innerHTML = '<div class="sub">此區間無資料</div>'; return; }
  let lo = Math.min(...all), hi = Math.max(...all);
  if (hi === lo) { hi += 0.01; lo -= 0.01; }
  const log = state.scale === "log" && lo > 0;
  const tr = log ? Math.log : (v) => v;
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
  const lastY = ys.filter(v => v != null).slice(-1)[0];
  if (lastY != null)
    el("text", { x: f.m.l + f.iw + 6, y: y(lastY) + 4, fill: css("--s1"), "font-size": 11, "font-weight": 600 }, f.svg)
      .textContent = "策略";

  attachCrosshair(c, f, xs, x, (i, px, py) => {
    const dot = (col) => '<span class="dot" style="background:' + col + '"></span>';
    let html = '<div class="t">' + fmtDate(xs[i]) + "</div>"
      + '<div class="r"><span>' + dot(css("--s1")) + "策略</span><b>"
      + (ys[i] == null ? "–" : fmtSignPct(ys[i] - 1)) + "</b></div>";
    if (bs && bs[i] != null)
      html += '<div class="r"><span>' + dot(css("--s2")) + "大盤</span><b>" + fmtSignPct(bs[i] - 1) + "</b></div>";
    showTip(html, px, py);
    const pts = [];
    if (ys[i] != null) pts.push(y(ys[i]));
    if (bs && bs[i] != null) pts.push(y(bs[i]));
    return pts;
  }, [css("--s1"), css("--s2")]);
}
function renderDrawdown() {
  const c = $("drawdown");
  if (!c.clientWidth) return;
  const f = frame(c);
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
  for (let i = 0; i < xs.length; i++)
    if (dd[i] != null) d += "L" + x(xs[i]).toFixed(1) + "," + y(dd[i]).toFixed(1);
  d += "L" + x(xs[xs.length - 1]).toFixed(1) + "," + y(0).toFixed(1) + "Z";
  el("path", { d, fill: css("--neg"), "fill-opacity": 0.18, stroke: css("--neg"), "stroke-width": 1.5 }, f.svg);
  el("text", { x: f.m.l + f.iw + 6, y: f.m.t + 12, fill: css("--muted"), "font-size": 11 }, f.svg).textContent = "回檔";
  attachCrosshair(c, f, xs, x, (i, px, py) => {
    showTip('<div class="t">' + fmtDate(xs[i]) + '</div><div class="r"><span>回檔</span><b>'
      + fmtPct(dd[i]) + "</b></div>", px, py);
    return dd[i] == null ? [] : [y(dd[i])];
  }, [css("--neg")]);
}
function renderYearChips() {
  const rows = yearlyReturns(P.daily.creturn);
  const brows = P.daily.benchmark ? yearlyReturns(P.daily.benchmark) : null;
  const cr = P.daily.creturn;
  const first = cr.find(v => v != null && v !== 0);
  const lastv = cr.filter(v => v != null).slice(-1)[0];
  const total = first != null && lastv != null ? lastv / first - 1 : null;
  const chip = (key, label, ret, tip) =>
    '<button class="ychip ' + (ret != null && ret < 0 ? "dn" : "up") + (state.range === key ? " on" : "")
    + '" data-r="' + key + '"' + (tip ? ' data-tip="' + tip + '"' : "") + ">" + label + " <b>" + fmtSignPct(ret, 1) + "</b></button>";
  $("ychips").innerHTML = chip("all", "全部", total)
    + rows.map((r, i) => chip("y:" + r.y, r.y, r.ret,
        brows && brows[i] && brows[i].ret != null ? "大盤 " + fmtSignPct(brows[i].ret, 1) : null)).join("");
  $("ychips").querySelectorAll(".ychip").forEach(b => b.addEventListener("click", () => {
    state.range = b.dataset.r === "all" ? "all" : b.dataset.r;
    renderControls(); renderEquity(); renderDrawdown(); renderYearChips();
  }));
}
function renderRolling() {
  const card = $("rolling-card");
  if (!ROLL) { card.style.display = "none"; return; }
  card.style.display = "";
  const xs = ROLL.idxs.map(i => DATES[i]);
  drawRollChart("roll-ret", xs, ROLL.annRet, {
    fmt: (v) => fmtSignPct(v, 0), fmtTip: (v) => fmtSignPct(v, 1),
    color: css("--s1"), label: "滾動 1 年報酬",
    refs: [{ v: 0, color: css("--axis"), dash: false }],
  });
  const ref = fullSharpe();
  drawRollChart("roll-sharpe", xs, ROLL.sharpe, {
    fmt: (v) => fmtNum(v, 1), fmtTip: (v) => fmtNum(v, 2),
    color: css("--s2"), label: "滾動 1 年夏普",
    refs: ref != null ? [{ v: ref, color: css("--muted"), dash: true, text: "全期 " + fmtNum(ref, 2) }] : [],
  });
}
function drawRollChart(cid, xs, vals, o) {
  const c = $(cid);
  if (!c || !c.clientWidth) return;
  const f = frame(c);
  const vv = vals.filter(v => v != null);
  if (vv.length < 2) return;
  let lo = Math.min(...vv), hi = Math.max(...vv);
  for (const r of o.refs) { lo = Math.min(lo, r.v); hi = Math.max(hi, r.v); }
  if (hi === lo) { hi += 0.01; lo -= 0.01; }
  const pad = (hi - lo) * 0.1;
  lo -= pad; hi += pad;
  const t0 = xs[0], t1 = xs[xs.length - 1];
  const x = (t) => f.m.l + (t - t0) / (t1 - t0 || 1) * f.iw;
  const y = (v) => f.m.t + (1 - (v - lo) / (hi - lo)) * f.ih;
  for (const tv of niceTicks(lo, hi, 3)) {
    el("line", { x1: f.m.l, x2: f.m.l + f.iw, y1: y(tv), y2: y(tv), stroke: css("--grid") }, f.svg);
    el("text", { x: f.m.l - 8, y: y(tv) + 4, "text-anchor": "end", fill: css("--muted"), "font-size": 11 }, f.svg)
      .textContent = o.fmt(tv);
  }
  drawXAxis(f, x, t0, t1);
  for (const r of o.refs) {
    el("line", { x1: f.m.l, x2: f.m.l + f.iw, y1: y(r.v), y2: y(r.v), stroke: r.color,
                 "stroke-width": 1.2, "stroke-dasharray": r.dash ? "5,4" : "" }, f.svg);
    if (r.text)
      el("text", { x: f.m.l + f.iw + 6, y: y(r.v) + 4, fill: r.color, "font-size": 10.5 }, f.svg).textContent = r.text;
  }
  let d = "", pen = false;
  for (let i = 0; i < xs.length; i++) {
    const v = vals[i];
    if (v == null) { pen = false; continue; }
    d += (pen ? "L" : "M") + x(xs[i]).toFixed(1) + "," + y(v).toFixed(1);
    pen = true;
  }
  el("path", { d, fill: "none", stroke: o.color, "stroke-width": 1.8 }, f.svg);
  el("text", { x: f.m.l, y: f.m.t + 4, fill: o.color, "font-size": 11, "font-weight": 600 }, f.svg)
    .textContent = o.label;
  attachCrosshair(c, f, xs, x, (i, px, py) => {
    const v = vals[i];
    showTip('<div class="t">' + fmtDate(xs[i]) + '</div><div class="r"><span>' + o.label + "</span><b>"
      + (v == null ? "–" : o.fmtTip(v)) + "</b></div>", px, py);
    return v == null ? [] : [y(v)];
  }, [o.color]);
}
function renderMonthly() {
  if (!P.return_table || !P.return_table.length) {
    $("monthly-card").style.display = "none";
    return;
  }
  const cells = P.return_table.flatMap(r => r.months.filter(v => v != null));
  if (cells.length) {
    const avg = cells.reduce((a, v) => a + v, 0) / cells.length;
    const win = cells.filter(v => v > 0).length / cells.length;
    $("mstats").innerHTML = "<span>平均月報酬 <b>" + fmtSignPct(avg, 2) + "</b></span>"
      + "<span>月勝率 <b>" + fmtPct(win, 1) + "</b></span>";
  }
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
function renderYearly() {
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

  const c = $("yearly-chart");
  if (!c.clientWidth) return;
  const f = frame(c);
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

/* ---------- SECTION 3 報酬結構 ---------- */
function tradeContribs() {
  const T = P.trades;
  if (!T || !T.ret.length) return [];
  const out = [];
  for (let i = 0; i < T.ret.length; i++)
    if (T.ret[i] != null && T.pos[i] != null) out.push(T.ret[i] * Math.abs(T.pos[i]));
  return out;
}
function renderStructureSection() {
  const ts = P.trade_summary || {};
  const sides = ts.sides || {};
  const hasShort = !!sides.short;
  const card = $("ls-card");
  if (!hasShort) {
    card.style.display = "none";
    $("sec3-row").classList.add("one");
  } else {
    card.style.display = "";
    $("sec3-row").classList.remove("one");
    const lc = sides.long ? sides.long.contrib : 0;
    const sc = sides.short.contrib;
    const tot = ts.total_contrib != null ? ts.total_contrib : lc + sc;
    const share = (s) => Math.abs(tot) > 1e-12 && s && s.contrib != null ? fmtPct(s.contrib / tot, 1) : "–";
    const row = (lbl, s) => !s ? "" :
      "<tr><td>" + lbl + "</td><td>" + s.n + "</td><td>" + fmtPct(s.win_rate, 1) + "</td><td>"
      + fmtSignPct(s.avg_ret, 2) + "</td><td>" + fmtSignPct(s.contrib, 1) + "</td><td>" + share(s) + "</td></tr>";
    let html = '<table class="ls"><thead><tr><th>方向</th><th>筆數</th><th>勝率</th><th>平均報酬</th><th>貢獻</th><th>貢獻占比</th></tr></thead><tbody>'
      + row("多單", sides.long) + row("空單", sides.short) + "</tbody></table>";
    const la = Math.abs(lc), sa = Math.abs(sc), tw = la + sa;
    if (tw > 1e-12) {
      const lw = (la / tw * 100).toFixed(1), sw = (sa / tw * 100).toFixed(1);
      html += '<div class="cbar">'
        + '<i style="width:' + lw + "%;background:" + (lc >= 0 ? css("--s1") : css("--neg")) + '"></i>'
        + '<i style="width:' + sw + "%;background:" + (sc >= 0 ? css("--s2") : css("--neg")) + '"></i></div>'
        + '<div class="sub" style="margin-top:4px">多單 ' + fmtSignPct(lc, 1) + " · 空單 " + fmtSignPct(sc, 1)
        + "（占 NAV 加權損益）</div>";
    }
    $("ls-body").innerHTML = html;
  }
  renderConcentration();
  /* takeaway */
  const parts = [];
  if (hasShort) {
    const lc = sides.long ? sides.long.contrib : 0, sc = sides.short.contrib;
    const tot = ts.total_contrib != null ? ts.total_contrib : lc + sc;
    const main = Math.abs(lc) >= Math.abs(sc) ? "多單" : "空單";
    parts.push("空單貢獻 " + (Math.abs(tot) > 1e-12 ? fmtSignPct(sc / tot, 0) + " 的損益" : fmtSignPct(sc, 1))
      + "，主要報酬來自" + main);
  }
  if (ts.top10_contrib_ratio != null)
    parts.push("前 10 筆交易貢獻總損益的 " + fmtPct(ts.top10_contrib_ratio, 0)
      + (ts.top10_contrib_ratio > 0.4 ? "，報酬高度集中於少數交易" : "，集中度尚可"));
  $("tk-structure").textContent = parts.length ? parts.join("；") + "。" : "無交易資料，無法分析報酬結構。";
}
function renderConcentration() {
  const ts = P.trade_summary || {};
  const tile = (k, v, cap, cls, tip) =>
    '<div class="tile" data-tip="' + tip + '"><div class="k">' + k + ' <span class="info">&#9432;</span></div>'
    + '<div class="v' + (cls ? " " + cls : "") + '">' + v + "</div>"
    + '<div class="c">' + cap + "</div></div>";
  const r10 = ts.top10_contrib_ratio;
  $("conc-tiles").innerHTML =
    tile("前 10 筆貢獻占比", r10 == null ? "–" : fmtPct(r10, 1),
      "經驗法則：>40% 表示報酬高度集中", r10 != null && r10 > 0.4 ? "dn" : "",
      "貢獻最高的 10 筆交易占總損益（報酬 × |持倉|加總）的比例")
    + tile("交易筆數", ts.n != null ? ts.n : "–",
      ts.win_rate != null ? "勝率 " + fmtPct(ts.win_rate, 1) : "", "", "已成立的交易筆數與逐筆勝率");
  const vals = tradeContribs();
  drawHist("contrib-hist", vals, (v) => fmtPct(v, 2));
  const T = P.trades;
  $("conc-note").textContent = !vals.length ? "無交易資料"
    : "每筆交易貢獻（報酬 × |持倉比重|，占 NAV）的分布"
      + (T && T.sampled ? "；以抽樣 " + T.ret.length + " / " + T.total + " 筆計算" : "");
}
/* shared histogram: green/red by sign */
function drawHist(cid, values, xfmt) {
  const c = $(cid);
  if (!c || !c.clientWidth) return;
  const f = frame(c);
  f.m.r = 16; f.iw = f.W - f.m.l - f.m.r;
  if (!values.length) return;
  const lo = Math.min(...values), hi = Math.max(...values);
  const nb = Math.min(40, Math.max(10, Math.round(Math.sqrt(values.length) * 1.5)));
  const w = (hi - lo) / nb || 1e-9;
  const bins = Array.from({ length: nb }, () => 0);
  for (const v of values) bins[Math.min(nb - 1, Math.floor((v - lo) / w))]++;
  const ymax = Math.max(...bins);
  const x = (v) => f.m.l + (v - lo) / (hi - lo || 1) * f.iw;
  const y = (n) => f.m.t + (1 - n / ymax) * f.ih;
  for (const tv of niceTicks(0, ymax, 4)) {
    el("line", { x1: f.m.l, x2: f.m.l + f.iw, y1: y(tv), y2: y(tv), stroke: css("--grid") }, f.svg);
    el("text", { x: f.m.l - 8, y: y(tv) + 4, "text-anchor": "end", fill: css("--muted"), "font-size": 11 }, f.svg).textContent = tv;
  }
  for (const tv of niceTicks(lo, hi, 6))
    el("text", { x: x(tv), y: f.H - 8, "text-anchor": "middle", fill: css("--muted"), "font-size": 11 }, f.svg)
      .textContent = xfmt(tv);
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
      "<b>" + n + "</b> 筆交易介於 " + xfmt(vlo) + " … " + xfmt(vlo + w), ev.clientX, ev.clientY));
    bar.addEventListener("mouseleave", hideTip);
  });
}

/* ---------- SECTION 4 回檔與痛苦 ---------- */
function renderDDSection() {
  const tile = (k, v, cap, cls, tip) =>
    '<div class="tile" data-tip="' + tip + '"><div class="k">' + k + ' <span class="info">&#9432;</span></div>'
    + '<div class="v' + (cls ? " " + cls : "") + '">' + v + "</div>"
    + '<div class="c">' + cap + "</div></div>";
  $("uw-tiles").innerHTML =
    tile("水下時間比例", fmtPct(UW.pctBelow, 1), "低於前高的交易日占比", "",
      "策略淨值低於歷史高點（回檔中）的交易日比例")
    + tile("最長水下天數", UW.maxDays + " 天", UW.ongoing ? "目前回檔進行中" : "高點到回復", "",
      "最長的一次「跌落前高到收復前高」所經歷的日曆天數")
    + tile("平均修復天數", UW.avgRec == null ? "–" : Math.round(UW.avgRec) + " 天",
      "低谷到收復前高（已完成事件）", "", "各回檔事件中，從低谷回到前高所需天數的平均");
  $("tk-dd").textContent = UW.pctBelow == null ? "無足夠資料計算水下統計。"
    : "策略有 " + fmtPct(UW.pctBelow, 0) + " 的交易日處於回檔中；最長水下 " + UW.maxDays + " 天"
      + (UW.ongoing ? "（進行中）" : "")
      + (UW.avgRec != null ? "，低谷平均 " + Math.round(UW.avgRec) + " 天修復" : "") + "。";
  renderDDPanel();
}
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
  const c = $("dd-chart");
  if (!c.clientWidth) return;
  const f = frame(c);
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

/* ---------- SECTION 5 實盤可行性 ---------- */
function renderLiveSection() {
  const m = P.metrics || {}, ts = P.trade_summary || {};
  const h = heroNumbers();
  const tile = (k, v, cap, cls, tip) =>
    '<div class="tile" data-tip="' + tip + '"><div class="k">' + k + ' <span class="info">&#9432;</span></div>'
    + '<div class="v' + (cls ? " " + cls : "") + '">' + v + "</div>"
    + '<div class="c">' + cap + "</div></div>";
  const bh = m.buyHigh != null ? m.buyHigh : ts.buy_high_ratio;
  const sl = m.sellLow != null ? m.sellLow : ts.sell_low_ratio;
  const bhc = ts.buy_high_contrib_ratio;
  const hasKinds = ts.entry_locked_n != null || ts.entry_touched_n != null;
  let limHtml =
    tile("買在漲停比率", bh == null ? "–" : fmtPct(bh, 1),
      bh == null ? "需 limit_up / limit_down 資料" : (ts.buy_high_n != null ? ts.buy_high_n + " 筆" : "") + " · 需 < 5%",
      bh != null && bh >= 0.05 ? "dn" : "", DESC.buyHigh)
    + tile("漲停報酬貢獻占比", bhc == null ? "–" : fmtPct(bhc, 1),
      bhc == null ? "無貢獻資料" : "若漲停買不到，這部分報酬將消失",
      bhc != null && bhc > 0.25 ? "dn" : "",
      "漲停進場交易的損益（報酬 × |持倉|）占策略總損益的比例");
  if (hasKinds) {
    const lkN = ts.entry_locked_n || 0, tcN = ts.entry_touched_n || 0;
    const lkc = ts.entry_locked_contrib_ratio;
    limHtml += tile("一字鎖死進場", lkN + " 筆",
      (lkc != null ? "貢獻 " + fmtPct(lkc, 1) + " · " : "") + "完全買不到",
      lkN > 0 ? "dn" : "",
      "開盤即漲停且全日未打開（open==low==limit）的進場，掛單完全無法成交")
      + tile("盤中觸及進場", tcN + " 筆", "有監控可買到", tcN > 0 ? "wa" : "",
        "當日曾於漲停價以下成交的漲停進場，監控下仍有機會買進");
  }
  limHtml += tile("賣在跌停", sl == null ? "–" : fmtPct(sl, 1),
    (sl == null ? "需 limit_up / limit_down 資料" : (ts.sell_low_n != null ? ts.sell_low_n + " 筆" : "") + " · 需 < 5%")
    + (ts.exit_locked_n != null ? " · 一字 " + ts.exit_locked_n + " / 觸及 " + (ts.exit_touched_n || 0) : ""),
    sl != null && sl >= 0.05 ? "dn" : "", DESC.sellLow);
  $("lim-tiles").innerHTML = limHtml;
  $("lim-note").textContent = "若漲停日實際買不到，「漲停報酬貢獻」這部分報酬將消失；下表依 |貢獻| 由大到小列出漲跌停成交。"
    + (hasKinds ? " 一字鎖死＝開盤即漲停且全日未打開，掛單完全無法成交；盤中觸及＝當日曾於漲停價下成交，監控下仍有機會買進。" : "");
  renderFillScenarios();
  renderLimitEvidence();
  const cap = m.capacity, capMin = m.capacityMinLeg, capAdv = m.capacityAdv;
  let capHtml = tile("胃納量（FinLab 法）", cap == null ? "–" : fmtMetric(cap, "wan"),
    cap == null ? "需 trading_value 欄位" : cap >= 5e7 ? "≥ 5000萬 · 容量充足" : cap >= 1e7 ? "1000萬–5000萬 · 容量有限" : "< 1000萬 · 容量不足",
    cap != null && cap < 1e7 ? "dn" : "", DESC.capacity);
  if (capMin != null || capAdv != null) {
    capHtml += tile("保守估計（min-leg 法）", capMin == null ? "–" : fmtMetric(capMin, "wan"),
      "取進出場較薄的一腿 · 較保守", capMin != null && capMin < 1e7 ? "dn" : "",
      "以進場/出場中成交金額較薄的一腿估計可部署資金，較 FinLab 法保守")
      + tile("ADV 法", capAdv == null ? "–" : fmtMetric(capAdv, "wan"),
        "訊號日前 20 日中位數", capAdv != null && capAdv < 1e7 ? "dn" : "",
        "以訊號日前 20 日成交金額中位數 (ADV) 估計，貼近實盤 sizing");
    $("cap-note").textContent = "FinLab 法以 5% 參與率平均進出兩腿；min-leg 法取較薄的一腿（較保守）；"
      + "ADV 法用訊號日前 20 日中位數成交金額（貼近實盤 sizing）。";
  } else {
    $("cap-note").textContent = "此處為 FinLab 法估計；更嚴格的 min_leg / ADV 估計可透過 report.capacity() 取得。";
  }
  $("cap-tiles").innerHTML = capHtml;
  const turn = ts.annual_turnover, cd = ts.cost_drag_annual;
  $("cost-tiles").innerHTML =
    tile("年換手率", turn == null ? "–" : fmtNum(turn, 1) + "x",
      "每年部署的部位總量 / NAV", "", "年化換手率：全部進場部位比重加總 ÷ 回測年數")
    + tile("成本拖累估計 / 年", cd == null ? "–" : fmtPct(cd, 1),
      "換手率 × (2×手續費 + 稅)", "", "以年換手率乘上單次往返成本估計的年化成本拖累")
    + tile("成本×2 時年化（近似）", cd == null || h.cagr == null ? "–" : fmtSignPct(h.cagr - cd, 1),
      "CAGR − 年拖累", cd != null && h.cagr != null && h.cagr - cd < 0 ? "dn" : "up",
      "回測已含一倍成本；若成本假設加倍，年化約再扣一次拖累");
  $("cost-note").textContent = "估計式：年換手率 × (2×手續費率 + 交易稅率)。回測報酬已內含一倍成本；"
    + "「成本×2 時年化」以 CAGR − 年拖累近似回答「若成本假設加倍還剩多少」。";
  /* takeaway */
  const parts = [];
  if (turn != null) parts.push("年換手約 " + fmtNum(turn, 1) + " 倍"
    + (cd != null ? "，成本拖累約 " + fmtPct(cd, 1) + "/年" : ""));
  if (bhc != null) parts.push("漲停進場貢獻 " + fmtPct(bhc, 1) + " 的報酬");
  else if (bh != null) parts.push("買在漲停占 " + fmtPct(bh, 1) + " 筆");
  const scn = P.fill_scenarios || [];
  const scnBase = scn.find(s => s.name === "baseline");
  const scnLocked = scn.find(s => s.name === "locked");
  if (scnBase && scnLocked && scnBase.cagr != null && scnLocked.cagr != null)
    parts.push("若一字鎖死均買不到，年化 " + fmtSignPct(scnBase.cagr, 1) + " → " + fmtSignPct(scnLocked.cagr, 1));
  if (cap != null) parts.push("胃納量約 " + fmtMetric(cap, "wan"));
  $("tk-live").textContent = parts.length ? parts.join("；") + "。"
    : "缺少成本與漲跌停資料，無法完整評估實盤可行性。";
}
function renderFillScenarios() {
  const S = P.fill_scenarios;
  const box = $("fill-scn");
  if (!S || !S.length) { box.innerHTML = ""; return; }
  const base = S.find(s => s.name === "baseline") || S[0];
  const NAME = {
    baseline: "baseline（全部成交）",
    locked: "排除一字鎖死（有監控）",
    at_limit: "排除所有漲停進場（保守）",
  };
  const cagrCell = (s) => {
    if (s.cagr == null) return "–";
    if (s.name === "baseline" || base.cagr == null || base.cagr <= 0)
      return "<b>" + fmtSignPct(s.cagr, 1) + "</b>";
    const drop = (base.cagr - s.cagr) / Math.abs(base.cagr);
    const col = drop < 0.10 ? "var(--good)" : drop < 0.30 ? "var(--warn)" : "var(--bad)";
    return '<b style="color:' + col + '">' + fmtSignPct(s.cagr, 1) + "</b>"
      + ' <span class="tsub">(' + fmtSignPct(s.cagr - base.cagr, 1) + ")</span>";
  };
  let rows = "";
  for (const s of S) {
    rows += "<tr><td>" + (NAME[s.name] || esc(s.name)) + "</td>"
      + "<td>" + cagrCell(s) + "</td>"
      + "<td>" + fmtSignPct(s.total_return, 1) + "</td>"
      + "<td>" + fmtPct(s.max_drawdown, 1) + "</td>"
      + "<td>" + fmtNum(s.daily_sharpe, 2) + "</td>"
      + "<td>" + fmtNum(s.calmar, 2) + "</td>"
      + "<td>" + (s.blocked_n != null ? s.blocked_n : "–") + "</td></tr>";
  }
  box.innerHTML = '<h2 style="margin-top:8px">買不到情境模擬 <span class="info" data-tip="將買不到的進場訊號歸零後重新回測，觀察策略在真實成交限制下還剩多少">&#9432;</span></h2>'
    + '<table class="stats"><thead><tr><th>情境</th><th>年化報酬</th><th>總報酬</th><th>最大回檔</th>'
    + "<th>Sharpe</th><th>Calmar</th><th>排除筆數</th></tr></thead><tbody>" + rows + "</tbody></table>"
    + '<div class="sub" style="margin:6px 0 10px">重新回測：被排除的進場訊號歸零、資金留在現金（非事後扣減）。</div>';
}
/* kind-aware limit badges: 一字 (locked, red) / 盤中 (touched, amber) with generic fallback */
function flagBadges(T, i) {
  let out = "";
  if (T.lim_entry && T.lim_entry[i]) {
    const k = T.entry_kind ? T.entry_kind[i] : null;
    if (k === "locked") out += '<span class="flag">一字漲停進</span>';
    else if (k === "touched") out += '<span class="flag wa">盤中漲停進</span>';
    else out += '<span class="flag">漲停進</span>';
  }
  if (T.lim_exit && T.lim_exit[i]) {
    const k = T.exit_kind ? T.exit_kind[i] : null;
    if (k === "locked") out += '<span class="flag">一字跌停出</span>';
    else if (k === "touched") out += '<span class="flag wa">盤中跌停出</span>';
    else out += '<span class="flag">跌停出</span>';
  }
  return out;
}
function renderLimitEvidence() {
  const T = P.trades;
  let html = "";
  if (!T || !T.ret.length) html = '<div class="sub">無交易資料</div>';
  else {
    const flagsKnown = (T.lim_entry || []).some(v => v != null) || (T.lim_exit || []).some(v => v != null);
    const contribOf = (i) => (T.ret[i] == null || T.pos[i] == null) ? null : T.ret[i] * Math.abs(T.pos[i]);
    const hits = [];
    for (let i = 0; i < T.ret.length; i++)
      if ((T.lim_entry && T.lim_entry[i]) || (T.lim_exit && T.lim_exit[i])) hits.push(i);
    if (!flagsKnown)
      html = '<div class="sub">無漲跌停判定資料（input_df 需含 limit_up / limit_down 欄位）</div>';
    else if (!hits.length)
      html = '<div class="sub">✓ 無任何交易發生在漲停買進或跌停賣出</div>';
    else {
      hits.sort((a, b) => Math.abs(contribOf(b) == null ? 0 : contribOf(b)) - Math.abs(contribOf(a) == null ? 0 : contribOf(a)));
      const shown = hits.slice(0, 30);
      html = '<h2 style="margin-top:8px">漲跌停成交明細 <span class="info" data-tip="進場當日即漲停（可能買不到）或出場當日即跌停（可能賣不掉）的交易，依 |貢獻| 排序">&#9432;</span></h2>'
        + '<table class="tt"><thead><tr><th>代號 / 名稱</th><th>進場</th><th>出場</th><th>報酬</th><th>貢獻 (NAV)</th><th>旗標</th></tr></thead><tbody>';
      for (const i of shown) {
        const name = T.name && T.name[i] ? ' <span class="tsub">' + esc(T.name[i]) + "</span>" : "";
        const flags = flagBadges(T, i);
        const cb = contribOf(i);
        html += "<tr><td>" + esc(T.stock[i]) + name + "</td><td>" + (T.entry[i] || "–") + "</td>"
          + "<td>" + (T.exit[i] || '<span class="tsub">持有中</span>') + "</td>"
          + "<td>" + (T.ret[i] == null ? "–" : fmtSignPct(T.ret[i], 1)) + "</td>"
          + "<td>" + (cb == null ? "–" : fmtSignPct(cb, 2)) + "</td><td>" + flags + "</td></tr>";
      }
      html += "</tbody></table>";
      if (hits.length > shown.length)
        html += '<div class="sub" style="margin-top:6px">僅顯示 |貢獻| 前 ' + shown.length + " 筆，共 " + hits.length + " 筆</div>";
    }
  }
  $("liq-list").innerHTML = html;
}

/* ---------- SECTION 6 交易顯微鏡 (collapsed) ---------- */
function renderMicroSummary() {
  const T = P.trades;
  $("micro-hint").textContent = T && T.ret.length
    ? "共 " + T.total + " 筆交易 · 點擊展開分布 / MAE / 模擬停損 / 明細"
    : "無交易資料";
}
function renderMicroCharts() {
  renderRetHist(); renderScatter(); renderStops(); renderTradesPanel(); renderStatGroups();
}
function renderRetHist() {
  const T = P.trades;
  if (!T || !T.ret.length) return;
  const rets = T.ret.filter(v => v != null);
  drawHist("hist", rets, (v) => fmtPct(v, 1));
  if (!rets.length) return;
  const sorted = [...rets].sort((a, b) => a - b);
  const q05 = sorted[Math.max(0, Math.floor(sorted.length * 0.05) - (sorted.length * 0.05 % 1 === 0 ? 1 : 0))];
  $("dist-note").textContent = q05 < 0
    ? "有 5% 的機率，單筆交易將有 " + fmtPct(-q05, 1) + " 以上的虧損"
    : "95% 的交易報酬高於 " + fmtSignPct(q05, 1);
}
function renderScatter() {
  const T = P.trades;
  if (!T || !T.ret.length) return;
  const c = $("scatter");
  if (!c.clientWidth) return;
  const f = frame(c);
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
      '<div class="t">' + esc(T.stock[i]) + " · " + (T.entry[i] || "–") + " → " + (T.exit[i] || "持有中") + "</div>"
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
  if (!T) { $("stops").innerHTML = '<div class="sub">無交易資料</div>'; return; }
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
function renderTradesPanel() {
  const T = P.trades;
  if (!T) { $("ttable-wrap").innerHTML = '<div class="sub">無交易資料</div>'; return; }
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
    const name = T.name && T.name[i] ? ' <span class="tsub">' + esc(T.name[i]) + "</span>" : "";
    const ret = T.ret[i] == null ? '<span class="tsub">–</span>'
      : '<span class="badge ' + (T.ret[i] >= 0 ? "up" : "dn") + '">' + (T.ret[i] >= 0 ? "▲ " : "▼ ") + fmtSignPct(T.ret[i], 1) + "</span>";
    const px = (v) => v == null ? "" : '<div class="tsub">$' + v + "</div>";
    const exitCell = T.exit[i] ? T.exit[i] + px(T.exit_px[i]) : '<span class="tsub">持有中</span>';
    const flags = flagBadges(T, i);
    html += "<tr><td>" + esc(T.stock[i]) + name + "</td><td>" + ret + "</td>"
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
function renderStatGroups() {
  if (!P.stat_groups) { $("stats-card").style.display = "none"; return; }
  $("stats-cols").innerHTML = P.stat_groups.map(g =>
    '<table class="stats kv">' + g.map(([k, v]) => "<tr><td>" + k + "</td><td>" + v + "</td></tr>").join("") + "</table>"
  ).join("");
}

/* ---------- info tooltips (ⓘ) ---------- */
$("app").addEventListener("mouseover", (ev) => {
  const n = ev.target.closest("[data-tip]");
  if (n && n.dataset.tip) showTip('<div class="t">' + esc(n.dataset.tip) + "</div>", ev.clientX, ev.clientY);
});
$("app").addEventListener("mouseout", (ev) => {
  if (ev.target.closest("[data-tip]")) hideTip();
});

/* ---------- boot ---------- */
function renderAll() {
  renderHeader();
  renderVerdict();
  renderPerfSection();
  renderStructureSection();
  renderDDSection();
  renderLiveSection();
  renderMicroSummary();
  if ($("micro").open) renderMicroCharts();
}
$("micro").addEventListener("toggle", () => { if ($("micro").open) renderMicroCharts(); });
renderAll();
let rsTimer = null;
addEventListener("resize", () => {
  clearTimeout(rsTimer);
  rsTimer = setTimeout(renderAll, 150);
});
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", renderAll);
</script>
</body>
</html>
"""
