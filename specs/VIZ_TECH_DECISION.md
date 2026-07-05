# Visualization Tech Decision — 2026-07-04

Supersedes the "use Plotly" line in REPORT_IMPLEMENTATION_PLAN.md Stage 3.
Based on a candidate survey (uPlot, ECharts 6, lightweight-charts 5, Observable
Plot, vega-lite, plotly.js, Chart.js, LightningChart, hand-rolled SVG) against:
Python decoupling, interactivity, 100k-pt performance, chart-type flexibility,
bundle/self-containment, license/health. Numbers as of July 2026.

## Decision (tiered)

### Tier 1 — library offline report (`viz.py` + `_viz_template.py`): keep the hand-rolled SVG
- 0 KB extra, no CDN, works in `srcdoc` iframes; covers crosshair/tooltip,
  range presets, log scale, dark mode, heatmap, scatter, histogram in ~400 lines.
- **Tripwires — switch when any fires:**
  1. drag-zoom/pan or brush-linked equity+drawdown becomes a requirement;
  2. candlestick drill-down (OHLC + volume + trade markers + pan);
  3. >~20k in-view points or >8k interactive scatter nodes
     (`MAX_EMBEDDED_TRADES=8000` is already the symptom).
- **Escape hatch = uPlot** (MIT, 50.8 KB raw min, canvas, cursor-sync,
  OHLC paths, 166k pts in ~30 ms): inline `uPlot.iife.min.js` into the template;
  file stays single-file/offline. Do NOT inline ECharts (~350-400 KB min
  tree-shaken, ~1.1 MB full) into per-report files.

### Tier 2 — standalone web service frontend: Apache ECharts 6 (tree-shaken)
- ~100 KB gz via `echarts/core` + explicit imports + CanvasRenderer (pin ≥6.1).
- Native: candlestick, heatmap, scatter, histogram, `axisPointer` crosshair,
  `dataZoom` (inside+slider) with linked panes (`xAxisIndex: [0,1]`), log axis,
  v6 design-token theming with OS dark mode. Apache-2.0, ASF governance.
- **LLM synergy**: charts are pure-JSON `option` objects — the NL-backtest agent
  can safely generate/patch chart config as data, unlike imperative uPlot code.
- Rejected: plotly.js (finance bundle 400 KB gz / 1.2 MB raw, SVG degrades
  20k-100k pts, no capability gain), Chart.js (financial plugin unmaintained),
  Observable Plot (no zoom/pan), vega-lite (weight w/o finance ergonomics),
  LightningChart (cost unjustified).
- Optional hybrid: TradingView lightweight-charts 5.x (61 KB gz, Apache-2.0 +
  attribution NOTICE, keep `attributionLogo: true`) for a dedicated
  trading-terminal-style price page only — it lacks scatter/heatmap/distribution.

## Shared payload contract (`viz.report_data()`)

Freeze as versioned JSON Schema (`specs/report_payload.schema.json`), add
`"schema": 1`; the offline template, the service frontend, and the MCP server
all consume the same object. Changes worth making before the service exists:
- `stat_groups` embeds pre-formatted display strings — document as
  non-contractual derived block (real frontends format from `stats`).
- Make `daily` multi-series-ready: add `daily.series: [{name, creturn}]`
  (single element today; keep `creturn` as alias) for future multi-strategy
  comparison.
- Intraday (future) will use epoch-ms `ts` instead of extending `dates`.
- Keep drawdown client-derived (peak within visible window) — write the rule
  into the schema doc.
- Add `generated_at` + library `version` into the payload itself.
