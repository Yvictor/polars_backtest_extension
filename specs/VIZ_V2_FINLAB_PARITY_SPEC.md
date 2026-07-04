# VIZ V2 — FinLab-Parity One-Page HTML Report Specification

Status: research complete, implementation not started.
Target: `polars_backtest.viz` schema 3 — a single self-contained HTML page (no CDN)
that matches or exceeds the completeness of FinLab's `report.display()` dashboard.

Sources mined (all verified against shipped code, not docs):

- `finlab/core/report.cpp` + `metrics.cpp` (Cython-embedded `.pyx` source reconstructed) —
  exact metric formulas and the `get_metrics()` payload shape.
- `finlab/core/everything.js` (the `<strategy-analytic>` Lit web component actually rendered
  by `report.display()` / `_repr_html_`) — **exact pass/fail thresholds (`Wr` object, line 1996)
  and the exact 0–100 dimension score function (`Ir`, line 1997)**, the full zh-tw label
  inventory, formatters and units.
- `finlab/core/dashboard.py` — HTML/iframe embedding, payload contract
  (`reportJson = {timestamps, strategy, benchmark, trades, metrics}` + `positionJson`).
- `finlab/analysis/{liquidityAnalysis,drawdownAnalysis,maeMfeAnalysis,alphaBetaAnalysis,periodStatsAnalysis}.py`.
- `finlab/ffn_core.py` (`calc_stats`, `drawdown_details`) — definitions behind
  `avg_drawdown`, `avg_drawdown_days`, `twelve_month_win_perc`.
- User screenshots of finlab.tw strategy page (ground truth for layout).
- Web/PyPI cross-check: finlab **2.0.14** wheel bundle (independent fetch) — same `Wr`/`Ir`
  logic (there named `qf`/`Vf`), plus `quality_need*` i18n tooltip strings that ARE the
  "需 ≥ 15%" captions, chip color rule, and a non-TW liquidity rule (§2). Official docs
  (www.finlab.finance/docs; doc.finlab.tw redirects there) do **not** document the scoring
  system or thresholds anywhere public, and no 策略上架/marketplace criteria are published —
  the shipped frontend is the only authoritative source.
- Our engine: `polars_backtest/src/report.rs` (`get_metrics`, `capacity`, `capacity_by_date`),
  `python/polars_backtest/viz.py` (schema-2 `report_data`), trades schema.

---

## 1. Page layout (one page, tabbed sections)

```
┌────────────────────────────────────────────────────────────────────┐
│ Header: title · param chips (resample/fee/tax/SL/TP/TS/trade@) ·   │
│         date range · engine version                                │
├────────────────────────────────────────────────────────────────────┤
│ HERO KPI BAR (4 tiles)                                             │
│  年度回報 +36.5%   最大回檔 -38.8%   夏普值 1.38   逐筆交易勝率 50.3%│
├────────────────────────────────────────────────────────────────────┤
│ SCORE CHIPS (5 clickable, 0-100 + colored ring)                    │
│  [獲利 80] [風險 33] [報酬比 80] [勝率 40] [流動性 94]             │
├────────────────────────────────────────────────────────────────────┤
│ METRIC TILE ROW (metrics of the selected dimension)                │
│  each tile: ✓/✗ dot · label · value+unit · caption "需 ≥ 15%" · ⓘ  │
├────────────────────────────────────────────────────────────────────┤
│ PANEL TABS (depend on selected dimension, see §4)                  │
│  獲利:   歷史績效 | 月報酬 | 交易 | 年度比較                        │
│  風險:   虧損歷史 | 交易                                            │
│  報酬比: 夏普與大盤比較 | 極端風報比 | 波動性                       │
│  勝率:   報酬分布 | 交易最大偏移 (MAE/MFE + 模擬停損/停利)          │
│  流動性: 胃納量 | 流動性風險 | 交易                                 │
├────────────────────────────────────────────────────────────────────┤
│ Footer: schema/version · riskfree assumption · data caveats        │
└────────────────────────────────────────────────────────────────────┘
```

Interactions:

- Clicking a score chip switches both the metric tile row and the panel tab set below
  (finlab behavior). Default dimension: 獲利.
- Every metric tile and panel title carries an ⓘ tooltip with the description text (§3 table).
- All charts share one hover tooltip implementation; time charts share x-cursor.
- Light/dark via `prefers-color-scheme` (finlab defaults dark; we keep our current behavior).

FinLab reference DOM: `<strategy-analytic>` web component inside an auto-resizing iframe
(`postMessage {frameHeight, tab}`); our v1 iframe approach in `viz.show()` is equivalent.

---

## 2. Dimension scores (0–100) — exact FinLab algorithm

Reverse-engineered from `everything.js` line 1996–2004 (function `Ir`, threshold table `Wr`).
This is deterministic and fully reproducible:

```
for each dimension d in {profitability, risk, ratio, winrate, liquidity}:
    passed(d) = number of metrics m in Wr[d] whose predicate passes on metrics[d][m]
    if d == liquidity and capacity passes: passed(d) += 10      # capacity bonus
    total(d)  = |Wr[d]|            # 5, 6, 5, 5, 6
    if d == liquidity: total(d) += 10                            # → 16
    score(d)  = round(100 * passed(d) / total(d))
```

Verified against screenshot: 獲利 4/5=80, 風險 2/6=33, 報酬比 4/5=80, 勝率 2/5=40,
流動性 15/16=94 (capacity passed ⇒ 11 + 4 others). All match.

Additional rules confirmed in the finlab 2.0.14 bundle:

- A metric only counts as passed if its value is a **finite number** and the predicate
  holds (NaN/undefined ⇒ fail). This is the basis of our missing-input rule in §5.
- **Non-TW markets: liquidity score is forced to 0** (market name not in
  {TW_STOCK, TSE_OTC, TSE, OTC}). We generalize: if no liquidity inputs exist, show the
  chip as 無資料 rather than a hard 0 (finlab-compat mode keeps the 0).
- Weight summary: 獲利/報酬比/勝率 checks are worth 20 pts each, 風險 ~16.7 pts,
  流動性 capacity alone 68.75 pts (11/16) and the other five 6.25 pts each.

Known FinLab quirk (documented, decision required): `Wr.risk` contains a `volatility < 0.2`
check, but FinLab's `get_metrics()` payload puts `volatility` under the `ratio` section, so
`metrics.risk.volatility` is `undefined` and the check **always fails** in FinLab production
(risk score max = 5/6 = 83). Our default: evaluate volatility against the real value
(corrected mode); optionally expose `scoring="finlab-compat"` that force-fails it for
bit-identical scores. Either way the denominator is 6.

Chip color ramp (FinLab-exact, from 2.0.14 bundle): score ≥ 70 → good (green),
40–69 → neutral (amber), < 40 → bad (red); ring fill = score/100.

---

## 3. Metric definitions, thresholds and captions (complete table)

All thresholds are the exact `Wr` predicates. "Caption" is the 需-string shown under the
tile — in finlab 2.0.14 these ship verbatim as `quality_need*` i18n strings (e.g. 需 ≥ 15%,
需介於 0–0.8, 需 < 40 天, 需 ≥ 50 萬); older bundles render ✓/✗ circle icons instead — we
render both dot and caption. Riskfree rate = 2% annually wherever Sharpe/Sortino need it
(FinLab default).

### 3.1 獲利 profitability (5 metrics)

| key | label | formula (FinLab source) | engine source | pass condition | caption | unit |
|---|---|---|---|---|---|---|
| annualReturn | 年度回報 | CAGR of daily-resampled creturn: `(last/first)^(365.25/days)-1` (ffn `cagr`) | `get_metrics` `annualReturn` | `> 0.15` | 需 ≥ 15% | % |
| alpha | Alpha | `cagr(strategy) − β·cagr(benchmark)`, daily returns | `get_metrics` `alpha` (needs benchmark) | `> 0.10` | 需 ≥ 10% | % |
| beta | Beta | `cov(r_s, r_b)/var(r_b)`, daily returns | `get_metrics` `beta` (needs benchmark) | `> 0 && < 0.8` | 需介於 0–0.8 | – |
| avgNStock | 平均持有 | mean daily count of non-zero positions | `get_metrics` `avgNStock` | `>= 5` | 需 ≥ 5 檔 | 檔 |
| maxNStock | 最多持有 | max daily count of non-zero positions | `get_metrics` `maxNStock` | `<= 20` | 需 ≤ 20 檔 | 檔 |

ⓘ descriptions (zh-tw, from component i18n): 年度回報「策略的年度回報」· Alpha「相對於基準的
策略的風險調整後表現的衡量」· Beta「策略對市場變動的敏感性的衡量」· 平均持有「投資組合中
持有的平均股票數」· 最多持有「投資組合中持有的最大股票數」.

### 3.2 風險 risk (5 displayed + volatility in scoring)

| key | label | formula | engine source | pass condition | caption | unit |
|---|---|---|---|---|---|---|
| maxDrawdown | 最大回檔 | min of drawdown series (ffn `max_drawdown`) | `get_metrics` `maxDrawdown` | `> -0.30` | 需 < 30% | % |
| avgDrawdown | 平均回檔幅度 | mean depth over drawdown episodes (ffn `drawdown_details`) | `get_metrics` `avgDrawdown` | `> -0.10` | 需 < 10% | % |
| avgDrawdownDays | 平均回檔時間 | mean calendar-day length of drawdown episodes | `get_metrics` `avgDrawdownDays` | `< 40` | 需 < 40 天 | 天 |
| valueAtRisk | Value at Risk | 5% quantile of **monthly** returns | `get_metrics` `valueAtRisk` | `> -0.07` | 需 < 7% | % |
| cvalueAtRisk | Conditional VaR | mean of monthly returns below VaR | `get_metrics` `cvalueAtRisk` | `> -0.10` | 需 < 10% | % |
| volatility | 波動性 (scoring only, see §2 quirk) | annualized std of daily returns | `get_metrics` `volatility` (ratio section) | `< 0.20` | 需 < 20% | % |

ⓘ: 最大回檔「從高點到低谷的最大百分比下降」· 平均回檔幅度「從高點到低谷的平均百分比下降」·
平均回檔時間「平均回撤天數」· VaR「給定信心區間的指定期間內預期的最大損失」·
CVaR「發生指定不良事件後的預期損失」· 波動性「策略回報的標準差」.

### 3.3 報酬比 ratio (5 scored)

| key | label | formula | engine source | pass condition | caption | unit |
|---|---|---|---|---|---|---|
| sharpeRatio | 夏普值 | ffn `daily_sharpe`, rf = 2% | `get_metrics` `sharpeRatio` | `> 1.3` | 需 > 1.3 | – |
| sortinoRatio | Sortino Ratio | ffn `daily_sortino`, rf = 2% | `get_metrics` `sortinoRatio` | `> 1.8` | 需 > 1.8 | – |
| calmarRatio | Calmar Ratio | CAGR / \|maxDrawdown\| | `get_metrics` `calmarRatio` | `> 0.9` | 需 > 0.9 | – |
| profitFactor | Profit Factor | \|Σ winning trade returns / Σ losing trade returns\| (1 if no trades) | `get_metrics` `profitFactor` | `> 1.5` | 需 > 1.5 | – |
| tailRatio | Tail Ratio | \|q95 / q05\| of daily returns (1 if q05=0) | `get_metrics` `tailRatio` | `> 1` | 需 > 1 | – |
| volatility | 策略報酬率波動 | displayed in FinLab payload under ratio; scored under risk | `get_metrics` `volatility` | (risk) | – | % |

ⓘ: 夏普「風險調整後表現的衡量」· Sortino「只考慮下行波動性的風險調整後表現的衡量」·
Calmar「年度回報與最大回撤的比率」· Profit Factor「總利潤與總損失的比率」·
Tail Ratio「右尾（贏）與左尾（輸）的比率」.

### 3.4 勝率 winrate (5 metrics)

| key | label | formula | engine source | pass condition | caption | unit |
|---|---|---|---|---|---|---|
| winRate | 逐筆交易勝率 | share of trades with `return > 0` | `get_metrics` `winRate` | `> 0.55` | 需 ≥ 55% | % |
| m12WinRate | 使用策略12個月勝大盤 | ffn `twelve_month_win_perc`: share of rolling-12M windows beating benchmark | `get_metrics` `m12WinRate` (needs benchmark) | `> 0.70` | 需 ≥ 70% | % |
| expectancy | 期望值 | mean trade `return` | `get_metrics` `expectancy` | `> 0.02` | 需 ≥ 2% | % |
| mae | 最大不利偏移 | mean of trades `mae` | `get_metrics` `mae` | `> -0.10` | 需 < 10% | % |
| mfe | 最大有利偏移 | mean of trades `gmfe` | `get_metrics` `mfe` | `> 0.10` | 需 ≥ 10% | % |

ⓘ: 勝率「盈利交易的百分比」· 12M勝率「12個月滾動勝率」· 期望值「每筆交易預期贏得（或損失）
的平均金額」· MAE「最大不利運動，或交易盈利前的最大損失」· MFE「最大有利運動，或交易變成
損失前的最大利潤」.

### 3.5 流動性 liquidity (6 metrics; capacity has 11× weight)

| key | label | formula (FinLab) | engine source | pass condition | caption | unit |
|---|---|---|---|---|---|---|
| capacity | 胃納量 | 10th percentile over trades of `(turnover@entry·5%/|position| + turnover@exit·5%/|position|)/2` | `get_metrics` `capacity` (needs `trading_values`) | `> 500_000` | 需 > 50 萬 | 萬 (value/1e4) |
| disposalStockRatio | 處置股 | max(entry, exit) share of trades whose stock is 處置股 at **signal date** (TW `etl:is_flagged_stock` bit 0x2) | **NOT in engine** — Python via user-supplied flag frame | `< 0.05` | 需 < 5% | % |
| warningStockRatio | 警示股 | same, bit 0x1 | **NOT in engine** — Python | `< 0.05` | 需 < 5% | % |
| fullDeliveryStockRatio | 全額交割股 | same, bit 0x4 | **NOT in engine** — Python | `< 0.05` | 需 < 5% | % |
| buyHigh | 買在漲停 | max(entry, exit) share of adverse fills at limit-up. FinLab proxy: `pct_change@trade_date > 0.95·range` (range = 10% after 2015-06-01 else 7%), direction-aware | `get_metrics` `buyHigh` (uses real `limit_up` column — stricter than FinLab's 95% proxy) | `< 0.05` | 需 < 5% | % |
| sellLow | 賣在跌停 | symmetric at limit-down | `get_metrics` `sellLow` | `< 0.05` | 需 < 5% | % |

Additional FinLab LiquidityAnalysis aggregates (shown in its analysis table, not scored):
`low_volume_stocks` = share of trades with `volume@entry/exit < 200_000` shares;
`low_turnover_stocks` = share with `turnover@entry/exit < NT$1_000_000`. Both computable
in Python from `input_df` (`volume` if supplied, `trading_value`). We show them in the
流動性 panel (§4.5).

ⓘ: 胃納量「不影響市場價格可以部署的最大資本金額」· 處置股/警示股/全額交割股 descriptions as
in i18n table · 買在漲停「購買高價股票的傾向」· 賣在跌停「賣出低價股票的傾向」.

### 3.6 Hero KPI bar

| tile | label | source | format |
|---|---|---|---|
| 1 (large) | 年度回報 | `metrics.annualReturn` | signed %, green/red by sign |
| 2 | 最大回檔 | `metrics.maxDrawdown` | signed %, red |
| 3 | 夏普值 | `metrics.sharpeRatio` | 2 decimals |
| 4 | 逐筆交易勝率 | `metrics.winRate` | % 1 decimal |

Formatting rules (from `MetricDisplay`): percentages ×100, 1 decimal, `+` sign for positives;
counts as integers with 檔; capacity as `value/1e4` with 萬; days with 天; `0 → "-"` for nStock.

---

## 4. Panel-by-panel spec

### 4.1 獲利 → tab 歷史績效 (historical performance)  [default view]

- **Cumulative return area chart**: strategy (brand color, area fill) vs 大盤 benchmark
  (gray line). y = creturn − 1 in %, x = date. Log/linear toggle (our v1 already has it —
  keep; exceeds FinLab). Range selector (全部 / 近年 or 1y/3y/5y/YTD).
- **Drawdown strip** below, sharing x-axis: strategy drawdown area (red).
- **Yearly-return chip row**: `全部` chip + one chip per year `2007 -7.1%`, background
  green/red by sign; clicking a year zooms the chart to that year. Data = YTD column of the
  return table (strategy); FinLab shows 大盤年報酬 on hover (`profitability_benchmarkYearlyReturn`).
- Data: `daily.dates`, `daily.creturn`, `daily.benchmark`, `return_table` (+ new
  `benchmark_return_table`), drawdown computed in JS.

### 4.2 獲利 → tab 月報酬 (monthly returns)

- **Monthly heatmap**: year rows × 12 month columns, diverging red-white-green
  (FinLab uses RdBu_r with cmid=0), cell text = % 1 decimal. (v1 has this.)
- **Header stats**: 平均月報酬 `mean(monthly returns)` and 月勝率
  `count(month > 0)/count(months)` (labels `profitability_avgMonthlyReturn`,
  `profitability_monthlyWinRatio`). Computed in JS from `return_table`.

### 4.3 獲利/風險/流動性 → tab 交易 (trade table, FinLab "持股報酬 stockList")

Columns (FinLab order and labels):

| col | label | source field | render |
|---|---|---|---|
| 名稱代號 | stock id + display name | `trades.stock` + `trades.name` (symbol_names) | `3122 笙泉`, monospace id |
| 報酬 | return | `trades.ret` | badge, green/red + ▲/▼ arrow, % 1dp |
| 進場 | entry | `trades.entry` + `trades.entry_px` | date + `$price` second line; sortable (default sort: 進場 desc) |
| 出場 | exit | `trades.exit` + `trades.exit_px` | date + `$price`; open trades show `–` |
| 持倉 | position | `trades.pos` | % 0–1dp |
| 指標 | indicators | `trades.mae`, `trades.gmfe` | two mini-badges `MAE -3.2%` `GMFE +8.1%` |
| (ours) | 流動性 | `trades.lim_entry`, `trades.lim_exit` | ⚠ chips 漲停買 / 跌停賣 when true |

Extra sortable fields available in payload: `pdays` (持倉天數), `mdd` (trade 最大回檔),
`bmfe` (虧損前有利偏移 — add to payload, see §5). Client-side sort on every column;
paginate/virtualize > 200 rows. FinLab embeds only `trades.tail(500)`; we embed up to
`MAX_EMBEDDED_TRADES = 8000` subsampled — keep, but switch to "most recent N" to match
FinLab semantics for the table while keeping full-range sampling for scatter plots.

### 4.4 獲利 → tab 年度比較 (yearly vs benchmark)

- **Grouped bar chart per year**: strategy yearly return vs 大盤 yearly return.
- Header: 贏大盤 `X / N 年` (years where strategy > benchmark) and 超額報酬
  (mean or per-year strategy − benchmark; FinLab labels `profitability_yearlyWinRate`,
  `profitability_exceedReturn`).
- Data: strategy `return_table` YTD + new `benchmark_return_table` YTD (computed in Python
  from benchmark series).

### 4.5 風險 → tab 虧損歷史 (drawdown history)

- **Drawdown depth chart**: strategy drawdown (red area/line) vs 大盤 drawdown (gray),
  shared x, y gridlines at e.g. −18%/−36%/−54% (auto thirds of min).
- **Episode selector**: clicking an episode (or a 跌幅排名 row) highlights the span and
  shows the caption `回檔幅度 2016 01M  -38.8%  844天` — label = trough year+month,
  depth, and total days (peak→recovery; ongoing episodes show days-to-date).
- **跌幅排名 ranked bar list**: top episodes sorted by depth
  (`2016 01M -38.8%`, `2025 07M -35.9%`, …), with a 策略/大盤 segmented toggle switching
  the list (and chart emphasis) between strategy and benchmark episodes.
  FinLab's DrawdownAnalysis keeps top-5 largest + top-5 longest; we ship top-10 of each.
- **(parity+) 再次創新高時間排名** (`metrics_risk_newHighTimeRank`): ranked list of the
  longest drawdown durations in 天 — 策略前十大 vs 大盤前十大. Same episode data sorted
  by `days` instead of `depth`.
- Data: `dd_episodes`, `benchmark_dd_episodes` (already in schema 2; extend to top-10 by
  depth AND by days), drawdown series computed in JS from `daily`.

### 4.6 報酬比 → panels

- **夏普與大盤比較** (`metrics_ratio_yearlySharpeRatio`): per-year Sharpe bars, strategy vs
  大盤; header 大於大盤：X 年 / 大於大盤的時間：Y%. Computed in Python (new
  `yearly_stats` payload: per-year sharpe for strategy and benchmark).
- **極端風報比與大盤比較** (`yearlyTailRatio` / `rollingTailRatio`): per-year tail ratio
  bars strategy vs 大盤 (tail ratio = −q95/q05 of daily returns within the year, per
  PeriodStatsAnalysis).
- **策略報酬率波動 (Volatility)**: rolling annualized volatility line (e.g. 60d window)
  strategy vs 大盤, plus the scalar `volatility` tile. Computed in JS from daily series.
- **(2.0.14 addition) 策略與大盤相關性 (correlation)**: rolling correlation of daily
  returns strategy vs 大盤. Computed in JS from daily series. Optional for parity.

### 4.7 勝率 → tab 報酬分布 (return distribution)

- **Histogram of per-trade returns**, win bars green / loss bars red (v1 has a variant).
- **Info sentence**: 「有 5% 的機率，交易將有 X % 以上的虧損」 where X = −q05 of trade
  returns (labels `distributionInfo1/2`). Computed in JS from `trades.ret`.

### 4.8 勝率 → tab 交易最大偏移 (MAE/MFE + simulated stops)

- **模擬停損 (simulated stop-loss)**: slider for SL threshold; recompute in JS from
  per-trade `mae`: trades with `mae ≤ −SL` are clipped to `return = −SL` (fees ignored),
  show 停損設定 / 停損造成的額外盈虧 (Δ mean return ×100) / 停損的交易比例.
- **報酬與最大不利偏移 scatter**: x = MAE, y = return, point color by win/loss; SL line.
  (v1 has MAE-vs-return scatter — extend with slider.)
- **模擬停利 (simulated take-profit)**: same with `gmfe ≥ TP` → `return = TP`; header
  停利設定 / 停利造成的額外盈虧 / 停利的交易比例; scatter x = GMFE (報酬與最大有利偏移).
- Data: `trades.mae`, `trades.gmfe`, `trades.ret` (all present).
- (Optional parity+ from MaeMfeAnalysis: edge ratio = Σgmfe/Σ|mae|, distribution stats per
  win/loss cohort for return/mae/bmfe/gmfe/mdd/pdays.)

### 4.9 流動性 → panels

- **投資組合胃納量 (portfolio capacity)**: big number `capacity` in 萬 + threshold marker
  at 50萬 + ⓘ「安全流動性交易的佔比 / 投資總資金」context. Parity+: sparkline of
  `report.capacity_by_date()` (engine already provides it — FinLab does NOT have this).
- **流動性風險 table** (from LiquidityAnalysis): rows = entry/exit, columns =
  買在漲停 buy_high · 賣在跌停 sell_low · 低成交量 low_volume (<200k shares) ·
  低成交金額 low_turnover (<NT$1M) · 警示股 · 處置股 · 全額交割股 (TW only, if flag data
  supplied). Cells = % of trades, heat-shaded (FinLab: YlGnBu, vmax 50%).
- **買進/出場成交量門檻** (FinLab 2.0.14 panels `entryVolume`/`exitVolume`): share of
  trades whose entry-day / exit-day volume (or turnover) is below the threshold, with an
  adjustable threshold slider — computable in JS from per-trade volume/turnover columns.
- **Per-trade flags**: the 交易 tab already shows ⚠ 漲停買/跌停賣 chips per trade
  (`lim_entry`/`lim_exit`); add 低量 chip when volume/turnover below thresholds.
- Aggregates in `trade_summary`: `buy_high_n/ratio`, `sell_low_n/ratio` (already computed in
  viz.py) + new `low_volume_ratio_entry/exit`, `low_turnover_ratio_entry/exit`.
- Interpretation guidance from official docs (show as ⓘ text): buy_high 或 sell_low > 10%
  「可能無法順利成交」; low_volume_stocks > 20% 「大資金可能無法全部執行」; 處置股 > 5%
  「策略可能專挑高風險標的」.

---

## 5. Data payload mapping (schema 3)

Payload keys → source. **(P)** = compute in Python `viz.report_data`, **(JS)** = derive
client-side, **(R)** = Rust engine, **(GAP)** = not available yet.

| payload field | source | status |
|---|---|---|
| `daily.dates/creturn` | (R) `report.daily_creturn()` | ✅ schema 2 |
| `daily.benchmark` | user-set `report.benchmark` joined on dates | ✅ schema 2 (user must set benchmark; without it: alpha/beta/m12WinRate/年度比較/大盤 overlays are hidden) |
| `metrics.*` (all §3 keys) | (R) `get_metrics()` — sections backtest/profitability/risk/ratio/winrate/liquidity; columns `annualReturn, alpha, beta, avgNStock, maxNStock, maxDrawdown, avgDrawdown, avgDrawdownDays, valueAtRisk, cvalueAtRisk, sharpeRatio, sortinoRatio, calmarRatio, volatility, profitFactor, tailRatio, winRate, m12WinRate, expectancy, mae, mfe, buyHigh, sellLow, capacity` | ✅ engine parity for 24/27 finlab metrics |
| `scores` {profitability, risk, ratio, winrate, liquidity} | (P) §2 formula over `metrics` | 🆕 trivial |
| `checks` per metric {value, pass, caption} | (P) §3 tables | 🆕 trivial |
| `return_table` (strategy) | (R) `get_return_table()` | ✅ schema 2 |
| `benchmark_return_table` | (P) resample benchmark creturn monthly/yearly | 🆕 Python |
| `yearly_compare` {year, strat, bench, excess} | (P) from the two return tables | 🆕 Python |
| `yearly_stats` {year, sharpe_s, sharpe_b, tail_s, tail_b} | (P) from daily series | 🆕 Python |
| `dd_episodes` / `benchmark_dd_episodes` (top-10 by depth, incl. start/trough/end/days/recovery_days) | (P) `_drawdown_episodes` | ✅ schema 2 (extend: also top-10 by days for 新高時間排名) |
| drawdown series | (JS) from creturn/benchmark | ✅ |
| `trades.{stock,name,entry,exit,ret,mae,gmfe,mdd,pdays,pos,entry_px,exit_px,lim_entry,lim_exit}` | (R) trades + (P) `_with_limit_flags(input_df)` | ✅ schema 2 |
| `trades.bmfe` | (R) trades has `bmfe` | 🆕 add column to payload |
| `trades.low_vol_entry/exit`, `low_to_entry/exit` | (P) join `input_df.volume`/`trading_value` at entry/exit dates | 🆕 Python (volume column optional) |
| `trade_summary.buy_high_*`, `sell_low_*` | (P) | ✅ schema 2 |
| `trade_summary.low_volume_ratio`, `low_turnover_ratio` | (P) | 🆕 Python |
| `liquidity_table` (entry/exit × categories) | (P) | 🆕 Python |
| capacity sparkline | (R) `report.capacity_by_date()` | ✅ engine (parity+) |
| 處置股/警示股/全額交割股 ratios | external TW dataset (`etl:is_flagged_stock` bitmask 0x1警示/0x2處置/0x4全額交割, evaluated at **entry_sig_date/exit_sig_date**) | ⛔ GAP — accept optional `flags_df(date, symbol, flag_bits)` argument in `report_data`; omit tiles (and drop from score denominator, see below) when absent |
| `smallCapRatio`, 市值門檻/交易標的市值 panels | needs per-symbol market value | ⛔ GAP — optional `market_value_df`; FinLab shows but does not score it |
| simulated SL/TP | (JS) from trades mae/gmfe | ✅ derivable |
| monthly win ratio / avg monthly return | (JS) from return_table | ✅ derivable |
| hero/params/stats/stat_groups | existing schema 2 | ✅ |

Scoring with missing inputs (deterministic rule): a metric whose value is `null`
(no benchmark → alpha/beta/m12WinRate; no flags_df → 處置/警示/全額交割; no
trading_values → capacity) counts as **failed** in finlab-compat mode (matches FinLab
JS: predicate on `undefined` → false). In default mode we instead **drop it from both
numerator and denominator** and mark the tile "無資料"; liquidity keeps the capacity
weight rule (11/(remaining+10)) only when capacity is evaluable. Both modes are pure
functions of the payload; the report footer states which mode rendered the scores.

---

## 6. Engine gaps discovered (ranked)

1. **TW regulatory flags (處置股/警示股/全額交割股)** — FinLab pulls a proprietary dataset
   (`etl:is_flagged_stock`). Engine/report cannot derive this from price data. Plan:
   optional `flags_df` input to `viz.report_data`; ratios computed in Python at signal
   dates. Without it we render 4-of-6 liquidity metrics (FinLab itself only adds these on
   TWMarket).
2. **Benchmark is optional/user-set** — alpha, beta, m12WinRate, all 大盤 overlays,
   年度比較, yearly sharpe/tail comparisons need `report.benchmark`. Engine supports it;
   docs must make it a first-class step (`report.benchmark = twii_df`).
3. **Per-year sharpe/tail/vol vs benchmark** — not in engine; compute in Python (§5
   `yearly_stats`). Cheap; no Rust change needed.
4. **`bmfe` not in the current trades payload** (engine computes it) — 1-line viz addition.
5. **Low-volume ratio** — engine has `trading_values` (turnover) for capacity but no share
   `volume`; the 200k-share check needs an optional `volume` column in `input_df`
   (turnover-only fallback: NT$1M check).
6. **Drawdown episode semantics** — FinLab (ffn) counts an episode as strictly
   below-previous-peak until full recovery to 0 drawdown; our `_drawdown_episodes` is
   equivalent (peak→trough→recovery) — no change, but label episodes as
   `{trough YYYY MM}M` for parity, and include ongoing episodes with `end=null`.
7. Not gaps (already engine-side, contrary to expectation): `avgDrawdownDays`
   (平均回檔時間), benchmark drawdown (computed in viz), per-trade at-limit flags
   (viz `_with_limit_flags` from `limit_up`/`limit_down`), capacity + `capacity_by_date`.

Where we exceed FinLab: real limit-price columns instead of the 95%-of-band proxy for
buyHigh/sellLow; capacity-by-date sparkline; log-scale equity toggle; full-trade-history
scatter (FinLab uploads only last 500 trades to the dashboard); documented deterministic
scoring incl. the volatility quirk.

---

## 7. Implementation staging (suggested)

1. **Payload (schema 3)**: `scores`, `checks` (+captions), `benchmark_return_table`,
   `yearly_compare`, `yearly_stats`, `trades.bmfe`, liquidity aggregates, optional
   `flags_df`/`volume` handling. Pure Python in `viz.py`; unit-test score function against
   the five screenshot values (80/33/80/40/94).
2. **Template layout**: hero bar, score chips, metric tile row with ✓/✗ + captions,
   dimension-switching, tab bars per dimension.
3. **Charts**: reuse v1 equity/drawdown/heatmap/scatter/histogram; add yearly compare bars,
   dd-episode ranked list + selector, yearly sharpe/tail bars, SL/TP sliders.
4. **Trade table**: FinLab column set, sorting, badges, liquidity chips.
5. **Polish**: ⓘ tooltips (zh-tw + en strings from §3), color ramp, footer provenance.
