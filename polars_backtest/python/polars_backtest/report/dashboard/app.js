/* The report page. All numbers come from the embedded payload; live ranges come from core.js.
   One store (S), and every action changes it and re-renders. */
"use strict";

/* ---------- payload ---------- */
const $ = (id) => document.getElementById(id);
const P = JSON.parse($("payload").textContent);
const decode = (c) => (c && c.d ? c.i.map((i) => (i == null ? null : c.d[i])) : c);
function rowsOf(table) {
  if (!table) return [];
  const names = Object.keys(table), cols = names.map((n) => decode(table[n])), out = [];
  const n = cols.length ? cols[0].length : 0;
  for (let i = 0; i < n; i++) {
    const row = {};
    names.forEach((name, k) => { row[name] = cols[k][i]; });
    out.push(row);
  }
  return out;
}
const forwardFill = (x) => { let last = null; return x.map((v) => (last = v == null ? last : v)); };
function backFilled(x) {
  const first = x.find((v) => v != null);
  return forwardFill(x).map((v) => (v == null ? first : v));
}

const DAYS = decode(P.curve.date), N = DAYS.length, LAST = N - 1;
const EQ = P.curve.creturn;
const BENCH = P.curve.bench_creturn ? backFilled(P.curve.bench_creturn) : null;
const GROSS = P.curve.gross_creturn || null;
const RF = P.run.riskfree_rate ?? 0.02;
const SUM = P.summary, PERF = SUM.performance;
const dayOf = (iso) => BT.dayIndex(DAYS, iso);
const RETS = EQ.map((v, i) => (i ? v / EQ[i - 1] - 1 : 0));
const SPAN_YEARS = BT.dayDiff(DAYS[0], DAYS[LAST]) / 365.25;
const pad2 = (n) => String(n).padStart(2, "0");

const MONTH_ROWS = rowsOf(P.monthly).map((r) => {
  const next = r.month === 12 ? `${r.year + 1}-01-01` : `${r.year}-${pad2(r.month + 1)}-01`;
  const a = dayOf(`${r.year}-${pad2(r.month)}-01`);
  const b = next > DAYS[LAST] ? LAST : dayOf(next) - 1;
  return { k: `${r.year}-${pad2(r.month)}`, y: r.year, m: r.month, r: r.ret, a, b };
});
const YEAR_ROWS = rowsOf(P.yearly).map((r) => {
  const a = dayOf(`${r.year}-01-01`), next = `${r.year + 1}-01-01`;
  const b = next > DAYS[LAST] ? LAST : dayOf(next) - 1;
  return { y: r.year, r: r.ret, bench: r.bench_ret ?? null, mdd: r.mdd, a, b };
});
const YEARS = YEAR_ROWS.map((y) => y.y);
const EPISODES = rowsOf(P.drawdowns).filter((r) => r.series === "strategy").sort((a, b) => a.rank - b.rank);

const TABLES = P.tables || {};
const TRADES = rowsOf(TABLES.trades).map((t, id) => ({
  ...t, id,
  e: t.entry_date ? dayOf(t.entry_date) : null,
  x: t.exit_date ? dayOf(t.exit_date) : null,
}));
const CLOSED = TRADES.filter((t) => t.status === "closed").sort((a, b) => b.x - a.x || b.id - a.id);
const OPEN = TRADES.filter((t) => t.status === "open");
const TRADE_DAYS = rowsOf(P.trade_days), HIST_DAYS = rowsOf(P.hist_days);
const HAS_TRADES = !!TABLES.trades;

const SYMBOL_ROWS = rowsOf(TABLES.symbol_index);
const NAMES = new Map(rowsOf(TABLES.symbols).map((r) => [r.symbol, r.name]));
SYMBOL_ROWS.forEach((r) => { if (r.name) NAMES.set(r.symbol, r.name); });

const byDate = (rows) => {
  const map = new Map();
  rows.forEach((r) => { if (!map.has(r.date)) map.set(r.date, []); map.get(r.date).push(r); });
  return map;
};
const has = (name) => P.coverage[name] === "present" || P.coverage[name] === "partial" || P.coverage[name] === "empty";
const FILL_ROWS = rowsOf(TABLES.fills), FILLS = byDate(FILL_ROWS);
const BLOCK_ROWS = rowsOf(TABLES.blocks), BLOCKS = byDate(BLOCK_ROWS);
const CASH = byDate(rowsOf(TABLES.cashflows));
const DAY_INDEX = new Map(rowsOf(P.day_index).map((r) => [r.date, r]));
const LEDGER = P.ledger_daily ? new Map(rowsOf(P.ledger_daily).map((r) => [r.date, r])) : null;
const ledgerSeries = (fn) => DAYS.map((d) => { const r = LEDGER && LEDGER.get(d); return r ? fn(r) : null; });
const HELD = (() => {
  let open = 0;
  return DAYS.map((d) => {
    const r = DAY_INDEX.get(d);
    if (r) open += r.n_entries - r.n_exits;
    return open;
  });
})();
const HAS_EXPOSURE = SUM.exposure.source === "ledger";
const COST_KEYS = ["commission", "tax", "surcharge", "borrow_cost", "hedge_cost"].filter((k) => SUM.costs[k] != null);
const ENTRY_V = SUM.entry_date ? BT.rollHold(EQ, SUM.entry_date.window_days) : null;
const ENTRY_B = SUM.entry_date && BENCH ? BT.rollHold(BENCH, SUM.entry_date.window_days) : null;
const CONC = SUM.concentration;
const CONC_STEPS = BT.flatBest(MONTH_ROWS.map((m) => m.r), CONC.k, EQ[LAST] / EQ[0], SPAN_YEARS);
const BEST_MONTHS = [...MONTH_ROWS].sort((a, b) => b.r - a.r).slice(0, CONC.k);

/* ---------- store and helpers ---------- */
const store = {
  get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* private window */ } },
};
const S = {
  view: "analysis",
  lang: store.get("report.lang") === "zh-TW" ? "zh-TW" : "en",
  theme: store.get("report.theme") || "dark",
  updown: store.get("report.updown") === "red-up" ? "red-up" : "green-up",
  cursor: null, pin: null, win: null, hoverYear: null, log: false, touch: false, scrubbed: false,
  open: new Set(), sub: {}, episode: 0, blockReason: null,
  month: null, hoverMonth: null, range: null, limitOnly: false, sort: "recent", page: 0,
  panel: null, hideNames: false, holdSort: "pnl",
};
const L = () => I18N[S.lang];
const t = (k, ...a) => {
  const v = L()[k];
  if (v === undefined) throw new Error("Missing string: " + k);
  return typeof v === "function" ? v(...a) : v;
};
const F = () => L().fmt;
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
const css = (k) => getComputedStyle(document.documentElement).getPropertyValue(k).trim();
const reduced = () => matchMedia("(prefers-reduced-motion:reduce)").matches;
const toneCls = (x) => (x < 0 ? "dn" : "up");
const symName = (code) => NAMES.get(code) || code;
const shown = (code) => (S.hideNames ? "•••" : symName(code));
const initials = (code) => {
  const words = symName(code).split(/\s+/).filter(Boolean);
  return esc((words.length > 1 ? words[0][0] + words[1][0] : symName(code).slice(0, 2)).toUpperCase());
};
const amount = (x) => {
  if (x == null) return "—";
  return P.capital ? F().money(x * P.capital) : F().pct(x, 2);
};
const ICON = {
  search: '<svg width="15" height="15" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"><circle cx="7" cy="7" r="4.6"/><path d="m10.6 10.6 3.4 3.4"/></svg>',
  gear: '<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.3"><circle cx="8" cy="8" r="2.1"/><path d="M8 1.6v2M8 12.400v2M1.600 8h2M12.400 8h2M3.500 3.500l1.400 1.400M11.100 11.100l1.400 1.400M3.500 12.500l1.400-1.400M11.100 4.900l1.400-1.400" stroke-linecap="round"/></svg>',
  globe: '<svg width="15" height="15" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.3"><circle cx="8" cy="8" r="6.2"/><path d="M1.8 8h12.400M8 1.800c2 2 2.700 4.100 2.700 6.200S10 12.200 8 14.200C6 12.200 5.300 10.100 5.300 8S6 3.800 8 1.800Z"/></svg>',
  x: '<svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"><path d="m2.500 2.500 7 7M9.500 2.500l-7 7"/></svg>',
  chev: '<svg class="chev" width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="m3 5 4 4 4-4"/></svg>',
  check: '<svg width="13" height="13" viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="m2.500 7.500 3 3 6-7"/></svg>',
  dcheck: '<svg width="17" height="13" viewBox="0 0 18 14" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="m1.500 7.500 3 3 6-7M8 10.500l1 .500 6-7.500"/></svg>',
  warn: '<svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round" stroke-linecap="round"><path d="M7 1.800 13 12.200H1Z"/><path d="M7 5.800v3M7 10.400v.1"/></svg>',
  left: '<svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="m9 2.500-4.500 4.500L9 11.500"/></svg>',
  right: '<svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="m5 2.500 4.500 4.500L5 11.500"/></svg>',
  arrows: '<svg width="14" height="10" viewBox="0 0 14 10" fill="none" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round"><path d="M1 5h12M3.500 2.500 1 5l2.500 2.500M10.500 2.500 13 5l-2.500 2.500"/></svg>',
  mark: '<svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M2 14.500 7 9l3.500 3L18 4"/><path d="M2 18h16" opacity=".4"/></svg>',
};
const GLYPH = {
  profitability: '<svg width="28" height="16" viewBox="0 0 28 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M1 13c3-1 4 0 6-2s3 1 5-1 3 0 5-3 3 1 4-1 3-3 5-5"/></svg>',
  risk: '<svg width="28" height="16" viewBox="0 0 28 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M1 3h26" opacity=".45"/><path d="M2 3l4 5 4-3 5 8 4-6 3 3 4-7"/></svg>',
  ratio: '<svg width="26" height="16" viewBox="0 0 26 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"><path d="M3 14a10 10 0 0 1 20 0" opacity=".6"/><path d="m13 14-6-3"/></svg>',
  winrate: '<svg width="26" height="14" viewBox="0 0 26 14"><g fill="currentColor"><circle cx="3" cy="4" r="1.500"/><circle cx="8" cy="4" r="1.500"/><circle cx="13" cy="4" r="1.500" opacity=".4"/><circle cx="18" cy="4" r="1.500"/><circle cx="23" cy="4" r="1.500" opacity=".4"/><circle cx="3" cy="10" r="1.500" opacity=".4"/><circle cx="8" cy="10" r="1.500"/><circle cx="13" cy="10" r="1.500"/><circle cx="18" cy="10" r="1.500" opacity=".4"/><circle cx="23" cy="10" r="1.500"/></g></svg>',
  exposure: '<svg width="28" height="16" viewBox="0 0 28 16"><g fill="currentColor"><rect x="1" y="2" width="26" height="4" rx="2" opacity=".3"/><rect x="1" y="2" width="19" height="4" rx="2"/><rect x="1" y="10" width="26" height="4" rx="2" opacity=".3"/><rect x="1" y="10" width="8" height="4" rx="2"/></g></svg>',
  cost: '<svg width="26" height="16" viewBox="0 0 26 16"><g fill="currentColor"><rect x="1" y="9" width="5" height="6" rx="1" opacity=".5"/><rect x="7.500" y="6" width="5" height="9" rx="1" opacity=".5"/><rect x="14" y="4" width="5" height="11" rx="1" opacity=".8"/><rect x="20.500" y="1" width="5" height="14" rx="1"/></g></svg>',
  liquidity: '<svg width="26" height="16" viewBox="0 0 26 16"><g fill="currentColor"><rect x="1" y="11" width="4" height="4" rx="1" opacity=".5"/><rect x="6.500" y="9" width="4" height="6" rx="1" opacity=".65"/><rect x="12" y="7" width="4" height="8" rx="1" opacity=".8"/><rect x="17.500" y="4" width="4" height="11" rx="1"/><rect x="23" y="1" width="3" height="14" rx="1"/></g></svg>',
};

/* ---------- top bar ---------- */
/* The label and its sentence come from what the payload counted, never from the setting. */
const STATUS = BT.embedStatus(P.embed);
const detailLabel = () => t(P.detail !== "full" ? "detailSummary" : STATUS.complete ? "detailFull" : "detailPartial");
function detailNote() {
  if (P.detail !== "full") return t("detailSummaryD");
  const parts = STATUS.parts.filter((p) => p.total > 0).map((p) => {
    const what = t("held_" + p.name);
    return p.embedded === p.total ? t("heldAll", p.total, what) : t("heldSome", p.embedded, p.total, what);
  });
  return parts.length ? t("heldLine", parts.join(t("listSep"))) : "";
}
function renderBar() {
  const coarse = matchMedia("(pointer:coarse)").matches;
  $("bar").innerHTML = `<div class="brand" title="${esc(P.run.title)}"><span aria-hidden="true">${ICON.mark}</span><span class="word">${esc(P.run.title)}</span><span class="sample" title="${esc(detailNote())}">${esc(detailLabel())}</span></div>
  <div class="seg main" role="group"><button id="tabA" aria-pressed="${S.view === "analysis"}">${t("tabAnalysis")}</button><button id="tabH" aria-pressed="${S.view === "holdings"}">${t("tabHoldings")}</button></div>
  <div class="tools"><button class="tool" id="btnSearch" aria-label="${t("search")}">${ICON.search}<span class="lbl">${t("search")}</span>${coarse ? "" : '<span class="kbd">Ctrl K</span>'}</button>
  <button class="tool" id="btnLang" aria-label="${t("switchLabel")}">${ICON.globe}<span>${t("switchTo")}</span></button>
  <button class="tool" id="btnSet" aria-label="${t("settings")}">${ICON.gear}</button></div>`;
  $("tabA").onclick = () => setView("analysis");
  $("tabH").onclick = () => setView("holdings");
  $("btnSearch").onclick = openSearch;
  $("btnSet").onclick = openSettings;
  $("btnLang").onclick = () => setLang(S.lang === "en" ? "zh-TW" : "en");
}
function setView(v) { S.view = v; renderAll(); scrollTo(0, 0); }
function setLang(l) { S.lang = l; store.set("report.lang", l); renderAll(); renderOverlay(); }
function applyTheme() {
  const dark = S.theme === "system" ? !matchMedia("(prefers-color-scheme:light)").matches : S.theme === "dark";
  document.documentElement.dataset.theme = dark ? "dark" : "light";
  document.documentElement.dataset.updown = S.updown;
  tokens();
}
function setTheme(v) { S.theme = v; store.set("report.theme", v); applyTheme(); renderAll(); renderOverlay(); }
function setUpdown(v) { S.updown = v; store.set("report.updown", v); applyTheme(); renderAll(); renderOverlay(); }

/* ---------- hero ---------- */
const H = { el: null, ctx: null, W: 0, H: 0, dpr: 1, a: 0, b: LAST, lo: 1, hi: 2, dmin: -0.3, paths: null, raf: 0, anim: 0 };
let TK = {};
function tokens() {
  TK = {
    rise: css("--rise"), fall: css("--fall"), ink: css("--ink"), muted: css("--muted"), faint: css("--faint"),
    well: css("--well"), line: css("--line2"), sans: css("--sans"), band: css("--raise2"),
  };
}
const TRACKS = new Map();
function trackOf(eq, a, b) {
  const key = `${eq === EQ ? "s" : "b"}${a}:${b}`;
  if (!TRACKS.has(key)) {
    if (TRACKS.size > 16) TRACKS.clear();
    TRACKS.set(key, BT.track(eq, DAYS, a, b));
  }
  return TRACKS.get(key);
}
const heroRange = () => (S.win ? [S.win.a, S.win.b] : [0, LAST]);
function domain(a, b) {
  const ia = Math.round(a), ib = Math.round(b), e0 = EQ[ia], b0 = BENCH ? BENCH[ia] : 1;
  let lo = Infinity, hi = -Infinity;
  for (let i = ia; i <= ib; i++) {
    lo = Math.min(lo, EQ[i] / e0); hi = Math.max(hi, EQ[i] / e0);
    if (BENCH) { lo = Math.min(lo, BENCH[i] / b0); hi = Math.max(hi, BENCH[i] / b0); }
  }
  const low = trackOf(EQ, ia, ib).mdd[ib - ia], lowB = BENCH ? trackOf(BENCH, ia, ib).mdd[ib - ia] : 0;
  return { lo, hi, dm: Math.min(low, lowB, -0.02) };
}
function geom() {
  const { W, H: hh } = H, ia = Math.round(H.a), ib = Math.round(H.b);
  const mainTop = hh > 260 ? 44 : 40, mainBot = hh * 0.66, ddTop = mainBot + 16, ddBot = hh - 14;
  const scale = (v) => (S.log ? Math.log(v) : v);
  const x = (i) => (i - H.a) / (H.b - H.a) * W;
  const y = (v) => mainBot - (scale(v) - scale(H.lo)) / (scale(H.hi) - scale(H.lo) || 1) * (mainBot - mainTop);
  const yd = (v) => ddTop + v / H.dmin * (ddBot - ddTop - 14);
  const e0 = EQ[ia], b0 = BENCH ? BENCH[ia] : 1;
  const ps = new Path2D(), pb = new Path2D(), pa = new Path2D(), pd = new Path2D(), pdl = new Path2D(), pbd = new Path2D();
  const T = trackOf(EQ, ia, ib), TB = BENCH ? trackOf(BENCH, ia, ib) : null, last = ib - ia;
  pa.moveTo(x(ia), mainBot);
  pd.moveTo(x(ia), ddTop);
  for (const i of BT.drawIndices(BENCH ? [EQ, BENCH] : [EQ], ia, ib, W)) {
    const X = x(i), Y = y(EQ[i] / e0), dv = EQ[i] / EQ[T.peak[i - ia]] - 1, step = i === ia ? "moveTo" : "lineTo";
    ps[step](X, Y);
    pdl[step](X, yd(dv));
    if (BENCH) { pb[step](X, y(BENCH[i] / b0)); pbd[step](X, yd(BENCH[i] / BENCH[TB.peak[i - ia]] - 1)); }
    pa.lineTo(X, Y);
    pd.lineTo(X, yd(dv));
  }
  pa.lineTo(x(ib), mainBot); pa.closePath();
  pd.lineTo(x(ib), ddTop); pd.closePath();
  H.paths = {
    ps, pb, pa, pd, pdl, pbd, x, y, yd, e0, b0, ia, ib, mainTop, mainBot, ddTop, ddBot, T,
    minI: T.low[last], minV: T.mdd[last], uwA: T.uwA[last], uwB: T.uwB[last],
  };
}
function strokeLayer(p, alpha) {
  const c = H.ctx;
  c.globalAlpha = alpha;
  const g = c.createLinearGradient(0, p.mainTop, 0, p.mainBot);
  g.addColorStop(0, TK.rise + "26");
  g.addColorStop(1, TK.rise + "00");
  c.fillStyle = g;
  c.fill(p.pa);
  c.lineJoin = "round"; c.lineCap = "round";
  if (BENCH) { c.strokeStyle = TK.muted; c.lineWidth = 1; c.setLineDash([3, 3]); c.stroke(p.pb); c.setLineDash([]); }
  c.strokeStyle = TK.rise; c.lineWidth = 2; c.stroke(p.ps);
  c.fillStyle = TK.fall + "3d"; c.fill(p.pd);
  if (BENCH) { c.strokeStyle = TK.muted; c.lineWidth = 1; c.setLineDash([3, 3]); c.stroke(p.pbd); c.setLineDash([]); }
  c.strokeStyle = TK.fall; c.lineWidth = 1.25; c.stroke(p.pdl);
  c.globalAlpha = 1;
}
function dot(x, y, r, col, ring) {
  const c = H.ctx;
  if (ring) { c.beginPath(); c.arc(x, y, r + 2, 0, 7); c.fillStyle = TK.well; c.fill(); }
  c.beginPath(); c.arc(x, y, r, 0, 7); c.fillStyle = col; c.fill();
}
function paintUnderwater(p, cur) {
  const c = H.ctx, hh = H.H, W = H.W;
  c.font = `11px ${TK.sans}`;
  c.fillStyle = TK.faint;
  const ux = p.x(p.uwA), vx = p.x(p.uwB), by = hh - 8, bracket = p.uwB > p.uwA;
  if (!bracket || ux > c.measureText(t("ddBand")).width + 16) c.fillText(t("ddBand"), 8, hh - 5);
  if (!bracket) return;
  c.globalAlpha = cur != null && cur < p.uwB ? 0.4 : 1;
  c.strokeStyle = TK.muted; c.lineWidth = 1;
  c.beginPath(); c.moveTo(ux + 0.5, by - 4); c.lineTo(ux + 0.5, by + 0.5); c.lineTo(vx - 0.5, by + 0.5); c.lineTo(vx - 0.5, by - 4); c.stroke();
  const label = t("ddMark", F().span(BT.dayDiff(DAYS[p.uwA], DAYS[p.uwB]))), width = c.measureText(label).width;
  const lx = clamp((ux + vx) / 2 - width / 2, 6, W - width - 6);
  if (vx - ux > width + 12) {
    c.fillStyle = TK.well; c.fillRect(lx - 4, by - 7, width + 8, 12);
    c.fillStyle = TK.muted; c.fillText(label, lx, by + 3);
  }
  c.globalAlpha = 1;
}
function paint() {
  H.raf = 0;
  if (!H.ctx || !H.W) return;
  const c = H.ctx, p = H.paths, { W, H: hh } = H, cur = S.cursor;
  c.setTransform(H.dpr, 0, 0, H.dpr, 0, 0);
  c.clearRect(0, 0, W, hh);
  if (S.hoverYear != null) {
    const y = YEAR_ROWS.find((o) => o.y === S.hoverYear);
    c.fillStyle = TK.band;
    c.fillRect(p.x(y.a), 0, p.x(y.b) - p.x(y.a), hh);
  }
  c.strokeStyle = TK.fall + "66"; c.lineWidth = 1;
  c.beginPath(); c.moveTo(0, p.ddTop + 0.5); c.lineTo(W, p.ddTop + 0.5); c.stroke();
  if (cur == null) strokeLayer(p, 1);
  else {
    const cx = p.x(cur);
    c.save(); c.beginPath(); c.rect(0, 0, cx, hh); c.clip(); strokeLayer(p, 1); c.restore();
    c.save(); c.beginPath(); c.rect(cx, 0, W - cx, hh); c.clip(); strokeLayer(p, 0.3); c.restore();
  }
  const ex = Math.min(p.x(p.ib), W - 7), ey = p.y(EQ[p.ib] / p.e0);
  c.globalAlpha = cur != null && cur < p.ib ? 0.4 : 1;
  c.beginPath(); c.arc(ex, ey, 10, 0, 7); c.fillStyle = TK.rise + "2e"; c.fill();
  dot(ex, ey, 4, TK.rise);
  c.globalAlpha = 1;
  if (p.minV < 0) {
    const mx = p.x(p.minI), my = p.yd(p.minV), label = F().pct(p.minV, 1);
    c.globalAlpha = cur != null && cur < p.minI ? 0.45 : 1;
    dot(mx, my, 3, TK.fall);
    c.font = `12px ${TK.sans}`;
    const tw = c.measureText(label).width, lx = mx + tw + 14 > W ? mx - tw - 7 : mx + 7, ly = Math.min(my + 4, hh - 3);
    c.fillStyle = TK.well; c.fillRect(lx - 3, ly - 11, tw + 6, 15);
    c.fillStyle = TK.muted; c.fillText(label, lx, ly);
    c.globalAlpha = 1;
  }
  paintUnderwater(p, cur);
  if (cur == null) return;
  const cx = Math.round(p.x(cur)) + 0.5, lowV = p.T.mdd[cur - p.ia];
  if (lowV < 0) {
    /* the to-date floor: the level the drawdown figure is reading */
    const lowI = p.T.low[cur - p.ia], ly = p.yd(lowV);
    c.strokeStyle = TK.fall + "99"; c.lineWidth = 1; c.setLineDash([2, 3]);
    c.beginPath(); c.moveTo(p.x(lowI), ly); c.lineTo(cx, ly); c.stroke(); c.setLineDash([]);
    dot(p.x(lowI), ly, 3, TK.fall);
  }
  c.strokeStyle = TK.muted; c.lineWidth = 1;
  c.beginPath(); c.moveTo(cx, 0); c.lineTo(cx, hh - 2); c.stroke();
  if (BENCH) dot(cx, p.y(BENCH[cur] / p.b0), 3.5, TK.ink, true);
  dot(cx, p.y(EQ[cur] / p.e0), 5, TK.rise, true);
  dot(cx, p.yd(EQ[cur] / EQ[p.T.peak[cur - p.ia]] - 1), 3.5, TK.fall, true);
}
const repaint = () => { if (!H.raf) H.raf = requestAnimationFrame(paint); };
function sizeCanvas() {
  const el = H.el;
  if (!el) return;
  const r = el.getBoundingClientRect();
  if (!r.width) return;
  H.dpr = Math.min(2.5, devicePixelRatio || 1);
  H.W = r.width; H.H = r.height;
  el.width = Math.round(r.width * H.dpr);
  el.height = Math.round(r.height * H.dpr);
  geom();
  paint();
}
function setRange(a, b) {
  cancelAnimationFrame(H.anim);
  const d = domain(a, b);
  if (reduced() || !H.W) { H.a = a; H.b = b; H.lo = d.lo; H.hi = d.hi; H.dmin = d.dm; geom(); paint(); return; }
  const f = { a: H.a, b: H.b, lo: H.lo, hi: H.hi, dm: H.dmin }, t0 = performance.now(), D = 280;
  const step = (now) => {
    const k = clamp((now - t0) / D, 0, 1), e = 1 - (1 - k) ** 3;
    H.a = f.a + (a - f.a) * e; H.b = f.b + (b - f.b) * e;
    H.lo = f.lo + (d.lo - f.lo) * e; H.hi = f.hi + (d.hi - f.hi) * e; H.dmin = f.dm + (d.dm - f.dm) * e;
    geom(); paint();
    if (k < 1) H.anim = requestAnimationFrame(step);
  };
  H.anim = requestAnimationFrame(step);
}
/* The one shared date range. Hero, trades and charts all read it, and every date action
   sets it. `first` is the first day inside; `a` is the close before it, the base for returns. */
const makeWin = (first, b, meta) => ({ a: Math.max(0, first - 1), first, b, ...meta });
function winLabel(win) {
  const f = F();
  if (win.kind === "year") return String(win.year);
  if (win.kind === "month") { const m = MONTH_ROWS.find((o) => o.k === win.month); return f.month(m.y, m.m); }
  return t("win", f.date(DAYS[win.first]), f.date(DAYS[win.b]));
}
const inWindow = (i) => !S.win || (i != null && i >= S.win.first && i <= S.win.b);
function windowValue(win) {
  const f = F(), { a, b } = win, label = winLabel(win);
  const k = BT.rangeKpi(EQ, DAYS, a, b, RF), T = trackOf(EQ, a, b), n = b - a;
  const bench = BENCH ? BENCH[b] / BENCH[a] - 1 : null;
  const labels = { year: ["yearRet", "yearDd"], month: ["monthRet", "monthDd"] }[win.kind];
  return {
    win: label, labL: labels ? t(labels[0], label) : t("winRet"), labR: labels ? t(labels[1], label) : t("maxdd"),
    v: k.total_return, m: k.max_drawdown, b: bench, mb: BENCH ? trackOf(BENCH, a, b).mdd[n] : null,
    noteL: bench != null ? t("noteYear", f.spct(k.total_return - bench, 1)) : t("noteSharpe", f.ratio(k.sharpe)),
    noteR: k.max_drawdown < 0 ? t("ddYear", f.dateShort(DAYS[T.low[n]])) : t("uwHigh"),
  };
}
/* The hero reads one window, shown above both figures, at three levels of honesty.
   At rest, the whole history: annualised return, max drawdown.
   On a year or preset, that window alone: its return, and its drawdown from highs inside it.
   At the cursor, window start to that date: total return, and the worst fall so far.
   A cursor return is not annualised under a year, and that day's own drawdown is a separate
   note, because the max can only hold or deepen as the cursor moves right. */
function heroValue() {
  const f = F(), [a, b] = heroRange();
  if (S.cursor != null) {
    const i = S.cursor, T = trackOf(EQ, a, b), k = i - a, v = EQ[i] / EQ[a] - 1, pk = T.peak[k];
    const yrs = BT.dayDiff(DAYS[a], DAYS[i]) / 365.25, now = EQ[i] / EQ[pk] - 1;
    return {
      win: t("win", f.date(DAYS[a]), f.date(DAYS[i])), labL: t("totalTo"), labR: t("ddTo"), v, m: T.mdd[k],
      b: BENCH ? BENCH[i] / BENCH[a] - 1 : null, mb: BENCH ? trackOf(BENCH, a, b).mdd[k] : null,
      noteL: yrs >= 1 ? t("noteAnn", f.spct(BT.annualise(1 + v, yrs), 1)) : t("noteShort"),
      noteR: now < 0 ? t("uwNow", f.pct(-now, 1), f.span(BT.dayDiff(DAYS[pk], DAYS[i]))) : t("uwHigh"),
    };
  }
  if (S.hoverYear != null) {
    const y = YEAR_ROWS.find((o) => o.y === S.hoverYear);
    return windowValue(makeWin(y.a, y.b, { kind: "year", year: y.y }));
  }
  if (S.win) return windowValue(S.win);
  const dd = SUM.drawdown, ym = (iso) => { const [y, m] = dateParts(iso); return f.month(y, m); };
  return {
    win: t("win", f.date(DAYS[0]), f.date(DAYS[LAST])), labL: t("annual"), labR: t("maxdd"),
    v: PERF.cagr, m: PERF.max_drawdown, b: SUM.benchmark?.cagr ?? null, mb: SUM.benchmark?.max_drawdown ?? null,
    noteL: t("noteRest", f.spct(PERF.total_return, 0), YEAR_ROWS.filter((y) => y.r > 0).length, YEARS.length),
    noteR: dd.underwater_start ? t("uwRest", f.span(dd.longest_underwater_days)) : t("uwHigh"),
    noteRx: dd.underwater_start ? t("uwDates", ym(dd.underwater_start), ym(dd.underwater_end)) : "",
  };
}
const bigHtml = (txt) => `${esc(txt.slice(0, -1))}<i>%</i>`;
function updateHero() {
  if (S.view !== "analysis" || !$("win")) return;
  const f = F(), v = heroValue(), neg = v.v < 0, txt = f.spct(v.v, 1), mt = f.pct(v.m, 1);
  $("win").textContent = v.win;
  $("labL").textContent = v.labL;
  $("labR").textContent = v.labR;
  $("bigNum").className = "big " + toneCls(v.v);
  $("bigNum").innerHTML = bigHtml(txt);
  $("bigDd").className = "big " + (v.m < 0 ? "dn" : "flat");
  $("bigDd").innerHTML = bigHtml(mt);
  if (BENCH) {
    $("vsB").textContent = `${t("bench")} ${f.spct(v.b, 1)}`;
    $("vsBd").textContent = `${t("bench")} ${f.pct(v.mb, 1)}`;
  }
  $("noteL").textContent = v.noteL;
  $("noteR").innerHTML = esc(v.noteR) + (v.noteRx ? `<span class="dx">${esc(v.noteRx)}</span>` : "");
  $("gR").classList.toggle("lit", !neg);
  $("gF").classList.toggle("lit", neg);
  const active = S.cursor != null;
  $("mult").style.opacity = active ? 0.3 : 1;
  $("hint").style.opacity = S.scrubbed || S.win ? 0 : 1;
  $("dayBtn").hidden = !(S.touch && active);
  const cy = active ? +DAYS[S.cursor].slice(0, 4) : S.win?.kind === "year" ? S.win.year : null;
  document.querySelectorAll(".yr").forEach((b) => b.classList.toggle("later", cy != null && +b.dataset.y > cy));
  H.el.setAttribute("aria-valuenow", S.cursor == null ? LAST : S.cursor);
  H.el.setAttribute("aria-valuetext", t("chartValue", v.win, txt, mt));
  repaint();
}
function presetWindow(key) {
  const [y, m, d] = dateParts(DAYS[LAST]);
  const iso = (Y, M) => `${Y}-${pad2(M)}-${pad2(d)}`;
  const from = { "3y": iso(y - 3, m), "1y": iso(y - 1, m), ytd: `${y}-01-01`, "6m": m > 6 ? iso(y, m - 6) : iso(y - 1, m + 6) }[key];
  return makeWin(dayOf(from), LAST, { kind: "preset", preset: key });
}
function yearStrip() {
  const f = F(), maxY = Math.max(...YEAR_ROWS.map((y) => Math.abs(y.r)), 1e-9);
  const excess = YEAR_ROWS.filter((y) => y.bench != null).map((y) => Math.abs(y.r - y.bench));
  const maxX = Math.max(...excess, 1e-9);
  return YEAR_ROWS.map((y) => {
    const days = y.b - y.a + 1, w = days / N, x = y.bench == null ? null : y.r - y.bench;
    const style = [`flex:${w} ${w} 0`, `--tone:var(${y.r < 0 ? "--fall" : "--rise"})`, `--a:${(0.14 + 0.74 * Math.pow(Math.abs(y.r) / maxY, 1.3)).toFixed(3)}`];
    if (x != null) style.push(`--xt:var(${x < 0 ? "--fall" : "--rise"})`, `--xa:${(0.3 + 0.7 * Math.abs(x) / maxX).toFixed(3)}`);
    const label = days < 70 ? "" : `<span class="l4">${y.y}</span><span class="l2">${String(y.y).slice(2)}</span>`;
    return `<button class="yr" data-y="${y.y}" aria-pressed="${S.win?.year === y.y}" aria-label="${esc(t("yearAria", y.y, f.spct(y.r, 1)))}" style="${style.join(";")}"><span class="sw"></span>${x == null ? "" : '<span class="ex"></span>'}<span class="lb">${label}</span></button>`;
  }).join("");
}
function renderHero() {
  const [a, b] = heroRange(), mult = EQ[b] / EQ[a], coarse = matchMedia("(pointer:coarse)").matches;
  const fig = (lab, big, cls, vs, note) => `<div class="fig"><div class="lab" id="${lab}"></div><div class="big ${cls}" id="${big}"></div>${BENCH ? `<div class="vs num"><b>---</b><span id="${vs}"></span></div>` : ""}<div class="note num" id="${note}"></div></div>`;
  const presets = [["all", t("all")], ["3y", "3Y"], ["1y", "1Y"], ["ytd", t("ytd")], ["6m", "6M"]];
  const cur = S.win ? S.win.preset || "" : "all", fixed = S.win && S.win.kind !== "preset";
  const named = S.win?.kind === "year" || S.win?.kind === "month";
  const clearChip = fixed ? `<button class="chip num" id="chipWin" aria-label="${esc(t("clearWin", winLabel(S.win)))}">${esc(named ? winLabel(S.win) : t("clearShort"))} ${ICON.x}</button>` : "";
  const dateBox = (id, label, value) => `<input class="dateinp num" type="date" id="${id}" aria-label="${t(label)}" min="${DAYS[0]}" max="${DAYS[LAST]}" value="${value}">`;
  const chip = `${clearChip}<div class="seg sm" id="presets">${presets.map((p) => `<button data-p="${p[0]}" aria-pressed="${cur === p[0]}">${esc(p[1])}</button>`).join("")}</div>
   ${dateBox("winFrom", "dateFrom", DAYS[S.win ? S.win.first : 0])}${dateBox("winTo", "dateTo", DAYS[S.win ? S.win.b : LAST])}`;
  $("heroSlot").innerHTML = `<section class="hero" aria-label="${t("annual")}, ${t("maxdd")}">
   <span class="glow rise lit" id="gR"></span><span class="glow fall" id="gF"></span><span class="glow side"></span>
   <div class="hhead"><div class="eyebrow num" id="win"></div><div class="chips">${chip}</div></div>
   <div class="figs">${fig("labL", "bigNum", "up", "vsB", "noteL")}${fig("labR", "bigDd", "dn", "vsBd", "noteR")}</div>
   <div class="plot"><canvas id="cv" tabindex="0" role="slider" aria-label="${esc(t("chartLabel"))}" aria-valuemin="0" aria-valuemax="${LAST}"></canvas>
    <span class="pilltag num" id="mult">${esc(F().times(mult))}</span>
    <span class="hint" id="hint">${ICON.arrows}<span>${coarse ? t("scrubTouch") : t("scrub")}</span></span>
    <button class="daybtn" id="dayBtn" hidden>${t("seeDay")}</button>
    <button class="chip logchip" id="logBtn" aria-pressed="${S.log}">${t(S.log ? "linear" : "log")}</button></div>
   <div class="strip" role="group" aria-label="${t("fullYear")}">${yearStrip()}</div>
   <div class="stripnote"><span>${t("stripNote")}</span>${BENCH ? `<span>${t("stripNoteEx")}</span>` : ""}</div></section>`;
  H.el = $("cv");
  H.ctx = H.el.getContext("2d");
  bindHero();
  H.a = a; H.b = b;
  const d = domain(a, b);
  H.lo = d.lo; H.hi = d.hi; H.dmin = d.dm;
  sizeCanvas();
  updateHero();
}
function bindHero() {
  const idx = (e) => {
    const r = H.el.getBoundingClientRect();
    return clamp(Math.round(H.a + (e.clientX - r.left) / r.width * (H.b - H.a)), Math.ceil(H.a), Math.floor(H.b));
  };
  const scrub = (e) => { S.cursor = idx(e); S.scrubbed = true; S.touch = e.pointerType === "touch"; updateHero(); };
  H.el.addEventListener("pointermove", (e) => { if (e.pointerType === "touch" && !e.buttons && e.pressure === 0) return; scrub(e); });
  H.el.addEventListener("pointerdown", scrub);
  H.el.addEventListener("pointerleave", (e) => { if (e.pointerType === "touch") return; S.cursor = S.pin; updateHero(); });
  H.el.addEventListener("click", (e) => { if (e.pointerType === "touch" || S.touch) return; openDay(idx(e)); });
  H.el.addEventListener("keydown", (e) => {
    const [a, b] = heroRange();
    if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
      e.preventDefault();
      const base = S.cursor == null ? b : S.cursor;
      S.cursor = clamp(base + (e.key === "ArrowLeft" ? -1 : 1), a, b);
      S.scrubbed = true;
      updateHero();
    } else if (e.key === "Enter" && S.cursor != null) openDay(S.cursor);
    else if (e.key === "Escape" && S.pin == null) { S.cursor = null; updateHero(); }
  });
  H.el.addEventListener("blur", () => { if (S.pin == null && !S.touch) { S.cursor = null; updateHero(); } });
  $("dayBtn").onclick = () => openDay(S.cursor);
  $("logBtn").onclick = () => { S.log = !S.log; renderHero(); };
  document.querySelectorAll(".yr").forEach((btn) => {
    const y = +btn.dataset.y;
    btn.onpointerenter = (e) => { if (e.pointerType === "mouse") { S.hoverYear = y; updateHero(); } };
    btn.onpointerleave = () => { S.hoverYear = null; updateHero(); };
    btn.onfocus = () => { S.hoverYear = y; updateHero(); };
    btn.onblur = () => { S.hoverYear = null; updateHero(); };
    btn.onclick = () => toggleYear(y);
  });
  if ($("chipWin")) $("chipWin").onclick = () => setWindow(null);
  $("winFrom").onchange = setDates;
  $("winTo").onchange = setDates;
  document.querySelectorAll("#presets button").forEach((b) => {
    b.onclick = () => setWindow(b.dataset.p === "all" ? null : presetWindow(b.dataset.p));
  });
}
function setWindow(win) {
  if (win && win.b <= win.a) return;
  const before = { a: H.a, b: H.b, lo: H.lo, hi: H.hi, d: H.dmin };
  S.win = win; S.hoverYear = null; S.cursor = S.pin = null; S.page = 0;
  if (S.panel?.type === "day") { S.panel = null; document.body.classList.remove("panel-open"); renderPanel(); }
  renderHero();
  const [a, b] = heroRange();
  H.a = before.a; H.b = before.b; H.lo = before.lo; H.hi = before.hi; H.dmin = before.d;
  geom();
  setRange(a, b);
  renderBehind(); renderHealth(); renderMonthly(); renderTrades();
}
/* Any two dates: the first trading day on or after one, the last on or before the other.
   A pair that holds no range puts the boxes back rather than guessing. */
function setDates() {
  const from = $("winFrom").value, to = $("winTo").value;
  const first = from ? dayOf(from) : 0;
  let last = to ? dayOf(to) : LAST;
  if (to && DAYS[last] > to) last--;
  const win = makeWin(first, last, { kind: "custom" });
  if (last < first || win.b <= win.a) {
    $("winFrom").value = DAYS[S.win ? S.win.first : 0];
    $("winTo").value = DAYS[S.win ? S.win.b : LAST];
    return;
  }
  setWindow(first === 0 && last === LAST ? null : win);
}
function toggleYear(year) {
  if (S.win?.year === year) { setWindow(null); return; }
  const y = YEAR_ROWS.find((o) => o.y === year);
  setWindow(makeWin(y.a, y.b, { kind: "year", year }));
}
function toggleMonth(k) {
  if (S.win?.month === k) { setWindow(null); return; }
  const m = MONTH_ROWS.find((o) => o.k === k);
  setWindow(makeWin(m.a, m.b, { kind: "month", month: k }));
}

/* ---------- small chart kit ---------- */
function showTip(x, y, html) {
  const e = $("tip");
  e.innerHTML = html;
  e.hidden = false;
  const w = e.offsetWidth, h = e.offsetHeight;
  e.style.left = clamp(x + 14, 8, innerWidth - w - 8) + "px";
  e.style.top = clamp(y - h - 10, 8, innerHeight - h - 8) + "px";
}
const hideTip = () => { $("tip").hidden = true; };
function nice(lo, hi, n = 3) {
  const raw = (hi - lo) / n || 1, p = 10 ** Math.floor(Math.log10(raw));
  const s = [1, 2, 2.5, 5, 10].map((k) => k * p).find((k) => k >= raw) || 10 * p, out = [];
  for (let v = Math.ceil(lo / s) * s; v <= hi + 1e-9; v += s) out.push(+v.toFixed(8));
  return out;
}
const legend = (items) => `<div class="legend">${items.map((i) => `<span><i class="${i.dash ? "dash" : ""}" style="background:${i.c};border-color:${i.c}"></i>${esc(i.n)}</span>`).join("")}</div>`;
function lineChart(el, o) {
  const W = el.clientWidth || 600, Ht = o.h || 200, pb = 20, pr = 44, pt = 8, a = o.a ?? 0, b = o.b ?? LAST;
  const vals = o.series.flatMap((s) => s.v.slice(a, b + 1).filter((v) => v != null));
  if (!vals.length) { el.innerHTML = ""; return; }
  let lo = Math.min(...vals), hi = Math.max(...vals);
  if (o.zero) { lo = Math.min(lo, 0); hi = Math.max(hi, 0); }
  if (lo === hi) hi = lo + 1;
  const ticks = nice(lo, hi, 3);
  lo = Math.min(lo, ticks[0]); hi = Math.max(hi, ticks[ticks.length - 1]);
  const x = (i) => (i - a) / (b - a) * (W - pr), y = (v) => pt + (hi - v) / (hi - lo) * (Ht - pt - pb);
  const points = BT.drawIndices(o.series.map((s) => s.v), a, b, W);
  const path = (s) => {
    let d = "", on = false;
    for (const i of points) {
      const v = s.v[i];
      if (v == null) { on = false; continue; }
      d += (on ? "L" : "M") + x(i).toFixed(1) + " " + y(v).toFixed(1);
      on = true;
    }
    return d;
  };
  let g = ticks.map((v) => `<line x1="0" x2="${W - pr}" y1="${y(v)}" y2="${y(v)}" stroke="var(--line)"/><text x="${W - pr + 6}" y="${y(v) + 4}">${esc(o.fmt(v))}</text>`).join("");
  const yrs = YEAR_ROWS.filter((k) => k.a >= a && k.a <= b), every = Math.ceil(yrs.length / (W < 480 ? 4 : 8)) || 1;
  g += yrs.filter((k, i) => i % every === 0).map((k) => `<text x="${x(k.a)}" y="${Ht - 5}">${k.y}</text>`).join("");
  if (o.band) g += `<rect x="${x(o.band[0])}" y="${pt}" width="${Math.max(1, x(o.band[1]) - x(o.band[0]))}" height="${Ht - pt - pb}" fill="var(--accent-soft)" stroke="var(--accent)" stroke-width=".6"/>`;
  g += o.series.map((s) => `<path d="${path(s)}" fill="none" stroke="${s.c}" stroke-width="${s.w || 1.6}" ${s.dash ? `stroke-dasharray="${s.dash}"` : ""} stroke-linejoin="round"/>`).join("");
  el.innerHTML = `<svg width="${W}" height="${Ht}" viewBox="0 0 ${W} ${Ht}" role="img" aria-label="${esc(o.label || "")}">${g}<line class="xh" y1="${pt}" y2="${Ht - pb}" stroke="var(--muted)" visibility="hidden"/><rect class="hit" x="0" y="0" width="${W - pr}" height="${Ht}" fill="transparent"/></svg>`;
  const xh = el.querySelector(".xh"), hit = el.querySelector(".hit");
  const at = (e) => { const r = hit.getBoundingClientRect(); return clamp(Math.round(a + (e.clientX - r.left) / (W - pr) * (b - a)), a, b); };
  hit.addEventListener("pointermove", (e) => {
    const i = at(e);
    xh.setAttribute("x1", x(i)); xh.setAttribute("x2", x(i)); xh.setAttribute("visibility", "visible");
    showTip(e.clientX, e.clientY, `<div class="tl">${esc(F().date(DAYS[i]))}</div>` + o.series.map((s) => `<div><span>${esc(s.n)}</span><span>${s.v[i] == null ? "—" : esc(o.fmt(s.v[i]))}</span></div>`).join(""));
  });
  hit.addEventListener("pointerleave", () => { xh.setAttribute("visibility", "hidden"); hideTip(); });
  if (o.onClick) hit.addEventListener("click", (e) => o.onClick(at(e)));
}
function barChart(el, o) {
  const W = el.clientWidth || 600, Ht = o.h || 190, pb = 20, pr = 44, pt = 8, n = o.cats.length, sl = (W - pr) / n, grouped = o.mode === "group";
  let lo = 0, hi = 0;
  for (let i = 0; i < n; i++) {
    if (grouped) o.series.forEach((s) => { lo = Math.min(lo, s.v[i] ?? 0); hi = Math.max(hi, s.v[i] ?? 0); });
    else hi = Math.max(hi, o.series.reduce((a, s) => a + s.v[i], 0));
  }
  if (lo === hi) hi = lo + 1;
  const ticks = nice(lo, hi, 3);
  lo = Math.min(lo, ticks[0]); hi = Math.max(hi, ticks[ticks.length - 1]);
  const y = (v) => pt + (hi - v) / (hi - lo) * (Ht - pt - pb), lab = Math.ceil(n / (W < 480 ? 6 : 14));
  let g = ticks.map((v) => `<line x1="0" x2="${W - pr}" y1="${y(v)}" y2="${y(v)}" stroke="var(--line)"/><text x="${W - pr + 6}" y="${y(v) + 4}">${esc(o.fmt(v))}</text>`).join("");
  for (let i = 0; i < n; i++) {
    const cx = i * sl + sl / 2;
    if (o.sel === i) g += `<rect x="${i * sl + 1}" y="0" width="${sl - 2}" height="${Ht}" rx="6" fill="var(--raise)"/>`;
    if (grouped) {
      const bw = Math.min(14, (sl - 6) / o.series.length);
      o.series.forEach((s, k) => {
        const v = s.v[i] ?? 0, y0 = y(Math.max(v, 0)), h = Math.max(1, Math.abs(y(v) - y(0))), col = typeof s.c === "function" ? s.c(v) : s.c;
        g += `<rect x="${cx - bw * o.series.length / 2 + k * bw + 0.5}" y="${y0}" width="${bw - 1}" height="${h}" rx="2" fill="${col}"/>`;
      });
    } else {
      let acc = 0;
      const bw = Math.min(30, sl - 2);
      o.series.forEach((s) => {
        const v = s.v[i];
        if (v <= 0) return;
        const top = y(acc + v), h = Math.max(0.5, y(acc) - top - (o.series.length > 1 ? 2 : 0));
        g += `<rect x="${cx - bw / 2}" y="${top}" width="${bw}" height="${h}" rx="2" fill="${s.c}"/>`;
        acc += v;
      });
    }
    if (i % lab === 0) g += `<text x="${cx}" y="${Ht - 5}" text-anchor="middle">${esc(o.cats[i])}</text>`;
  }
  g += `<line x1="0" x2="${W - pr}" y1="${y(0)}" y2="${y(0)}" stroke="var(--line2)"/>`;
  g += o.cats.map((c, i) => `<rect data-i="${i}" x="${i * sl}" y="0" width="${sl}" height="${Ht}" fill="transparent"/>`).join("");
  el.innerHTML = `<svg width="${W}" height="${Ht}" viewBox="0 0 ${W} ${Ht}" role="img" aria-label="${esc(o.label || "")}">${g}</svg>`;
  el.querySelectorAll("[data-i]").forEach((r) => {
    const i = +r.dataset.i;
    r.addEventListener("pointermove", (e) => {
      if (o.onHover) o.onHover(i);
      showTip(e.clientX, e.clientY, `<div class="tl">${esc(o.tipTitle ? o.tipTitle(i) : o.cats[i])}</div>` + o.series.map((s) => `<div><span>${esc(s.n)}</span><span>${esc(o.fmt(s.v[i] ?? 0))}</span></div>`).join(""));
    });
    r.addEventListener("pointerleave", hideTip);
  });
}
/* Canvas, because a large run can have tens of thousands of points. */
function scatterChart(el, points, onPick) {
  const W = el.clientWidth || 600, Ht = 240, pb = 22, pl = 8, pr = 44, pt = 8, dpr = Math.min(2.5, devicePixelRatio || 1), f = F();
  el.innerHTML = `<canvas class="scat" role="img" aria-label="${esc(t("scatter"))}" style="height:${Ht}px"></canvas>`;
  const cv = el.firstChild, c = cv.getContext("2d");
  cv.width = W * dpr; cv.height = Ht * dpr;
  c.scale(dpr, dpr);
  const xs = points.map((p) => p.mae), ys = points.map((p) => p.ret);
  const xlo = Math.min(...xs, 0), xhi = Math.max(...xs, 0), ylo = Math.min(...ys, 0), yhi = Math.max(...ys, 0);
  const x = (v) => pl + (v - xlo) / (xhi - xlo || 1) * (W - pl - pr), y = (v) => pt + (yhi - v) / (yhi - ylo || 1) * (Ht - pt - pb);
  c.strokeStyle = TK.line; c.lineWidth = 1;
  c.beginPath(); c.moveTo(x(xlo), y(0)); c.lineTo(x(xhi), y(0)); c.moveTo(x(0), pt); c.lineTo(x(0), Ht - pb); c.stroke();
  c.fillStyle = TK.muted;
  c.font = `11px ${TK.sans}`;
  nice(ylo, yhi, 3).forEach((v) => c.fillText(f.spct(v, 0), W - pr + 6, y(v) + 4));
  nice(xlo, xhi, 3).forEach((v) => c.fillText(f.spct(v, 0), x(v) - 10, Ht - 6));
  c.globalAlpha = points.length > 2000 ? 0.45 : 0.75;
  points.forEach((p) => {
    c.fillStyle = p.ret < 0 ? TK.fall : TK.rise;
    c.beginPath(); c.arc(x(p.mae), y(p.ret), 2.2, 0, 7); c.fill();
  });
  c.globalAlpha = 1;
  const nearest = (e) => {
    const r = cv.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
    let best = null, dist = 144;
    points.forEach((p) => {
      const d = (x(p.mae) - mx) ** 2 + (y(p.ret) - my) ** 2;
      if (d < dist) { dist = d; best = p; }
    });
    return best;
  };
  cv.addEventListener("pointermove", (e) => {
    const p = nearest(e);
    if (!p) { hideTip(); return; }
    showTip(e.clientX, e.clientY, `<div class="tl">${esc(shown(p.symbol))}</div><div><span>${t("returnLabel")}</span><span>${esc(f.spct(p.ret, 1))}</span></div><div><span>${t("worstOpen")}</span><span>${esc(f.spct(p.mae, 1))}</span></div>`);
  });
  cv.addEventListener("pointerleave", hideTip);
  cv.addEventListener("click", (e) => { const p = nearest(e); if (p) onPick(p); });
}
function subTabs(k, items, def) {
  const cur = S.sub[k] || def;
  return `<div class="seg sm" data-sub="${k}">${items.map((i) => `<button data-v="${i[0]}" aria-pressed="${cur === i[0]}">${esc(i[1])}</button>`).join("")}</div>`;
}
function bindSub(el, redraw) {
  el.querySelectorAll("[data-sub]").forEach((w) => w.querySelectorAll("button").forEach((b) => {
    b.onclick = () => { S.sub[w.dataset.sub] = b.dataset.v; redraw(); };
  }));
}
function table(heads, rows) {
  return `<div class="tablewrap flush"><table><thead><tr>${heads.map((h) => `<th scope="col">${esc(h)}</th>`).join("")}</tr></thead><tbody>${rows.map((r) => `<tr>${r.map((c) => `<td>${esc(c)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
}

/* ---------- behind the headline ---------- */
const needsHtml = (key, ms, rerun) => `<div class="needs"><b>${t(key)}</b><span>${esc(t("missing", t(ms)))}</span>${rerun ? `<span>${t("rerun")} <code>${key === "needsLedger" ? "backtest_with_report(..., ledger=True)" : "to_bundle(..., input_df=df)"}</code></span>` : ""}</div>`;
function renderBehind() {
  const f = F(), E = SUM.entry_date, N2 = SUM.net_of_costs, LF = SUM.limit_fills;
  const word = (level, bad, good) => (level === "na" ? "" : `<span class="w ${level === "green" ? "good" : "watch"}">${level === "green" ? ICON.check : ICON.warn}${t(level === "green" ? good : bad)}</span>`);
  const tile = (k, q, inner) => `<button class="ins" data-ins="${k}"><span class="q">${esc(q)}</span>${inner}</button>`;
  const items = [
    tile("entry", t("qEntry"), E ? `<span class="f ${toneCls(E.worst)}">${esc(f.spct(E.worst, 1))}</span><span class="s">${esc(t("sEntry", f.spct(E.median, 1), f.pct(E.positive_share, 0)))}</span>${word(E.level, "wEntryBad", "wEntryOk")}` : `<span class="s">${t("shortHistory")}</span>`),
    tile("conc", t("qConc", CONC.k), `<span class="f ${toneCls(CONC.cagr_ex_best)}">${esc(f.spct(CONC.cagr_ex_best, 1))}<small>${t("aYear")}</small></span><span class="s">${esc(t("sConc", f.spct(CONC.cagr, 1)))}</span>${word(CONC.level, "wConcBad", "wConcOk")}`),
    tile("cost", t("qNet"), N2 && N2.cost_share != null ? `<span class="f">${esc(f.pct(N2.cost_share, 0))}<small>${t("ofGross")}</small></span><span class="s">${esc(t("sNet", f.spct(N2.gross_cagr, 1), f.spct(N2.net_cagr, 1)))}</span>${word(N2.level, "wCostBad", "wCostOk")}` : needsHtml("needsLedger", "msCost")),
    tile("limit", t("qLimit"), LF ? `<span class="f">${esc(f.pct(LF.share_return, 1))}</span><span class="s">${esc(t("sLimit", f.pct(LF.share_trades, 1), t("approx")))}</span>${word(LF.level, "wLimitBad", "wLimitOk")}` : needsHtml("needsInput", "msLimit")),
  ];
  $("behindSlot").innerHTML = `<section class="sec"><div class="shead"><h2>${t("behind")}</h2><span class="side">${t("behindSide")}</span></div>${S.win ? `<p class="scope">${t("wholeNote")}</p>` : ""}<div class="card tiles4">${items.join("")}</div></section>`;
  document.querySelectorAll(".ins").forEach((b) => { b.onclick = () => openPanel({ type: "insight", k: b.dataset.ins }); });
}
const pair = (a, b) => `<div><span>${esc(a)}</span><span>${esc(b)}</span></div>`;
const source = (...cols) => `<div class="dsrc">${t("dataFrom")}: ${cols.map((x) => `<code>${x}</code>`).join(" · ")}</div>`;
function insightEntry() {
  const f = F(), E = SUM.entry_date;
  if (!E) return { title: t("qEntry"), body: `<p class="prose">${t("shortHistory")}</p>` };
  const bench = E.bench_median != null ? pair(t("benchMed"), f.spct(E.bench_median, 1)) + pair(t("benchWorst"), f.spct(E.bench_worst, 1)) : "";
  const lg = [{ n: t("strategy"), c: "var(--rise)" }];
  if (ENTRY_B) lg.push({ n: t("bench"), c: "var(--muted)", dash: 1 });
  return {
    title: t("qEntry"),
    body: `<div><div class="pfig ${toneCls(E.worst)}">${esc(f.spct(E.worst, 1))}</div><div class="mut num small">${t("worstStart")}, ${esc(f.date(E.worst_start))}</div></div>
    <p class="prose">${t("iEntryP")}</p>
    <div class="pgrid">${pair(t("medStart"), f.spct(E.median, 1))}${pair(t("bestStart"), f.spct(E.best, 1))}${pair(t("posStart"), f.pct(E.positive_share, 0))}${bench}</div>
    <div class="psec"><h3>${t("iEntryCh")}</h3></div><div class="chart" id="insCh"></div>${legend(lg)}
    <p class="prose">${t("iEntryO", Math.max(1, Math.round(SPAN_YEARS)))}</p>${source("curve.date", "curve.creturn", "curve.bench_creturn")}`,
  };
}
function insightConc() {
  const f = F();
  return {
    title: t("qConc", CONC.k),
    body: `<div><div class="pfig ${toneCls(CONC.cagr_ex_best)}">${esc(f.spct(CONC.cagr_ex_best, 1))}</div><div class="mut num small">${esc(t("sConc", f.spct(CONC.cagr, 1)))}</div></div>
    <p class="prose">${t("iConcP", CONC.k, CONC.n_months)}</p>
    <div class="psec"><h3>${t("iConcCh")}</h3></div><div class="chart" id="insCh"></div>
    <div class="psec"><h3>${t("bestMonths")}</h3></div><div class="tiles">${BEST_MONTHS.map((m) => `<div class="mini num"><strong>${esc(f.month(m.y, m.m))}</strong><span style="color:var(--rise)">${esc(f.spct(m.r, 1))}</span></div>`).join("")}</div>${source("monthly.ret")}`,
  };
}
function insightCost() {
  const f = F(), C = SUM.costs, N2 = SUM.net_of_costs;
  if (!N2) {
    const est = C.annual_drag == null ? "" : `<div class="pgrid">${pair(`${t("m").mDrag[0]} (${t("estimate")})`, f.pct(C.annual_drag, 2))}</div><p class="dsrc">${t("estNote")}</p>`;
    return { title: t("qNet"), body: `${needsHtml("needsLedger", "msCost", true)}<p class="prose">${t("iCostP")}</p>${est}` };
  }
  return {
    title: t("qNet"),
    body: `<div><div class="pfig">${esc(f.pct(N2.cost_share, 0))}</div><div class="mut num small">${esc(t("sNet", f.spct(N2.gross_cagr, 1), f.spct(N2.net_cagr, 1)))}</div></div>
    <p class="prose">${t("iCostP")}</p><div class="psec"><h3>${t("iCostCh")}</h3></div><div class="chart" id="insCh"></div>${legend([{ n: t("grossLine"), c: "var(--muted)", dash: 1 }, { n: t("netLine"), c: "var(--rise)" }])}
    <div class="psec"><h3>${t("costBy")}</h3></div><div class="pgrid">${COST_KEYS.map((k) => pair(t("cost_" + k), f.pct(C[k], 2))).join("")}</div>${source("ledger_daily", "curve.gross_creturn")}`,
  };
}
function insightLimit() {
  const f = F(), LF = SUM.limit_fills, ST = SUM.structure;
  if (!LF) return { title: t("qLimit"), body: `${needsHtml("needsInput", "msLimit", true)}<p class="prose">${t("iLimitP")}</p>` };
  const rows = [];
  if (ST.entry_locked_n != null) rows.push(pair(t("locked"), f.int(ST.entry_locked_n)));
  if (ST.entry_touched_n != null) rows.push(pair(t("touched"), f.int(ST.entry_touched_n)));
  return {
    title: t("qLimit"),
    body: `<div><div class="pfig">${esc(f.pct(LF.share_return, 1))}</div><div class="mut num small">${esc(t("sLimit", f.pct(LF.share_trades, 1), t("approx")))}</div></div>
    <p class="prose">${t("iLimitP")}</p>
    ${table(["", t("tradeShare")], [[t("limEntry"), f.pct(ST.buy_high_ratio, 1)], [t("limExit"), f.pct(ST.sell_low_ratio, 1)]])}
    ${rows.length ? `<div class="pgrid">${rows.join("")}</div><p class="dsrc">${t("lockedD")} ${t("touchedD")}</p>` : ""}
    ${HAS_TRADES ? `<div><button class="link" id="insList">${t("listLimit")} &darr;</button></div>` : ""}${source("trades.limit_entry", "trades.limit_exit", "trades.ret", "trades.weight")}`,
  };
}
const INSIGHTS = { entry: insightEntry, conc: insightConc, cost: insightCost, limit: insightLimit };
function drawInsight(k) {
  const f = F(), el = $("insCh");
  if (el && k === "entry") {
    const series = [{ n: t("strategy"), c: "var(--rise)", v: ENTRY_V }];
    if (ENTRY_B) series.unshift({ n: t("bench"), c: "var(--muted)", dash: "3 3", w: 1, v: ENTRY_B });
    lineChart(el, { h: 190, zero: 1, b: LAST - SUM.entry_date.window_days, fmt: (v) => f.spct(v, 0), onClick: openDay, label: t("iEntryCh"), series });
  } else if (el && k === "conc") {
    barChart(el, { mode: "group", h: 170, cats: CONC_STEPS.map((_, i) => String(i)), tipTitle: (i) => `${t("removed")} ${i}`, fmt: (v) => f.spct(v, 1), label: t("iConcCh"), series: [{ n: t("annual"), c: (v) => (v < 0 ? "var(--fall)" : "var(--rise)"), v: CONC_STEPS }] });
  } else if (el && k === "cost") {
    lineChart(el, { h: 190, fmt: (v) => f.times(v), onClick: openDay, label: t("iCostCh"), series: [{ n: t("grossLine"), c: "var(--muted)", dash: "3 3", w: 1.2, v: GROSS }, { n: t("netLine"), c: "var(--rise)", v: EQ }] });
  }
  if ($("insList")) $("insList").onclick = () => {
    S.limitOnly = true; S.page = 0;
    closePanel(); renderTrades();
    $("tradesSec").scrollIntoView({ behavior: reduced() ? "auto" : "smooth", block: "start" });
  };
}

/* ---------- flags ---------- */
function flagValue(flag) {
  const f = F(), v = flag.value;
  if (v == null) return t("flagNa");
  if (flag.code === "capacity") return f.money(v);
  if (flag.code === "long_short_balance") return f.spct(v, 1);
  if (flag.code === "alpha_decay") return `${f.ratio(flag.recent)} / ${f.ratio(flag.full)}`;
  return f.pct(v, 1);
}
function flagsHtml() {
  return `<div class="flagrow"><span class="flaglab">${t("flags")} · ${t("wholePeriod")}</span>${SUM.flags.map((flag) => `<span class="fpill ${flag.level}" title="${esc(flag.rule)}"><i></i>${esc(t("flag_" + flag.code))} <b class="num">${esc(flagValue(flag))}</b></span>`).join("")}</div>`;
}

/* ---------- strategy review ---------- */
function metricFormat(value, fmt) {
  const f = F();
  if (fmt === "pct") return f.pct(value, 1);
  if (fmt === "wan") return f.money(value);
  if (fmt === "days") return f.days(Math.round(value));
  if (fmt === "int") return f.ratio(value, 1);
  return f.ratio(value, 2);
}
function engineRows(dim) {
  const labels = L().m;
  return P.quality[dim].checks.filter((c) => c.value != null && labels[c.key]).map((c) => ({
    n: labels[c.key][0], d: labels[c.key][1], val: metricFormat(c.value, c.fmt), std: TARGETS[c.key], ok: c.pass,
  }));
}
const plainRow = (key, val, extra) => ({ n: L().m[key][0], d: L().m[key][1], val, ok: null, ...extra });
const ledgerRow = (key) => plainRow(key, "—", { na: true });
const levelState = { green: "good", amber: "ok", red: "watch", na: "na" };
function healthGroups() {
  const f = F(), m = P.metrics, E = SUM.exposure, C = SUM.costs, T = SUM.turnover, B = SUM.benchmark, DD = SUM.drawdown;
  const avgRec = BT.mean(EPISODES.filter((e) => e.recovery_days != null).map((e) => e.recovery_days));
  const costFlag = SUM.flags.find((x) => x.code === "cost_sensitivity");
  const ledgerCost = C.source === "ledger", est = ledgerCost ? {} : { est: true };
  const excess = B?.cagr != null ? PERF.cagr - B.cagr : null;
  const gs = [
    { k: "profitability", sum: excess != null ? t("s_profitability", f.spct(PERF.cagr, 1), f.spct(excess, 1)) : t("s_profitabilityNoB", f.spct(PERF.cagr, 1)), m: engineRows("profitability"), draw: drawYears },
    { k: "risk", sum: t("s_risk", f.pct(-PERF.max_drawdown, 1), f.span(DD.longest_underwater_days)), m: [...engineRows("risk"), plainRow("mUnder", f.pct(DD.share_below_high, 0)), plainRow("mLong", f.span(DD.longest_underwater_days)), ...(avgRec != null ? [plainRow("mRec", f.days(Math.round(avgRec)))] : [])], draw: drawDrawdown },
    { k: "ratio", sum: t("s_ratio", f.ratio(PERF.sharpe)), m: engineRows("ratio"), draw: drawRisk },
    { k: "winrate", sum: SUM.trades.win_rate != null ? t("s_winrate", f.pct(SUM.trades.win_rate, 0)) : t("noTrades"), m: engineRows("winrate"), draw: drawWin },
    { k: "exposure", sum: HAS_EXPOSURE ? t("s_exposure", f.pct(E.avg_gross, 0), f.pct(E.avg_net, 0)) : t("s_exposureNo"), state: HAS_EXPOSURE ? "none" : "na", m: HAS_EXPOSURE ? [plainRow("mGross", f.pct(E.avg_gross, 0)), plainRow("mNet", f.pct(E.avg_net, 0)), plainRow("mCash", f.pct(E.avg_cash, 0))] : [ledgerRow("mGross"), ledgerRow("mNet"), ledgerRow("mCash")], draw: drawExposure },
    { k: "cost", sum: ledgerCost ? t("s_cost", f.pct(C.annual_drag, 2), f.times(T.annual_two_sided)) : t("s_costNo", f.pct(C.annual_drag, 2)), state: ledgerCost ? levelState[costFlag.level] : "na", m: [plainRow("mDrag", f.pct(C.annual_drag, 2), est), SUM.net_of_costs?.cost_share != null ? plainRow("mShare", f.pct(SUM.net_of_costs.cost_share, 0)) : ledgerRow("mShare"), ledgerCost ? plainRow("mTurn", f.times(T.annual_two_sided)) : plainRow("mEntry", f.times(T.annual_entry_weight), est), plainRow("mDouble", f.spct(C.annual_drag != null ? PERF.cagr - C.annual_drag : null, 1), est)], draw: drawCost },
    { k: "liquidity", sum: m.capacity != null ? t("s_liquidity", f.money(m.capacity)) : t("s_liquidityNo"), m: [...engineRows("liquidity"), ...(m.capacityMinLeg != null ? [plainRow("capMin", f.money(m.capacityMinLeg))] : []), ...(m.capacityAdv != null ? [plainRow("capAdv", f.money(m.capacityAdv))] : [])], draw: drawLiquidity },
  ];
  return gs.map((g) => {
    const judged = g.m.filter((x) => x.ok === true || x.ok === false), pass = judged.filter((x) => x.ok).length, ratio = judged.length ? pass / judged.length : 0;
    return { ...g, pass, judged: judged.length, state: g.state || (judged.length ? (ratio >= 0.8 ? "good" : ratio >= 0.6 ? "ok" : "watch") : "none") };
  });
}
const STATE_ICON = { good: ICON.dcheck, ok: ICON.check, watch: ICON.warn, na: ICON.warn };
function metricHtml(m) {
  const mark = m.ok === true ? "pass" : m.ok === false ? "warn" : "";
  const icon = m.ok === true ? ICON.check : m.ok === false ? ICON.warn : "";
  const note = m.na ? t("needsLedger") : m.std ? t("std", m.std) : "";
  return `<div class="metric"><span class="mk ${mark}">${icon}</span><span class="n">${esc(m.n)}${m.est ? `<span class="stag">${t("estimate")}</span>` : ""}</span><span class="d">${esc(m.d)}</span><span class="v">${esc(m.val)}</span><span class="t">${esc(note)}</span></div>`;
}
function renderHealth() {
  const gs = healthGroups(), cnt = (k) => gs.filter((g) => g.state === k).length;
  const rows = gs.map((g) => {
    const open = S.open.has(g.k), state = g.state === "none" ? "" : `<span class="state ${g.state}">${STATE_ICON[g.state]}${t(g.state)}</span>`;
    const body = open ? `<div class="hbody" id="body-${g.k}"><div class="passed num">${g.judged ? t("passed", g.pass, g.judged) : ""}</div>${g.m.map(metricHtml).join("")}<div class="sub" id="sub-${g.k}"></div></div>` : "";
    return `<button class="hrow" data-k="${g.k}" aria-expanded="${open}"><span class="tile">${GLYPH[g.k]}</span><span class="tx"><span class="q">${esc(t("q_" + g.k))}</span><span class="s">${esc(g.sum)}</span></span>${state}${ICON.chev}</button>${body}`;
  }).join("");
  $("healthSlot").innerHTML = `<section class="sec"><div class="shead"><h2>${t("health")}</h2><span class="side num">${t("tally", cnt("good"), cnt("ok"), cnt("watch"))}</span></div>${flagsHtml()}${S.win ? `<p class="scope">${t("reviewNote")}</p>` : ""}<div class="card">${rows}</div></section>`;
  document.querySelectorAll(".hrow").forEach((b) => {
    b.onclick = () => {
      const k = b.dataset.k;
      if (S.open.has(k)) S.open.delete(k); else S.open.add(k);
      renderHealth();
      document.querySelector(`.hrow[data-k="${k}"]`).focus({ preventScroll: true });
    };
  });
  gs.forEach((g) => { if (S.open.has(g.k)) g.draw($("sub-" + g.k)); });
}
const wholeTag = () => (S.win ? `<span class="r">${t("wholePeriod")}</span>` : "");
function drawYears(el) {
  const [first, last] = S.win ? [S.win.first, S.win.b] : [0, LAST];
  const f = F(), ys = YEAR_ROWS.filter((y) => y.b >= first && y.a <= last), hasB = !!BENCH, sel = ys.length - 1;
  const cap = (i) => `<span class="num">${ys[i].y}&nbsp;&nbsp;<span style="color:var(${ys[i].r < 0 ? "--fall" : "--rise"})">${t("strategy")} ${f.spct(ys[i].r, 1)}</span>${hasB ? `&nbsp;&nbsp;${t("bench")} ${f.spct(ys[i].bench, 1)}` : ""}</span>`;
  const beat = ys.filter((y) => y.bench != null && y.r > y.bench).length;
  const lg = [{ n: t("strategy"), c: "var(--rise)" }];
  if (hasB) lg.push({ n: t("bench"), c: "var(--faint)" });
  el.innerHTML = `<div class="subhead"><h3>${t("yearVs")}</h3>${hasB ? `<span class="r num">${t("yearVsSide", beat, ys.length)}</span>` : ""}</div><div class="caption" id="yCap">${cap(sel)}</div><div class="chart" id="yCh"></div>${legend(lg)}${S.win ? `<p class="dsrc">${t("wholeYears")}</p>` : ""}`;
  const series = [{ n: t("strategy"), c: (v) => (v < 0 ? "var(--fall)" : "var(--rise)"), v: ys.map((y) => y.r) }];
  if (hasB) series.push({ n: t("bench"), c: "var(--faint)", v: ys.map((y) => y.bench) });
  barChart($("yCh"), { mode: "group", cats: ys.map((y) => String(y.y).slice(2)), tipTitle: (i) => String(ys[i].y), label: t("yearVs"), fmt: (v) => f.spct(v, 0), sel, onHover: (i) => { $("yCap").innerHTML = cap(i); }, series });
}
function drawDrawdown(el) {
  if (!EPISODES.length) return;
  const f = F(), e = EPISODES[S.episode] || EPISODES[0], maxD = Math.abs(EPISODES[0].depth);
  const ym = (iso) => { const [y, m] = dateParts(iso); return f.month(y, m); };
  const a = dayOf(e.start), b = e.end ? dayOf(e.end) : LAST, [va, vb] = heroRange();
  const band = [Math.max(a, va), Math.min(b, vb)];
  const dd = EQ.map((v, i) => v / EQ[trackOf(EQ, 0, LAST).peak[i]] - 1);
  const bdd = BENCH ? BENCH.map((v, i) => v / BENCH[trackOf(BENCH, 0, LAST).peak[i]] - 1) : null;
  const bDepth = (x) => (bdd ? Math.min(...bdd.slice(dayOf(x.start), (x.end ? dayOf(x.end) : LAST) + 1)) : null);
  const series = [{ n: t("strategy"), c: "var(--fall)", v: dd }];
  const lg = [{ n: t("strategy"), c: "var(--fall)" }];
  if (bdd) { series.unshift({ n: t("bench"), c: "var(--faint)", v: bdd, w: 1 }); lg.push({ n: t("bench"), c: "var(--faint)" }); }
  el.innerHTML = `<div class="subhead"><div><h3 class="num">${esc(ym(e.trough))}</h3><span class="r num">${esc(f.date(e.start))} – ${e.end ? esc(f.date(e.end)) : "…"}</span></div><div class="rightcol"><div class="pfig dn big28">${esc(f.pct(e.depth, 1))}</div><span class="r num">${esc(t("ddDepth", f.days(e.days)))}</span></div></div>
  ${HAS_TRADES ? `<div><button class="link" id="ddList">${t("listTrades")} &darr;</button></div>` : ""}
  <div class="chart" id="ddCh"></div>${legend(lg)}
  <div class="subhead"><h3>${t("ddRanks")}</h3>${wholeTag()}</div><div class="ranks">${EPISODES.slice(0, 5).map((x, i) => {
    const bd = bDepth(x);
    return `<button class="rank" data-i="${i}" aria-pressed="${x === e}"><span class="num">${esc(ym(x.trough))}</span><span class="tr"><b style="width:${Math.abs(x.depth) / maxD * 100}%;background:var(--fall)"></b>${bd != null ? `<b style="width:${Math.min(100, Math.abs(bd) / maxD * 100)}%;background:var(--faint)"></b>` : ""}</span><span class="val"><span style="font-size:13px;color:var(--fall)">${esc(f.pct(x.depth, 1))}</span><span>${esc(f.days(x.days))}</span></span></button>`;
  }).join("")}</div>`;
  lineChart($("ddCh"), { h: 190, zero: 1, a: va, b: vb, fmt: (v) => f.pct(v, 0), band: band[1] > band[0] ? band : null, label: t("underwater"), series, onClick: openDay });
  if ($("ddList")) $("ddList").onclick = () => {
    setWindow(makeWin(a + 1, b, { kind: "range" }));
    $("tradesSec").scrollIntoView({ behavior: reduced() ? "auto" : "smooth", block: "start" });
  };
  el.querySelectorAll(".rank").forEach((btn) => { btn.onclick = () => { S.episode = +btn.dataset.i; drawDrawdown(el); }; });
}
function drawRisk(el) {
  const f = F(), tab = S.sub.risk || "sharpe", w = +(S.sub.riskW || "252");
  el.innerHTML = `<div class="subhead">${subTabs("risk", [["sharpe", t("sharpeTab")], ["vol", t("volTab")]], "sharpe")}${subTabs("riskW", [["252", t("w1y")], ["63", t("w3m")]], "252")}</div><div class="chart" id="rCh"></div>`;
  const sharpe = tab === "sharpe", [a, b] = heroRange();
  lineChart($("rCh"), { h: 190, a, b, zero: sharpe, fmt: sharpe ? (v) => f.ratio(v, 1) : (v) => f.pct(v, 0), label: t(sharpe ? "sharpeTab" : "volTab"), series: [{ n: t(sharpe ? "sharpeTab" : "volTab"), c: "var(--rise)", v: BT.rolling(RETS, w, sharpe ? "sharpe" : "vol", RF) }] });
  bindSub(el, () => drawRisk(el));
}
function drawWin(el) {
  const f = F(), ST = SUM.structure, parts = [], W = windowFigures();
  const H2 = P.hist && W.n ? BT.windowHist(P.hist, HIST_DAYS, ...dateRange()) : null;
  const sideRow = (name, s) => (s ? [name, f.int(s.n), f.pct(s.win_rate, 1), f.spct(s.avg_ret, 2), f.spct(s.contrib, 1)] : null);
  if (H2) {
    parts.push(`<div class="subhead"><h3>${t("dist")}</h3><span class="r num">${t("distSide", f.spct(W.mean, 1))}</span></div><div class="chart" id="wCh"></div>`);
  }
  if (W.short) {
    const rows = [sideRow(t("long"), W.long), sideRow(t("short"), W.short)].filter(Boolean);
    parts.push(`<div class="subhead"><h3>${t("sides")}</h3></div>${table([t("side"), t("nTrades"), t("winCol"), t("avgCol"), t("contribCol")], rows)}`);
  }
  if (!S.win && ST.top10_contrib_ratio != null) parts.push(`<p class="caption">${esc(t("top10", f.pct(ST.top10_contrib_ratio, 0)))}</p>`);
  const points = windowClosed().filter((x) => x.mae != null && x.ret != null);
  if (points.length) {
    parts.push(`<div class="subhead"><h3>${t("scatter")}</h3></div><p class="caption">${t("scatterD")}${esc(sampledNote("trades"))}</p><div class="chart" id="scCh"></div>`);
  }
  if (W.stops.length) {
    const replay = W.stops.map((r, i) => [f.pct(P.stops[i], 0), f.int(r.hit), f.pct(r.win, 1), f.spct(r.avg, 2)]);
    parts.push(`<div class="subhead"><h3>${t("stops")}</h3></div>${table([t("stopCol"), t("stopHit"), t("stopWin"), t("stopAvg")], replay)}<p class="dsrc">${t("stopsD")}</p>`);
  }
  el.innerHTML = parts.join("");
  if ($("wCh")) {
    const cats = H2.counts.map((_, i) => H2.low + i * H2.width), neg = H2.counts.map((c, i) => (cats[i] + H2.width / 2 < 0 ? c : 0)), pos = H2.counts.map((c, i) => (cats[i] + H2.width / 2 >= 0 ? c : 0));
    barChart($("wCh"), { mode: "stack", h: 180, cats: cats.map((c) => f.spct(c, 0)), label: t("dist"), fmt: (x) => f.int(x), tipTitle: (i) => `${f.spct(cats[i], 1)} … ${f.spct(cats[i] + H2.width, 1)}`, series: [{ n: t("tradesAxis"), c: "var(--fall)", v: neg }, { n: t("tradesAxis"), c: "var(--rise)", v: pos }] });
  }
  if ($("scCh")) scatterChart($("scCh"), points, (p) => openPanel({ type: "trade", id: p.id }));
}
const view = () => ({ a: heroRange()[0], b: heroRange()[1] });
function drawExposure(el) {
  const f = F();
  const held = () => lineChart($("eCh"), { h: 200, zero: 1, ...view(), fmt: (v) => String(Math.round(v)), onClick: openDay, label: t("expHold"), series: [{ n: t("held"), c: "var(--c1)", v: HELD }] });
  if (!HAS_EXPOSURE) {
    el.innerHTML = `${needsHtml("needsLedger", "msExp", true)}${HAS_TRADES ? `<div class="subhead"><h3>${t("expHold")}</h3></div><div class="chart" id="eCh"></div>` : ""}`;
    if (HAS_TRADES) held();
    return;
  }
  const tab = S.sub.exp || "ls";
  el.innerHTML = `<div class="subhead">${subTabs("exp", [["ls", t("expLS")], ["cash", t("expCash")], ["hold", t("expHold")]], "ls")}</div><div class="chart" id="eCh"></div><div id="eLg"></div>`;
  const share = (col) => ledgerSeries((r) => (r.equity > 0 ? r[col] / r.equity : null));
  if (tab === "ls") {
    lineChart($("eCh"), { h: 200, zero: 1, ...view(), fmt: (v) => f.pct(v, 0), onClick: openDay, label: t("expLS"), series: [{ n: t("long"), c: "var(--c1)", v: share("gross_long") }, { n: t("short"), c: "var(--c2)", v: share("gross_short") }, { n: t("net"), c: "var(--ink)", v: share("net_exposure"), dash: "3 3", w: 1.2 }] });
    $("eLg").innerHTML = legend([{ n: t("long"), c: "var(--c1)" }, { n: t("short"), c: "var(--c2)" }, { n: t("net"), c: "var(--ink)", dash: 1 }]);
  } else if (tab === "cash") {
    lineChart($("eCh"), { h: 200, zero: 1, ...view(), fmt: (v) => f.pct(v, 0), onClick: openDay, label: t("expCash"), series: [{ n: t("cash"), c: "var(--c3)", v: share("cash") }] });
  } else held();
  bindSub(el, () => drawExposure(el));
}
/* Periods for totals inside the range: calendar years, or months under two years. */
function chartPeriods() {
  const first = S.win ? S.win.first : 0, last = S.win ? S.win.b : LAST, f = F();
  const short = BT.dayDiff(DAYS[first], DAYS[last]) < 730;
  const periods = short
    ? MONTH_ROWS.map((m) => ({ a: m.a, b: m.b, cat: `${String(m.y).slice(2)}/${m.m}`, title: f.month(m.y, m.m) }))
    : YEAR_ROWS.map((y) => ({ a: y.a, b: y.b, cat: String(y.y).slice(2), title: String(y.y) }));
  return BT.periodsIn(periods, first, last);
}
function periodSum(periods, fn) {
  return periods.map((p) => {
    let total = 0;
    for (let i = p.a; i <= p.b; i++) { const r = LEDGER.get(DAYS[i]); if (r) total += fn(r) ?? 0; }
    return total;
  });
}
function drawCost(el) {
  const f = F();
  if (SUM.costs.source !== "ledger") {
    el.innerHTML = `${needsHtml("needsLedger", "msCost", true)}<p class="dsrc">${t("estNote")}</p>`;
    return;
  }
  const tab = S.sub.cost || "cost", periods = chartPeriods(), cats = periods.map((p) => p.cat), title = (i) => periods[i].title;
  el.innerHTML = `<div class="subhead">${subTabs("cost", [["cost", t("costTab")], ["turn", t("turnTab")]], "cost")}<span class="r">${t(tab === "cost" ? "costNote" : "turnNote")}</span></div><div class="chart" id="cCh"></div><div id="cLg"></div>`;
  if (tab === "cost") {
    const colors = ["--c1", "--c2", "--c3", "--c4", "--compare"];
    const parts = COST_KEYS.map((k, i) => ({ n: t("cost_" + k), c: `var(${colors[i]})`, v: periodSum(periods, (r) => (r.equity > 0 ? r[k] / r.equity : 0)) }));
    barChart($("cCh"), { mode: "stack", h: 200, cats, tipTitle: title, fmt: (v) => f.pct(v, 1), label: t("costTab"), series: parts });
    $("cLg").innerHTML = legend(parts);
  } else {
    barChart($("cCh"), { mode: "stack", h: 200, cats, tipTitle: title, fmt: (v) => f.times(v), label: t("turnTab"), series: [{ n: t("turnover"), c: "var(--c1)", v: periodSum(periods, (r) => r.turnover) }] });
  }
  bindSub(el, () => drawCost(el));
}
function drawLiquidity(el) {
  const f = F(), parts = [];
  const SC = P.fill_scenarios ? rowsOf(P.fill_scenarios) : null;
  if (SC) {
    const order = { baseline: 0, locked: 1, at_limit: 2 };
    const rows = [...SC].sort((a, b) => (order[a.name] ?? 9) - (order[b.name] ?? 9)).map((s) => [I18N[S.lang]["sc_" + s.name] || s.name, f.spct(s.cagr, 1), f.pct(s.max_drawdown, 1), f.ratio(s.daily_sharpe), f.int(s.blocked_n)]);
    parts.push(`<div class="subhead"><h3>${t("fillTitle")}</h3>${wholeTag()}</div>${table([t("scenario"), t("scCagr"), t("scMdd"), t("scSharpe"), t("scBlocked")], rows)}<p class="dsrc">${t("fillD")}</p>`);
  }
  if (!has("blocks")) {
    parts.push(needsHtml("needsLedger", "msBlocks", true));
  } else if (P.embed.blocks?.rows_total) {
    /* Whole-period counts cover every block; a date window can only count the rows it has. */
    const rows = BLOCK_ROWS.filter((b) => inWindow(dayOf(b.date))), sel = S.blockReason;
    const counts = new Map(S.win ? [] : Object.entries(SUM.execution?.blocks_by_reason ?? {}));
    if (S.win) rows.forEach((b) => counts.set(b.reason, (counts.get(b.reason) || 0) + 1));
    const keys = [...counts.keys()].sort((a, b) => counts.get(b) - counts.get(a)), max = counts.get(keys[0]) || 1;
    const list = rows.filter((b) => !sel || b.reason === sel).slice(-6).reverse();
    const total = S.win ? rows.length : P.embed.blocks.rows_total;
    parts.push(`<div class="subhead"><h3>${t("blockTitle")}</h3><span class="r num">${t("blockSide", f.int(total))}</span></div>
    <div class="ranks">${keys.map((k) => `<button class="rank blockrow" data-k="${esc(k)}" aria-pressed="${sel === k}"><span>${esc(k)}</span><span class="tr"><b style="width:${counts.get(k) / max * 100}%;background:var(--amber)"></b></span><span class="val">${f.int(counts.get(k))}</span></button>`).join("")}</div>
    <div class="subhead"><h3>${t("recent")}</h3></div><div>${list.map(blockRow).join("")}</div><p class="dsrc">${esc(sampledNote("blocks"))}</p>`);
  }
  el.innerHTML = parts.join("");
  el.querySelectorAll(".blockrow").forEach((b) => { b.onclick = () => { S.blockReason = S.blockReason === b.dataset.k ? null : b.dataset.k; drawLiquidity(el); }; });
  el.querySelectorAll("[data-day]").forEach((b) => { b.onclick = () => openDay(dayOf(b.dataset.day)); });
}
function blockRow(b) {
  const f = F();
  return `<button class="fill" data-day="${esc(b.date)}"><span class="nm2">${esc(shown(b.symbol))} <span class="fnt num">${esc(b.symbol)}</span></span><span class="rv">${esc(amount(b.notional))}</span><span class="sm num">${esc(f.date(b.date))}${b.side ? " · " + esc(b.side) : ""}</span><span class="sm blocked right">${esc(b.reason)}</span></button>`;
}

/* ---------- monthly heatmap ---------- */
function renderMonthly() {
  const f = F(), mx = Math.max(...MONTH_ROWS.map((m) => Math.abs(m.r)), 1e-9), byKey = Object.fromEntries(MONTH_ROWS.map((m) => [m.k, m]));
  let g = `<div class="h"></div>` + Array.from({ length: 12 }, (_, i) => `<div class="h">${i + 1}</div>`).join("") + `<div class="h">${t("fullYear")}</div>`;
  [...YEARS].reverse().forEach((y) => {
    g += `<div class="yl">${y}</div>`;
    for (let m = 1; m <= 12; m++) {
      const k = `${y}-${pad2(m)}`, o = byKey[k];
      if (!o) { g += `<div class="cell none"></div>`; continue; }
      const al = 0.16 + 0.66 * Math.pow(Math.min(1, Math.abs(o.r) / (mx * 0.82)), 0.95), lab = f.month(y, m);
      g += `<button class="cell${o.b - o.a + 1 < 15 ? " partial" : ""}" data-k="${k}" aria-pressed="${S.win?.month === k}" aria-label="${esc(lab + " " + f.spct(o.r, 1))}" style="--tone:var(${o.r < 0 ? "--fall" : "--rise"});--a:${al.toFixed(3)}">${f.cell(o.r)}</button>`;
    }
    const yo = YEAR_ROWS.find((o) => o.y === y);
    g += `<div class="yt ${toneCls(yo.r)}">${f.spct(yo.r, 0)}</div>`;
  });
  $("monthSlot").innerHTML = `<section class="sec"><div class="shead"><h2>${t("monthly")}</h2><span class="side num">${t("monthlySide", MONTH_ROWS.filter((m) => m.r > 0).length, MONTH_ROWS.length)}</span></div><div class="card heat"><div class="hnote" id="hnote"></div><div class="hgrid">${g}</div></div></section>`;
  document.querySelectorAll("button.cell").forEach((c) => {
    const k = c.dataset.k;
    c.onpointerenter = (e) => { if (e.pointerType === "mouse") { S.hoverMonth = k; monthNote(); } };
    c.onpointerleave = () => { if (S.hoverMonth) { S.hoverMonth = null; monthNote(); } };
    c.onclick = () => toggleMonth(k);
  });
  monthNote();
}
function monthNote() {
  const f = F(), e = $("hnote"), k = S.hoverMonth || S.win?.month;
  if (!e) return;
  if (!k) { e.className = "hnote"; e.textContent = t("clickMonth"); return; }
  const o = MONTH_ROWS.find((m) => m.k === k);
  e.className = "hnote on";
  e.style.setProperty("--tone", `var(${o.r < 0 ? "--fall" : "--rise"})`);
  const n = windowFigures().n;
  e.innerHTML = `<span>${esc(f.month(o.y, o.m))} ${esc(f.spct(o.r, 1))}</span>` + (S.win?.month === k && HAS_TRADES ?`<button class="link" id="toTrades">${t("seeTrades", n)} &darr;</button>` : "");
  if ($("toTrades")) $("toTrades").onclick = () => $("tradesSec").scrollIntoView({ behavior: reduced() ? "auto" : "smooth", block: "start" });
}

/* ---------- trades ---------- */
/* A trade belongs to the window its exit falls in. */
const windowClosed = () => (S.win ? CLOSED.filter((x) => inWindow(x.x)) : CLOSED);
const dateRange = () => (S.win ? [DAYS[S.win.first], DAYS[S.win.b]] : [DAYS[0], DAYS[LAST]]);
/* Complete figures for the range, from every trade, whatever rows this file holds. */
const windowFigures = () => BT.windowTrades(TRADE_DAYS, ...dateRange(), P.stops);
function tradeList() {
  let l = windowClosed();
  if (S.limitOnly) l = l.filter((x) => x.limit_entry || x.limit_exit);
  if (S.sort === "win") l = [...l].sort((a, b) => (b.ret ?? 0) - (a.ret ?? 0));
  else if (S.sort === "lose") l = [...l].sort((a, b) => (a.ret ?? 0) - (b.ret ?? 0));
  return l;
}
const lastYear = DAYS[LAST].slice(0, 4);
function dshort(iso) { const f = F(); return iso.startsWith(lastYear + "-") ? f.dateShort(iso) : f.date(iso); }
function tradeRow(x) {
  const f = F();
  const tags = (x.side === "short" ? `<span class="stag">${t("shortTag")}</span>` : "") + (x.n_segments > 1 ? `<span class="stag">${esc(t("segTag", x.n_segments))}</span>` : "");
  const when = x.exit_date ? `${esc(dshort(x.entry_date))} → ${esc(dshort(x.exit_date))} · ${esc(f.days(BT.dayDiff(x.entry_date, x.exit_date)))}` : esc(dshort(x.entry_date));
  return `<button class="row" data-t="${x.id}"><span class="ava" aria-hidden="true">${initials(x.symbol)}</span><span class="l"><span class="nm" style="display:block">${esc(shown(x.symbol))}<span class="cd num">${esc(x.symbol)}</span>${tags}</span><span class="m num" style="display:block">${when}</span></span><span class="r"><span class="f ${toneCls(x.ret ?? 0)}" style="display:block">${esc(f.spct(x.ret, 1))}</span><span class="p" style="display:block">${f.price(x.entry_price)} → ${f.price(x.exit_price)}</span></span></button>`;
}
function renderTrades() {
  const f = F(), l = tradeList(), per = 20, pages = Math.max(1, Math.ceil(l.length / per));
  S.page = clamp(S.page, 0, pages - 1);
  const a = S.page * per, items = l.slice(a, a + per);
  let chip = "";
  const nums = [...new Set([0, S.page - 1, S.page, S.page + 1, pages - 1].filter((i) => i >= 0 && i < pages))].sort((x, y) => x - y);
  let prev = -1;
  const pg = nums.map((i) => { const gap = i - prev > 1 ? "<span>…</span>" : ""; prev = i; return `${gap}<button data-p="${i}" ${i === S.page ? 'aria-current="page"' : ""}>${i + 1}</button>`; }).join("");
  const empty = t(HAS_TRADES ? "noTrades" : "noTradesFile");
  $("tradesSlot").innerHTML = `<section class="sec" id="tradesSec" style="scroll-margin-top:56px"><div class="shead"><div class="ttl"><h2>${t("tradeTitle")}</h2><span class="side num">${t("tradeTotal", S.limitOnly ? l.length : windowFigures().n)}</span></div><div class="seg sm" id="sortSeg">${[["recent", "sortRecent"], ["win", "sortWin"], ["lose", "sortLose"]].map((o) => `<button data-s="${o[0]}" aria-pressed="${S.sort === o[0]}">${t(o[1])}</button>`).join("")}</div></div>
  ${S.win || S.limitOnly ? `<div class="chiprow">${S.win ? `<button class="fchip num" id="fchipWin" aria-label="${t("clear")}">${esc(winLabel(S.win))} ${ICON.x}</button>` : ""}${S.limitOnly ? `<button class="fchip" id="fchipLim" aria-label="${t("clear")}">${t("limitChip")} ${ICON.x}</button>` : ""}</div>` : ""}
  <div class="card">${items.length ? items.map(tradeRow).join("") : `<div class="empty" style="padding:20px">${empty}</div>`}
  <div class="tfoot num"><span>${t("closedOnly")}${esc(sampledNote("trades"))}</span><span>${l.length ? t("range", a + 1, Math.min(a + per, l.length), l.length) : ""}</span></div>
  ${pages > 1 ? `<div class="pager num"><button data-p="${S.page - 1}" aria-label="${t("prev")}" ${S.page === 0 ? "disabled" : ""}>‹</button>${pg}<button data-p="${S.page + 1}" aria-label="${t("next")}" ${S.page >= pages - 1 ? "disabled" : ""}>›</button></div>` : ""}</div></section>`;
  document.querySelectorAll("#sortSeg button").forEach((b) => { b.onclick = () => { S.sort = b.dataset.s; S.page = 0; renderTrades(); }; });
  document.querySelectorAll(".pager [data-p]").forEach((b) => { b.onclick = () => { S.page = +b.dataset.p; renderTrades(); $("tradesSec").scrollIntoView({ block: "start" }); }; });
  document.querySelectorAll("#tradesSec .row").forEach((b) => { b.onclick = () => openPanel({ type: "trade", id: +b.dataset.t }); });
  if ($("fchipWin")) $("fchipWin").onclick = () => setWindow(null);
  if ($("fchipLim")) $("fchipLim").onclick = () => { S.limitOnly = false; S.page = 0; renderTrades(); };
}
function sampledNote(name) {
  const e = P.embed[name];
  if (!e || e.rows_embedded >= e.rows_total) return "";
  return ` ${t(name === "trades" ? "sampled" : "sampledDays", e.rows_embedded, e.rows_total)}`;
}

/* ---------- holdings view ---------- */
function renderHoldings() {
  const f = F(), hs = OPEN.filter((x) => x.ret != null);
  if (!hs.length) {
    $("main").innerHTML = `<section class="hero"><div class="eyebrow">${t("noHold")}</div></section>`;
    return;
  }
  const avg = BT.mean(hs.map((x) => x.ret)), up = hs.filter((x) => x.ret > 0).length, mx = Math.max(...hs.map((x) => Math.abs(x.ret)), 1e-9);
  const weight = (x) => Math.abs(x.weight ?? 0);
  const tiles = [...hs].sort((a, b) => weight(b) - weight(a)).map((x) => `<button class="tt" data-t="${x.id}" style="--tone:var(${x.ret < 0 ? "--fall" : "--rise"});--a:${(26 + 60 * Math.abs(x.ret) / mx).toFixed(0)}%;background:color-mix(in oklab,var(--tone) var(--a),var(--well))"><span class="a">${esc(shown(x.symbol))}</span><span class="b">${esc(f.spct(x.ret, 1))}</span></button>`).join("");
  const sorters = { pnl: (a, b) => b.ret - a.ret, weight: (a, b) => weight(b) - weight(a), entry: (a, b) => b.e - a.e };
  const sorted = [...hs].sort(sorters[S.holdSort]);
  $("main").innerHTML = `<section class="hero"><span class="glow ${avg < 0 ? "fall" : "rise"} lit"></span><div class="hhead"><div class="eyebrow">${t("holdTitle", hs.length)}</div><button class="chip" id="hideN">${t(S.hideNames ? "showNames" : "hideNames")}</button></div>
  <div class="figure"><div class="big ${toneCls(avg)}">${bigHtml(f.spct(avg, 2))}</div><div class="vs">${t("holdRet")}</div></div>
  <p class="lead">${esc(t("holdLead", up, hs.length - up))}</p><div class="tree">${tiles}</div></section>
  <section class="sec"><div class="shead"><div class="ttl"><h2>${t("holdDetail")}</h2><span class="side num">${hs.length}</span></div><div class="seg sm" id="holdSort">${[["pnl", "sortPnl"], ["weight", "sortWeight"], ["entry", "sortEntry"]].map((o) => `<button data-s="${o[0]}" aria-pressed="${S.holdSort === o[0]}">${t(o[1])}</button>`).join("")}</div></div>
  <div class="card">${sorted.map((x) => `<button class="row" data-t="${x.id}"><span class="ava" aria-hidden="true">${initials(x.symbol)}</span><span class="l"><span class="nm" style="display:block">${esc(shown(x.symbol))}<span class="cd num">${esc(x.symbol)}</span>${x.side === "short" ? `<span class="stag">${t("shortTag")}</span>` : ""}</span><span class="m num" style="display:block">${esc(t("holdMeta", dshort(x.entry_date), f.pct(weight(x), 1)))}</span></span><span class="r"><span class="f ${toneCls(x.ret)}" style="display:block">${esc(f.spct(x.ret, 2))}</span></span></button>`).join("")}</div></section>`;
  document.querySelectorAll("#holdSort button").forEach((b) => { b.onclick = () => { S.holdSort = b.dataset.s; renderHoldings(); }; });
  $("hideN").onclick = () => { S.hideNames = !S.hideNames; renderHoldings(); };
  document.querySelectorAll("[data-t]").forEach((b) => { b.onclick = () => openPanel({ type: "trade", id: +b.dataset.t }); });
}

/* ---------- detail panel ---------- */
function openPanel(p) { S.panel = p; document.body.classList.add("panel-open"); renderPanel(); }
function closePanel() {
  S.panel = null; S.pin = null; S.cursor = null;
  document.body.classList.remove("panel-open");
  renderPanel();
  updateHero();
}
function openDay(i) { S.pin = i; S.cursor = i; updateHero(); openPanel({ type: "day", i }); }
const shell = (head, body) => `<div class="phead">${head}<button class="iconbtn" id="pClose" aria-label="${t("close")}">${ICON.x}</button></div><div class="pbody">${body}</div>`;
function fillRow(x) {
  const f = F(), buy = (x.notional ?? 0) >= 0;
  return `<div class="fill"><span class="nm2">${esc(shown(x.symbol))} <span class="fnt num">${esc(x.symbol)}</span></span><span class="rv">${esc(amount(x.notional == null ? null : Math.abs(x.notional)))}</span><span class="sm num">${t(buy ? "buy" : "sell")} @ ${f.price(x.price)}${x.reason ? " · " + esc(x.reason) : ""}</span><span class="sm num right">${x.cost != null ? `${t("dayCost")} ${esc(amount(x.cost))}` : ""}</span></div>`;
}
const copyLine = (code) => `<div class="codeline"><code>${esc(code)}</code><button class="link" data-copy="${esc(code)}">${t("copy")}</button></div>`;
const MAX_LISTED = 60;
function holdingsAt(i) {
  return TRADES.filter((x) => x.e != null && x.e <= i && (x.x == null || x.x > i));
}
function dayBody(i) {
  const f = F(), date = DAYS[i], idx = DAY_INDEX.get(date) || {}, led = LEDGER && LEDGER.get(date);
  const T = trackOf(EQ, 0, LAST), dd = EQ[i] / EQ[T.peak[i]] - 1, fs = FILLS.get(date) || [], bs = BLOCKS.get(date) || [], cs = CASH.get(date) || [];
  const hold = holdingsAt(i), cells = [pair(t("cumulative"), f.spct(EQ[i] / EQ[0] - 1, 1)), pair(t("drawdown"), f.pct(dd, 1))];
  if (BENCH) cells.splice(1, 0, pair(t("benchDay"), f.spct(i ? BENCH[i] / BENCH[i - 1] - 1 : 0, 2)));
  if (led) {
    if (led.gross_exposure != null && led.equity > 0) cells.push(pair(t("gross"), f.pct(led.gross_exposure / led.equity, 0)));
    if (idx.cost_total != null) cells.push(pair(t("dayCost"), amount(idx.cost_total)));
    if (idx.traded_notional != null) cells.push(pair(t("traded"), amount(idx.traded_notional)));
  }
  const parts = [`<div><div class="pfig ${toneCls(RETS[i])}">${esc(f.spct(RETS[i], 2))}</div><div class="mut small">${t("dayReturn")}</div></div><div class="pgrid">${cells.join("")}</div>`];
  /* day_index counts every row, so a day left out of this file says so instead of looking quiet. */
  const nFills = idx.n_fills ?? fs.length, nBlocks = idx.n_blocks ?? bs.length, nCash = idx.n_cashflows ?? cs.length;
  const omitted = nFills > fs.length || nBlocks > bs.length || nCash > cs.length;
  if (has("fills")) {
    const list = fs.length ? fs.map(fillRow).join("") : nFills ? "" : `<div class="empty">${t("noFills")}</div>`;
    parts.push(`<div class="psec"><h3>${t("dayFills")}<span class="num">${f.int(nFills)}</span></h3>${list}</div>`);
  } else {
    const events = TRADES.flatMap((x) => [x.e === i ? [x, "enter"] : null, x.x === i ? [x, "exit"] : null].filter(Boolean));
    parts.push(needsHtml("needsLedger", "msDay", true));
    parts.push(`<div class="psec"><h3>${t("dayFills")}<span class="num">${events.length}</span></h3><p class="dsrc pad">${t("tradesOnly")}</p>${events.length ? events.map(([x, k]) => `<button class="fill" data-t="${x.id}"><span class="nm2">${esc(shown(x.symbol))} <span class="fnt num">${esc(x.symbol)}</span></span><span class="rv">${t(k)}</span></button>`).join("") : `<div class="empty">${t("noFills")}</div>`}</div>`);
  }
  if (nBlocks) parts.push(`<div class="psec"><h3>${t("dayBlocked")}<span class="num">${f.int(nBlocks)}</span></h3>${bs.map(blockRow).join("")}</div>`);
  if (nCash) parts.push(`<div class="psec"><h3>${t("dayCash")}<span class="num">${f.int(nCash)}</span></h3>${cs.map((c) => `<div class="fill"><span class="nm2">${esc(c.symbol)}</span><span class="rv">${esc(amount(c.amount))}</span><span class="sm">${esc(c.kind ?? "")}</span></div>`).join("")}</div>`);
  if (omitted) parts.push(`<p class="prose">${t("notEmbedded")}</p>`);
  /* The count comes from complete daily entries and exits; the names from the rows in this file. */
  const listed = hold.slice(0, MAX_LISTED), partial = P.coverage.trades === "present" && listed.length < HELD[i];
  if (P.coverage.trades === "present") parts.push(`<div class="psec"><h3>${t("holdings")}<span class="num">${esc(f.names(HELD[i]))}</span></h3><div class="pchips pad">${listed.map((x) => `<button class="pchip" data-t="${x.id}"><span class="ava">${initials(x.symbol)}</span>${esc(shown(x.symbol))}</button>`).join("")}</div></div>`);
  if (partial) parts.push(`<p class="dsrc">${esc(t("partialNames", listed.length, HELD[i]))}</p>`);
  if (omitted || partial) parts.push(copyLine(`db.day("${date}")`));
  return parts.join("");
}
function tradeBody(x) {
  const f = F(), cells = [
    pair(t("entered"), f.price(x.entry_price)), pair(t("exited"), x.exit_price != null ? f.price(x.exit_price) : t("inBook")),
    pair(t("held2"), x.exit_date ? f.days(BT.dayDiff(x.entry_date, x.exit_date)) : "—"), pair(t("sideLabel"), t(x.side === "short" ? "short" : "long")),
  ];
  if (x.mae != null) cells.push(pair(t("worstOpen"), f.spct(x.mae, 1)));
  if (x.gmfe != null) cells.push(pair(t("bestOpen"), f.spct(x.gmfe, 1)));
  const fs = FILL_ROWS.filter((k) => k.symbol === x.symbol && k.date >= x.entry_date && (!x.exit_date || k.date <= x.exit_date));
  const fills = has("fills") ? (P.detail === "full" ? `<div class="psec"><h3>${t("dayFills")}<span class="num">${fs.length}</span></h3>${fs.map(fillRow).join("")}<p class="dsrc pad">${esc(sampledNote("fills"))}</p></div>` : "") : needsHtml("needsLedger", "msFills", true);
  const days = [x.e, x.x].filter((d) => d != null).map((d) => `<button class="chip num" data-day="${DAYS[d]}">${esc(t("openDay", f.date(DAYS[d])))}</button>`).join("");
  return `<div><div class="pfig ${toneCls(x.ret ?? 0)}">${esc(f.spct(x.ret, 1))}</div><div class="mut num small">${esc(dshort(x.entry_date))}${x.exit_date ? " → " + esc(dshort(x.exit_date)) : ""}</div></div><div class="pgrid">${cells.join("")}</div>${fills}<div class="chiprow">${days}</div>`;
}
function symbolBody(code) {
  const f = F(), tr = CLOSED.filter((x) => x.symbol === code), op = OPEN.find((x) => x.symbol === code), row = SYMBOL_ROWS.find((r) => r.symbol === code);
  return `<div class="pgrid">${pair(t("nTrades"), f.int(row ? row.n_trades : tr.length))}${pair(t("winRate"), row?.win_rate != null ? f.pct(row.win_rate, 0) : "—")}${pair(t("contrib"), row ? f.spct(row.contrib_proxy, 1) : "—")}${pair(t("inBook"), op ? f.spct(op.ret, 1) : t("none"))}</div>
  <div class="psec"><h3>${t("tradeTitle")}<span class="num">${t("symbolTrades", tr.length)}</span></h3><div>${tr.slice(0, 30).map(tradeRow).join("")}</div></div>${copyLine(`db.symbol(${JSON.stringify(code)})`)}`;
}
function renderPanel() {
  const el = $("panel"), p = S.panel;
  el.classList.toggle("open", !!p);
  if (!p) { el.innerHTML = ""; return; }
  const f = F();
  if (p.type === "insight") {
    const o = INSIGHTS[p.k]();
    el.innerHTML = shell(`<div class="tt2">${esc(o.title)}</div>`, (S.win ? `<p class="scope">${t("wholeNote")}</p>` : "") + o.body);
    drawInsight(p.k);
  } else if (p.type === "day") {
    const i = p.i;
    el.innerHTML = shell(`<button class="iconbtn" id="pPrev" aria-label="${t("prevDay")}" ${i <= 0 ? "disabled" : ""}>${ICON.left}</button><button class="iconbtn" id="pNext" aria-label="${t("nextDay")}" ${i >= LAST ? "disabled" : ""}>${ICON.right}</button><div class="tt2 num">${esc(f.date(DAYS[i]))}</div>`, dayBody(i));
    $("pPrev").onclick = () => openDay(i - 1);
    $("pNext").onclick = () => openDay(i + 1);
    try { history.replaceState(null, "", `#day=${DAYS[i]}`); } catch (e) { /* srcdoc iframe */ }
  } else if (p.type === "trade") {
    const x = TRADES[p.id];
    el.innerHTML = shell(`<span class="ava">${initials(x.symbol)}</span><div class="tt2">${esc(shown(x.symbol))} <span class="fnt num">${esc(x.symbol)}</span></div>`, tradeBody(x));
  } else {
    el.innerHTML = shell(`<span class="ava">${initials(p.code)}</span><div class="tt2">${esc(shown(p.code))} <span class="fnt num">${esc(p.code)}</span></div>`, symbolBody(p.code));
  }
  el.querySelectorAll("[data-t]").forEach((b) => { b.onclick = () => openPanel({ type: "trade", id: +b.dataset.t }); });
  el.querySelectorAll("[data-day]").forEach((b) => { b.onclick = () => openDay(dayOf(b.dataset.day)); });
  el.querySelectorAll("[data-copy]").forEach((b) => {
    b.onclick = () => {
      try { navigator.clipboard.writeText(b.dataset.copy); b.textContent = t("copied"); } catch (e) { /* no clipboard here */ }
    };
  });
  $("pClose").onclick = closePanel;
  el.querySelector(".pbody").scrollTop = 0;
}

/* ---------- search and settings ---------- */
const OV = { type: null, q: "", sel: 0 };
function openSearch() { OV.type = "search"; OV.q = ""; OV.sel = 0; renderOverlay(); $("palQ").focus(); }
function openSettings() { OV.type = "settings"; renderOverlay(); }
function closeOv() { OV.type = null; $("ov").innerHTML = ""; }
function matches() {
  const q = OV.q.trim().toLowerCase();
  const hit = (r) => !q || r.symbol.toLowerCase().includes(q) || symName(r.symbol).toLowerCase().includes(q);
  return SYMBOL_ROWS.filter(hit).slice(0, 100);
}
function renderPal() {
  const m = matches();
  OV.sel = clamp(OV.sel, 0, Math.max(0, m.length - 1));
  $("palList").innerHTML = m.length
    ? m.map((r, i) => `<button class="pali" role="option" aria-selected="${i === OV.sel}" data-s="${esc(r.symbol)}"><span class="c">${esc(r.symbol)}</span><span class="n">${esc(symName(r.symbol))}</span><span class="g num">${esc(t("symbolTrades", r.n_trades))}</span></button>`).join("")
    : `<div class="empty" style="padding:14px 16px">${t(P.detail === "full" ? "noResult" : "detailSummaryD")}</div>`;
  $("palList").querySelectorAll(".pali").forEach((b) => { b.onclick = () => { closeOv(); openPanel({ type: "symbol", code: b.dataset.s }); }; });
  const sel = $("palList").querySelector('[aria-selected="true"]');
  if (sel) sel.scrollIntoView({ block: "nearest" });
}
function renderOverlay() {
  const o = $("ov");
  if (!OV.type) { o.innerHTML = ""; return; }
  if (OV.type === "search") {
    o.innerHTML = `<div class="scrim" id="scrim"><div class="pal" role="dialog" aria-label="${t("search")}"><div class="palin">${ICON.search}<input id="palQ" placeholder="${esc(t("searchPh"))}" autocomplete="off" aria-label="${t("search")}" value="${esc(OV.q)}"><span class="kbd" style="display:inline">${t("esc")}</span></div><div class="pallist" id="palList" role="listbox"></div></div></div>`;
    const q = $("palQ");
    q.oninput = () => { OV.q = q.value; OV.sel = 0; renderPal(); };
    q.onkeydown = (e) => {
      const m = matches();
      if (e.key === "ArrowDown") { e.preventDefault(); OV.sel = Math.min(OV.sel + 1, m.length - 1); renderPal(); }
      else if (e.key === "ArrowUp") { e.preventDefault(); OV.sel = Math.max(OV.sel - 1, 0); renderPal(); }
      else if (e.key === "Enter" && m[OV.sel]) { const code = m[OV.sel].symbol; closeOv(); openPanel({ type: "symbol", code }); }
    };
    renderPal();
  } else {
    const op = (id, cur, val, a, b, extra = "") => `<button class="opt" data-${id}="${val}" aria-pressed="${cur === val}"><span>${esc(a)}</span><small>${esc(b)}</small>${extra}</button>`;
    const dots = (u, d) => `<span class="dots"><i style="background:var(${u})"></i><i style="background:var(${d})"></i></span>`;
    o.innerHTML = `<div class="scrim" id="scrim"><div class="dlg" role="dialog" aria-label="${t("setTitle")}"><div class="dhead"><span>${t("setTitle")}</span><button class="iconbtn" id="setClose" aria-label="${t("close")}">${ICON.x}</button></div>
    <div class="dsec"><div><h3>${t("setLang")}</h3><p>${t("setLangD")}</p></div><div class="opts">${op("lang", S.lang, "en", t("langEn"), t("langEnD"))}${op("lang", S.lang, "zh-TW", t("langZh"), t("langZhD"))}</div></div>
    <div class="dsec"><div><h3>${t("setLook")}</h3><p>${t("setLookD")}</p></div><div class="opts">${op("look", S.theme, "light", t("light"), t("lightD"))}${op("look", S.theme, "dark", t("dark"), t("darkD"))}${op("look", S.theme, "system", t("system"), t("systemD"))}</div></div>
    <div class="dsec"><div><h3>${t("setDir")}</h3><p>${t("setDirD")}</p></div><div class="opts">${op("dir", S.updown, "green-up", t("greenUp"), t("greenUpD"), dots("--green", "--red"))}${op("dir", S.updown, "red-up", t("redUp"), t("redUpD"), dots("--red", "--green"))}</div></div></div></div>`;
    $("setClose").onclick = closeOv;
    o.querySelectorAll("[data-lang]").forEach((b) => { b.onclick = () => setLang(b.dataset.lang); });
    o.querySelectorAll("[data-look]").forEach((b) => { b.onclick = () => setTheme(b.dataset.look); });
    o.querySelectorAll("[data-dir]").forEach((b) => { b.onclick = () => setUpdown(b.dataset.dir); });
  }
  $("scrim").onclick = (e) => { if (e.target.id === "scrim") closeOv(); };
}

/* ---------- page assembly ---------- */
function statsHtml() {
  const f = F(), rows = Object.entries(P.stats).filter(([k]) => k !== "rf").map(([k, v]) => [k, Math.abs(v) < 10 ? f.ratio(v, 4) : f.int(v)]);
  return `<details class="statbox"><summary>${t("stats")}</summary>${table(["", ""], rows)}${P.params.length ? `<div class="chiprow pad">${P.params.map((p) => `<span class="chip num">${esc(p)}</span>`).join("")}</div>` : ""}</details>`;
}
function coverageHtml() {
  const cov = (n, on) => `<span>${esc(n)} <b>${on ? "✓" : ""}</b>${on ? "" : `<i>${t("covAbsent")}</i>`}</span>`;
  const note = detailNote();
  return `<div class="covrow num"><b>${t("cov")}</b>${cov(t("covCurve"), true)}${cov(t("covTrades"), !!TABLES.trades)}${cov(t("covBench"), !!BENCH)}${cov(t("covLedger"), !!LEDGER)}<span>${esc(note)}</span></div>`;
}
function renderAll() {
  document.documentElement.lang = L().htmlLang;
  renderBar();
  if (S.view === "analysis") {
    const failed = SUM.checks.fail;
    const money = P.capital ? t("moneyNoteCap", F().money(P.capital)) : t("moneyNote");
    $("main").innerHTML = `<div class="msample">${esc(detailLabel())}</div>${failed ? `<div class="banner">${ICON.warn}${esc(t("checkFail", failed))}</div>` : ""}<div id="heroSlot"></div><div id="behindSlot"></div><div id="healthSlot"></div><div id="monthSlot"></div><div id="tradesSlot"></div>${statsHtml()}${coverageHtml()}<div class="foot"><span>${esc(t("footNote", P.library_version))}</span><span>${esc(money)}</span></div>`;
    renderHero(); renderBehind(); renderHealth(); renderMonthly(); renderTrades();
  } else {
    H.el = null; H.ctx = null;
    renderHoldings();
  }
  renderPanel();
}
let resizeTimer = 0, lastWidth = 0;
new ResizeObserver(() => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    const w = $("main").clientWidth;
    if (w === lastWidth) return;
    lastWidth = w;
    if (S.view === "analysis" && H.el) { sizeCanvas(); renderHealth(); }
  }, 60);
}).observe($("main"));
addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") { e.preventDefault(); openSearch(); return; }
  if (e.key === "Escape") { if (OV.type) closeOv(); else if (S.panel) closePanel(); }
});
matchMedia("(prefers-color-scheme:light)").addEventListener("change", () => { if (S.theme === "system") { applyTheme(); renderAll(); } });
applyTheme();
renderAll();
if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => { tokens(); if (H.el) paint(); });
const startDay = P.day || new URLSearchParams(location.hash.slice(1)).get("day");
if (startDay) openDay(dayOf(startDay));
