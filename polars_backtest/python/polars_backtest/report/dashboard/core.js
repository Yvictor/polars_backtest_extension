/* Pure calculations for live date ranges. No DOM here, so node can run this file directly.
   Formulas repeat report/metrics.py, and tests/test_report_dashboard.py holds the two copies
   together. Anything that does not move with the page belongs in Python, not here. */
"use strict";
const BT = (() => {
  const NPERIODS = 252;

  const utc = (iso) => Date.UTC(+iso.slice(0, 4), +iso.slice(5, 7) - 1, +iso.slice(8, 10));
  const dayDiff = (a, b) => Math.round((utc(b) - utc(a)) / 864e5);
  const annualise = (growth, years) => (years > 0 && growth > 0 ? Math.pow(growth, 1 / years) - 1 : null);
  const mean = (x) => (x.length ? x.reduce((a, b) => a + b, 0) / x.length : null);

  function std(x) {
    if (x.length < 2) return null;
    const m = mean(x);
    return Math.sqrt(x.reduce((a, v) => a + (v - m) * (v - m), 0) / (x.length - 1));
  }

  function median(x) {
    const s = [...x].sort((p, q) => p - q), m = s.length >> 1;
    return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
  }

  /* First index whose date is on or after iso, capped at the last day. */
  function dayIndex(dates, iso) {
    let lo = 0, hi = dates.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (dates[mid] < iso) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }

  /* Everything a cursor can show, from one range: one pass per range, O(1) per scrub.
     peak: index of the last high. mdd and low: the deepest fall so far.
     uwA and uwB: the longest stretch below a prior high so far. */
  function track(eq, dates, a, b) {
    const n = b - a + 1;
    const T = {
      a, b,
      peak: new Int32Array(n), mdd: new Float64Array(n), low: new Int32Array(n),
      uwA: new Int32Array(n), uwB: new Int32Array(n),
    };
    let pk = a, m = 0, lo = a, ua = a, ub = a, best = 0;
    for (let i = a; i <= b; i++) {
      const under = eq[i] < eq[pk];
      if (under || i - pk > 1) {
        const days = dayDiff(dates[pk], dates[i]);
        if (days > best) { best = days; ua = pk; ub = i; }
      }
      if (!under) pk = i;
      const d = eq[i] / eq[pk] - 1;
      if (d < m) { m = d; lo = i; }
      const k = i - a;
      T.peak[k] = pk; T.mdd[k] = m; T.low[k] = lo; T.uwA[k] = ua; T.uwB[k] = ub;
    }
    return T;
  }

  /* Whole-range figures, the same definitions as metrics.performance. */
  function rangeKpi(eq, dates, a, b, rf) {
    const rets = [0];
    for (let i = a + 1; i <= b; i++) rets.push(eq[i] / eq[i - 1] - 1);
    const rfp = Math.pow(1 + rf, 1 / NPERIODS) - 1;
    const excess = rets.map((r) => r - rfp);
    const downside = excess.map((r) => Math.min(r, 0));
    const total = eq[b] / eq[a] - 1;
    const cagr = annualise(1 + total, dayDiff(dates[a], dates[b]) / 365.25);
    const mdd = track(eq, dates, a, b).mdd[b - a];
    const root = Math.sqrt(NPERIODS), vol = std(rets), sdEx = std(excess), sdDn = std(downside);
    return {
      total_return: total,
      cagr,
      volatility: vol == null ? null : vol * root,
      sharpe: sdEx ? (mean(excess) / sdEx) * root : null,
      sortino: sdDn ? (mean(excess) / sdDn) * root : null,
      calmar: cagr != null && mdd < 0 ? cagr / -mdd : null,
      max_drawdown: mdd,
    };
  }

  /* Outcome of every w-day hold, one per start date. */
  function rollHold(eq, w) {
    const v = new Array(eq.length).fill(null);
    for (let i = 0; i + w < eq.length; i++) v[i] = eq[i + w] / eq[i] - 1;
    return v;
  }

  /* Annualised return as the best months are set flat, one at a time. */
  function flatBest(monthRets, k, growth, years) {
    const best = [...monthRets].sort((a, b) => b - a).slice(0, k);
    const steps = [annualise(growth, years)];
    let g = growth;
    for (const r of best) { g /= 1 + r; steps.push(annualise(g, years)); }
    return steps;
  }

  /* Rolling Sharpe or volatility over the w daily returns ending on each day.
     Sharpe is on excess return, the same definition as rangeKpi. */
  function rolling(rets, w, kind, rf) {
    const rfp = Math.pow(1 + rf, 1 / NPERIODS) - 1;
    const out = new Array(rets.length).fill(null);
    for (let i = w - 1; i < rets.length; i++) {
      const x = rets.slice(i - w + 1, i + 1), sd = std(x);
      if (!sd) continue;
      out[i] = (kind === "sharpe" ? (mean(x) - rfp) / sd : sd) * Math.sqrt(NPERIODS);
    }
    return out;
  }

  /* Which points to draw in [a, b] at a given width. Thinning is for drawing only: every
     bucket keeps its first, last, high and low in each series, so no extreme disappears
     from a line. Calculations always read the complete arrays. */
  function drawIndices(series, a, b, pixels) {
    const n = b - a + 1, size = Math.max(1, Math.floor(n / Math.max(1, pixels)));
    if (size <= 2) return Array.from({ length: n }, (_, i) => a + i);
    const keep = new Set([a, b]);
    for (let lo = a; lo <= b; lo += size) {
      const hi = Math.min(b, lo + size - 1);
      keep.add(lo); keep.add(hi);
      for (const s of series) {
        let low = null, high = null;
        for (let i = lo; i <= hi; i++) {
          if (s[i] == null) continue;
          if (low == null || s[i] < s[low]) low = i;
          if (high == null || s[i] > s[high]) high = i;
        }
        if (low != null) { keep.add(low); keep.add(high); }
      }
    }
    return [...keep].sort((p, q) => p - q);
  }

  /* Trade figures for a date range, summed from the per-exit-day aggregates of every trade.
     Rows in the page are a trimmed sample, so they are never used for these. */
  function windowTrades(days, from, to, stops) {
    const blank = () => ({ n: 0, wins: 0, sum_ret: 0, contrib: 0 });
    const sides = { long: blank(), short: blank() };
    const stopSums = stops.map(() => ({ hit: 0, win: 0, sum: 0 }));
    let nMae = 0;
    for (const r of days) {
      if (r.date < from || r.date > to) continue;
      const s = sides[r.side];
      s.n += r.n; s.wins += r.wins; s.sum_ret += r.sum_ret; s.contrib += r.contrib;
      nMae += r.n_mae ?? 0;
      stopSums.forEach((x, i) => {
        x.hit += r[`hit_${i}`] ?? 0; x.win += r[`win_${i}`] ?? 0; x.sum += r[`sum_${i}`] ?? 0;
      });
    }
    const n = sides.long.n + sides.short.n, sumRet = sides.long.sum_ret + sides.short.sum_ret;
    const side = (s) => (s.n ? { n: s.n, win_rate: s.wins / s.n, avg_ret: s.sum_ret / s.n, contrib: s.contrib } : null);
    return {
      n,
      wins: sides.long.wins + sides.short.wins,
      mean: n ? sumRet / n : null,
      long: side(sides.long),
      short: side(sides.short),
      stops: nMae ? stopSums.map((x) => ({ hit: x.hit, win: x.win / nMae, avg: x.sum / nMae })) : [],
    };
  }

  /* The whole-period histogram's bins, counted over a date range. */
  function windowHist(hist, days, from, to) {
    const counts = new Array(hist.counts.length).fill(0);
    let n = 0;
    for (const r of days) {
      if (r.date < from || r.date > to) continue;
      counts[r.bin] += r.n;
      n += r.n;
    }
    return { low: hist.low, width: hist.width, counts, n };
  }

  /* The periods overlapping [first, last], clipped to it, so a total never counts days
     outside the range. Periods carry their own day indexes a and b. */
  function periodsIn(periods, first, last) {
    return periods.filter((p) => p.b >= first && p.a <= last)
      .map((p) => ({ ...p, a: Math.max(p.a, first), b: Math.min(p.b, last) }));
  }

  /* What a page holds against its bundle, read from the counts the payload carries. */
  function embedStatus(embed) {
    const parts = ["trades", "fills", "blocks"].filter((k) => embed[k]).map((name) => ({
      name, embedded: embed[name].rows_embedded, total: embed[name].rows_total,
    }));
    return { complete: parts.every((p) => p.embedded === p.total), parts };
  }

  return {
    NPERIODS, annualise, dayDiff, dayIndex, drawIndices, embedStatus, flatBest, mean, median,
    periodsIn, rangeKpi, rollHold, rolling, std, track, windowHist, windowTrades,
  };
})();
if (typeof module !== "undefined") module.exports = BT;
