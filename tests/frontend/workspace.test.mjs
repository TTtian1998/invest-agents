import test from "node:test";
import assert from "node:assert/strict";
import {
  escapeHTML,
  number,
  percent,
  clamp,
  safeUrl,
  market,
  tag,
} from "../../src/zhiku/app/static/ui.js";
import {
  overview,
  universe,
  evidence,
  validation,
  symbolDetail,
  stockTable,
} from "../../src/zhiku/app/static/views.js";
const data = () => ({
  service: { mode: "demo", demo: true },
  report: {
    state: "ready",
    asof: "2026-10-20",
    freshness: "stale",
    snapshot_id: "a".repeat(64),
  },
  summary: {
    formal_candidates: 0,
    research_observations: 1,
    universe_size: 9,
    quality_passed: 6,
    missing_ratio: 0.093,
  },
  candidates: [],
  research_observations: [
    {
      symbol: "000333",
      name: "示例家电个股",
      formula_score: 81.6,
      qlib_score: null,
      screens: { greenblatt: 100 },
      source: "formula",
      status: "research_only",
    },
  ],
  gates: {
    status: "unavailable",
    reason: "not_published",
    approved_screens: [],
    statuses: {},
  },
  model: {
    available: false,
    inference_available: false,
    reason: "not_published",
  },
  sources: [{ id: "daily", mode: "demo", status: "demo", asof: "2026-10-20" }],
  news: { clusters: [], emergency: { active: null, reasons: [] } },
  macro: { calendar_7d: [] },
  screen_evidence: {
    greenblatt: [{ symbol: "000333", score: 100, breakdown: { roc: 0.53 } }],
  },
  audits: [],
  notes: [],
});
const state = { pool: "candidate", query: "", market: "all", sort: "high" };
test("untrusted strings and attributes are escaped", () => {
  assert.equal(
    escapeHTML(`<img src=x onerror="alert('x')">`),
    "&lt;img src=x onerror=&quot;alert(&#39;x&#39;)&quot;&gt;",
  );
  assert.ok(!tag("test", '"><script>').includes("<script>"));
});
test("missing/invalid metrics never become zero", () => {
  assert.equal(number(null), "—");
  assert.equal(number(NaN), "—");
  assert.equal(percent(null), "未知");
  assert.equal(number(0), "0.0");
  assert.equal(clamp(Infinity), 0);
  assert.equal(clamp(150), 100);
});
test("links reject executable URLs, credentials and demo placeholders", () => {
  for (const url of [
    "javascript:alert(1)",
    "data:text/html,hello",
    "https://user:pass@host.test",
    "https://example.invalid/n1",
    "/local/path",
  ])
    assert.equal(safeUrl(url), null);
  assert.equal(
    safeUrl("https://example.com/report"),
    "https://example.com/report",
  );
});
test("market classification distinguishes CN HK and US", () => {
  assert.equal(market("000333"), "CN");
  assert.equal(market("0700.HK"), "HK");
  assert.equal(market("AAPL"), "US");
});
test("formal empty pool is explicit and observations never promoted", () => {
  const html = universe(data(), state);
  assert.match(html, /当前没有正式候选/);
  assert.match(html, /查看研究观察/);
  assert.doesNotMatch(html, /<td/);
});
test("observation table uses research score and unavailable quant", () => {
  const html = universe(data(), { ...state, pool: "observation" });
  assert.match(html, /公式研究分/);
  assert.match(html, /不可用/);
  assert.match(html, /仅供研究/);
  assert.doesNotMatch(html, /总分/);
});
test("search and market filters produce actionable empty state", () => {
  assert.match(
    universe(data(), { ...state, pool: "observation", query: "nonexistent" }),
    /清除筛选/,
  );
  assert.match(
    universe(data(), { ...state, pool: "observation", market: "US" }),
    /显示 0 \/ 1/,
  );
});
test("report text cannot inject scripts or HTML across views", () => {
  const d = data();
  d.research_observations[0].name = "<script>alert(1)</script>";
  d.notes = ["<img onerror=alert(1)>"];
  d.news.clusters = [
    {
      headline: "<img onerror=alert(1)>",
      tags: [],
      sources: [{ url: "javascript:alert(1)", source: "<script>" }],
    },
  ];
  for (const html of [
    overview(d),
    universe(d, { ...state, pool: "observation" }),
    evidence(d),
    validation(d),
    symbolDetail(d, "000333", "observation"),
  ]) {
    assert.doesNotMatch(html, /<script>/);
    assert.doesNotMatch(html, /<img onerror/);
    assert.doesNotMatch(html, /href="javascript/);
  }
});
test("missing emergency is unknown, not safe", () => {
  assert.match(evidence(data()), /应急状态未知/);
  assert.doesNotMatch(evidence(data()), /当前未进入应急模式/);
});
test("stale emergency state is scoped to snapshot", () => {
  const d = data();
  d.news.emergency.active = false;
  assert.match(evidence(d), /快照未记录应急触发/);
  assert.match(evidence(d), /不能代表当前市场风险/);
});
test("source states are not overridden by demo mode", () => {
  const d = data();
  d.sources = [{ id: "news", mode: "unknown", status: "missing", asof: null }];
  const html = overview(d);
  assert.match(html, /缺少数据/);
});
test("detail retains evidence and never invents AI judgement", () => {
  const html = symbolDetail(data(), "000333", "observation");
  assert.match(html, /没有该标的的 AI 审核证据/);
  assert.match(html, /roc/);
  assert.match(html, /0.53/);
  assert.equal(symbolDetail(data(), "UNKNOWN", "observation"), "");
});
test("CSP-compatible views do not emit inline code/style handlers", () => {
  const d = data();
  for (const html of [
    overview(d),
    universe(d, state),
    evidence(d),
    validation(d),
    symbolDetail(d, "000333", "observation"),
    stockTable(d.research_observations),
  ])
    assert.doesNotMatch(html, /\s(?:style|on\w+)=/);
});

test('overlapping observations do not falsely claim failed approval', () => {
  const d = data();
  d.candidates = [{ ...d.research_observations[0], total: 81.6 }];
  d.summary.formal_candidates = 1;
  for (const html of [overview(d), universe(d, {...state, pool: 'observation'}), symbolDetail(d, '000333', 'observation')]) {
    assert.doesNotMatch(html, /没有通过全部验收|尚未成为正式候选|未验收的研究观察/);
  }
});
