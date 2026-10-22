/** Safe view primitives. Untrusted snapshot text is escaped, never executable markup. */
export const escapeHTML = (value) =>
  String(value ?? "").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
export const number = (value, digits = 1) =>
  typeof value === "number" && Number.isFinite(value)
    ? value.toFixed(digits)
    : "—";
export const percent = (value) =>
  typeof value === "number" && Number.isFinite(value)
    ? `${(value * 100).toFixed(1)}%`
    : "未知";
export const list = (value) => (Array.isArray(value) ? value : []);
export const object = (value) =>
  value && typeof value === "object" && !Array.isArray(value) ? value : {};
export const clamp = (value) =>
  typeof value === "number" && Number.isFinite(value)
    ? Math.max(0, Math.min(100, value))
    : 0;
export const tag = (text, kind = "") =>
  `<span class="tag ${["blue", "amber", "red", "green"].includes(kind) ? kind : ""}">${escapeHTML(text)}</span>`;
export const icons = {
  overview:
    '<rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/>',
  universe:
    '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M3 9h18M8 9v11M16 9v11"/>',
  evidence:
    '<path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9zM14 3v6h6M8 13h8M8 17h5"/>',
  shield: '<path d="m12 3 8 3v6c0 5-8 9-8 9s-8-4-8-9V6zM8 12l3 3 5-6"/>',
  lock: '<rect x="5" y="10" width="14" height="11" rx="2"/><path d="M8 10V6a4 4 0 0 1 8 0v4M12 14v3"/>',
  calendar:
    '<rect x="3" y="5" width="18" height="16" rx="2"/><path d="M7 3v4M17 3v4M3 10h18M7 14h3M14 14h3M7 17h3"/>',
  refresh:
    '<path d="M20 8a8 8 0 0 0-14-3L3 8m0-5v5h5M4 16a8 8 0 0 0 14 3l3-3m0 5v-5h-5"/>',
  download: '<path d="M12 3v12m-4-4 4 4 4-4M4 15v5h16v-5"/>',
  search: '<circle cx="10.5" cy="10.5" r="6.5"/><path d="m16 16 5 5"/>',
  info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v6m0-10v.5"/>',
  warning: '<path d="m12 3 10 18H2zM12 9v5m0 3v.5"/>',
  chart: '<path d="M4 3v17h17M8 15v-4m5 4V6m5 9V9"/>',
  arrow: '<path d="M5 12h14m-5-5 5 5-5 5"/>',
  chevron: '<path d="m9 5 7 7-7 7"/>',
  close: '<path d="m6 6 12 12M6 18 18 6"/>',
  menu: '<path d="M4 6h16M4 12h16M4 18h16"/>',
  layers: '<path d="m12 3 10 5-10 5L2 8zm-10 9 10 5 10-5M2 16l10 5 10-5"/>',
  clock: '<circle cx="12" cy="12" r="9"/><path d="M12 6v6l4 2"/>',
  check: '<path d="m5 12 4 4L19 6"/>',
};
export const icon = (name) =>
  `<svg class="icon" viewBox="0 0 24 24" aria-hidden="true">${icons[name] || icons.info}</svg>`;
export function hydrateIcons(root = document) {
  root.querySelectorAll("[data-icon]").forEach((el) => {
    el.innerHTML = icon(el.dataset.icon);
  });
}
export const screenNames = {
  greenblatt: "格林布拉特",
  piotroski: "财务质量 F-Score",
  dividend: "红利策略",
  multifactor: "多因子",
  graham: "格雷厄姆",
  lynch_peg: "彼得·林奇 PEG",
  canslim: "CAN SLIM",
  livermore: "趋势动量",
  fisher: "费雪成长",
};
export const reasonNames = {
  demo: "演示样本",
  snapshot: "已存快照",
  partial: "部分来源缺失",
  disabled: "已停用",
  validated: "已验收",
  passed: "已通过",
  not_published: "尚未发布",
  stale: "已过期",
  expired: "已过期",
  provider_mismatch: "来源模式不匹配",
  source_mode_mismatch: "来源模式不匹配",
  config_mismatch: "配置指纹不匹配",
  future: "未来数据，不可用",
  invalid: "验收文件无效",
  unavailable: "不可用",
  approved: "已通过验收",
  ready: "可用",
  unverified: "待验证",
  unknown: "未知",
  missing: "缺少数据",
  current: "当日快照",
  insufficient: "样本不足",
  failed: "未通过",
  no_current_features_or_scores: "缺少当前特征",
  checksum_mismatch: "完整性校验失败",
  source_mismatch: "数据来源不匹配",
  code_mismatch: "策略版本不匹配",
};
export const readable = (value, fallback = "未知") =>
  reasonNames[value] || (value ? String(value) : fallback);
export function market(symbol) {
  if (String(symbol).endsWith(".HK")) return "HK";
  if (/^\d{6}$/.test(String(symbol))) return "CN";
  return "US";
}
export const marketNames = { CN: "A 股", HK: "港股", US: "美股" };
export function safeUrl(raw) {
  try {
    const u = new URL(raw);
    return ["https:", "http:"].includes(u.protocol) &&
      !u.username &&
      !u.password &&
      !u.hostname.endsWith(".invalid")
      ? u.href
      : null;
  } catch {
    return null;
  }
}
export function empty(title, description, action = "") {
  return `<div class="empty-state"><div class="empty-icon">${icon("layers")}</div><h3>${escapeHTML(title)}</h3><p>${escapeHTML(description)}</p>${action}</div>`;
}
export function panel(title, description, body, extra = "", footer = "") {
  return `<section class="panel"><div class="panel-head"><div><h2>${escapeHTML(title)}</h2>${description ? `<p>${escapeHTML(description)}</p>` : ""}</div>${extra}</div><div class="panel-body">${body}</div>${footer ? `<div class="panel-foot">${footer}</div>` : ""}</section>`;
}
