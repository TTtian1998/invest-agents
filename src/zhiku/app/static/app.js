import { escapeHTML as e, icon, hydrateIcons, readable, empty } from "./ui.js";
import {
  overview,
  universe,
  evidence,
  validation,
  symbolDetail,
} from "./views.js";

const pages = {
  overview: ["今日概览", "从研究线索，到可以复核的判断。"],
  universe: ["候选与观察", "区分正式候选资格与公式研究记录，逐条核对。"],
  evidence: ["消息与证据", "追溯来源，核对时效，保留不确定性。"],
  validation: ["研究验证", "查看发布状态、数据边界与每一次判断的来处。"],
};
const state = {
  data: null,
  page: "overview",
  pool: "candidate",
  query: "",
  market: "all",
  sort: "high",
  snapshot: "latest",
  loading: false,
  request: 0,
};
const $ = (s) => document.querySelector(s);
let toastTimer;
let lastDialogTrigger;

function notify(message) {
  const node = $("#toast");
  node.textContent = message;
  node.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    node.hidden = true;
  }, 4000);
}
function getRoute() {
  const value = location.hash.slice(1);
  return Object.hasOwn(pages, value) ? value : "overview";
}
function hydrate() {
  hydrateIcons();
  document.querySelectorAll("[data-width]").forEach((node) => {
    node.style.width = `${Math.max(0, Math.min(100, Number(node.dataset.width) || 0))}%`;
  });
}
function render() {
  state.page = getRoute();
  const [title, description] = pages[state.page];
  $("#page-title").textContent = title;
  $("#page-crumb").textContent = title;
  $("#page-description").textContent = description;
  document.title = `${title} · 智库`;
  document.querySelectorAll("[data-page]").forEach((node) => {
    const active = node.dataset.page === state.page;
    node.classList.toggle("active", active);
    if (active) node.setAttribute("aria-current", "page");
    else node.removeAttribute("aria-current");
  });
  if (!state.data) {
    hydrate();
    return;
  }
  const d = state.data;
  const target = $("#content");
  if (d.report.state !== "ready") {
    const missing = d.report.state === "missing";
    target.innerHTML = `<section class="panel error-state">${empty(missing ? "还没有研究快照" : "这个快照暂时不可用", missing ? "请先在独立知识库中生成日报，再回来刷新。页面不会自动采集数据、训练模型或加载演示内容。" : "数据损坏、来源冲突或边界校验未通过。为避免误读，已停止展示这个快照的研究结果。", '<button class="button primary" data-action="retry">' + icon("refresh") + "重新读取</button>")}<div class="panel-body"><div class="muted-card">${d.report.issues.map(e).join("<br>") || "使用 zk --provider demo run-daily --force --date YYYY-MM-DD 生成独立的演示日报；真实数据应使用单独的 live 知识库。"}</div></div></section>`;
  } else {
    target.innerHTML = (
      { overview, evidence, validation }[state.page] ||
      ((data) => universe(data, state))
    )(d);
  }
  target.setAttribute("aria-busy", "false");
  hydrate();
}
function renderTrust() {
  const d = state.data;
  const banner = $("#trust-banner");
  banner.className = `trust-banner ${d.service.mode === "live" ? "live" : d.service.demo ? "" : "unknown"}`;
  const mode =
    {
      demo: "离线演示",
      live: "真实数据快照",
      disabled: "数据源已停用",
      unknown: "来源未知",
    }[d.service.mode] || "来源未知";
  $("#mode-badge").textContent = mode;
  const stale = d.report.freshness === "stale";
  const time = d.report.asof
    ? `${d.report.asof} 截面${stale ? " · 距今 " + d.report.age_days + " 天" : ""}`
    : "尚无研究截面";
  banner.innerHTML = `${icon(d.service.demo ? "info" : "clock")}<span><strong>${e(mode)}${d.service.demo ? " · 全部为虚构样本" : ""}</strong>${e(d.service.demo ? "仅用于验证工作流，不能用于实盘决策。" : d.report.state === "ready" ? "只读展示已生成报告，不代表实时行情。" : "尚未确认来源与时效，不作正常状态判断。")}${stale ? " 该快照已过期，请勿当作今日信号。" : ""}${d.report.freshness === "future" ? " 该快照日期在未来，不能作为当前信号。" : ""}${d.report.historical ? " 当前查看历史归档。" : ""}</span><span class="banner-right">${e(time)}</span>`;
  $("#heading-eyebrow").textContent = d.report.asof
    ? `研究截面 ${d.report.asof}`
    : "研究工作台";
  $("#nav-count").textContent = d.summary.research_observations;
  $("#footer-snapshot").textContent = d.report.snapshot_id
    ? `快照 ${d.report.snapshot_id.slice(0, 12)} · ${d.report.asof}`
    : "数据来源尚未就绪";
  const options =
    Array.isArray(d.snapshots) && d.snapshots.length
      ? d.snapshots
      : [{ id: "latest", label: "最新快照" }];
  $("#snapshot-select").innerHTML = options
    .map(
      (s) =>
        `<option value="${e(s.id)}" ${s.id === state.snapshot ? "selected" : ""}>${e(s.label || s.asof || s.id)}</option>`,
    )
    .join("");
  const exp = $("#export");
  exp.href = `/api/export?snapshot=${encodeURIComponent(state.snapshot)}`;
  exp.setAttribute("aria-disabled", String(d.report.state !== "ready"));
  if (d.report.state !== "ready") exp.setAttribute("tabindex", "-1");
  else exp.removeAttribute("tabindex");
}
async function load({ feedback = false } = {}) {
  // A previous success message must not survive a new, failed or invalid read.
  clearTimeout(toastTimer);
  $("#toast").hidden = true;
  $("#toast").textContent = "";
  const id = ++state.request;
  state.loading = true;
  $("#export").setAttribute("aria-disabled", "true");
  $("#export").setAttribute("tabindex", "-1");
  $("#refresh").disabled = true;
  $("#snapshot-select").disabled = true;
  $("#content").setAttribute("aria-busy", "true");
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 15000);
  try {
    const response = await fetch(
      `/api/workspace?snapshot=${encodeURIComponent(state.snapshot)}`,
      {
        cache: "no-store",
        headers: { Accept: "application/json" },
        signal: controller.signal,
      },
    );
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    if (id !== state.request) return;
    if (!data || data.schema_version !== 1 || !data.report || !data.summary)
      throw new Error("Invalid schema");
    state.data = data;
    renderTrust();
    render();
    if (feedback && data.report.state === "ready")
      notify("快照已重新读取，未触发数据采集或模型运行。");
  } catch (error) {
    if (id !== state.request) return;
    state.data = null;
    $("#trust-banner").className = "trust-banner unknown";
    $("#trust-banner").innerHTML =
      `${icon("warning")}<span><strong>无法读取工作台</strong>未能确认快照与来源状态，已清除旧数据展示。</span>`;
    $("#content").innerHTML =
      `<section class="panel error-state">${empty("连接暂时不可用", "请检查服务是否仍在运行后重试。为避免把上一次数据当成最新结果，页面不会继续展示旧快照。", '<button class="button primary" data-action="retry">' + icon("refresh") + "重新连接</button>")}</section>`;
    $("#export").setAttribute("aria-disabled", "true");
    $("#export").setAttribute("tabindex", "-1");
    $("#mode-badge").textContent = "连接中断";
    $("#heading-eyebrow").textContent = "研究工作台";
    $("#footer-snapshot").textContent = "未能确认当前数据";
    $("#nav-count").textContent = "—";
    $("#content").setAttribute("aria-busy", "false");
  } finally {
    clearTimeout(timer);
    if (id === state.request) {
      state.loading = false;
      $("#refresh").disabled = false;
      $("#snapshot-select").disabled = false;
    }
  }
}
function closeDialog() {
  const dialog = $("#symbol-dialog");
  if (dialog.open) dialog.close();
}
function closeMenu() {
  const menu = $("#sidebar");
  menu.classList.remove("open");
  $("#menu-toggle").setAttribute("aria-expanded", "false");
  $("#menu-toggle").setAttribute("aria-label", "展开导航");
}
function setPool(pool) {
  state.pool = pool === "observation" ? "observation" : "candidate";
  state.query = "";
  state.market = "all";
  if (state.page !== "universe") {
    location.hash = "universe";
  } else render();
}

document.addEventListener("click", (event) => {
  const node = event.target instanceof Element ? event.target : null;
  if (!node) return;
  const pool = node.closest("[data-pool]");
  if (pool) {
    event.preventDefault();
    setPool(pool.dataset.pool);
    return;
  }
  const row = node.closest("[data-symbol]");
  if (row && state.data) {
    lastDialogTrigger = row;
    $("#symbol-content").innerHTML = symbolDetail(
      state.data,
      row.dataset.symbol,
      row.dataset.kind,
    );
    hydrate();
    if ($("#symbol-content").innerHTML) {
      $("#symbol-dialog").showModal();
      $('#symbol-dialog [data-action="close-dialog"]').focus();
    }
    return;
  }
  const action = node.closest("[data-action]")?.dataset.action;
  if (action === "retry") {
    load({ feedback: true });
    return;
  }
  if (action === "close-dialog") {
    closeDialog();
    return;
  }
  if (action === "clear-filters") {
    state.query = "";
    state.market = "all";
    state.sort = "high";
    render();
    $("#search")?.focus();
    return;
  }
  if (node.closest("[data-page]")) {
    closeDialog();
    closeMenu();
  }
  if (!node.closest("#sidebar") && !node.closest("#menu-toggle")) closeMenu();
});
document.addEventListener("input", (event) => {
  if (event.target.id === "search") {
    const node = event.target;
    const cursor = node.selectionStart;
    state.query = node.value;
    render();
    const next = $("#search");
    next.focus();
    try {
      next.setSelectionRange(cursor, cursor);
    } catch {
      /* search input selection varies by browser */
    }
  }
});
document.addEventListener("change", (event) => {
  if (event.target.id === "market-filter") {
    state.market = event.target.value;
    render();
  }
  if (event.target.id === "sort") {
    state.sort = event.target.value;
    render();
  }
});
$("#snapshot-select").addEventListener("change", (event) => {
  state.snapshot = event.target.value;
  closeDialog();
  load();
});
$("#refresh").addEventListener("click", () => load({ feedback: true }));
$("#menu-toggle").addEventListener("click", () => {
  const opened = $("#sidebar").classList.toggle("open");
  $("#menu-toggle").setAttribute("aria-expanded", String(opened));
  $("#menu-toggle").setAttribute(
    "aria-label",
    opened ? "关闭导航" : "展开导航",
  );
});
$("#symbol-dialog").addEventListener("click", (event) => {
  if (event.target === $("#symbol-dialog")) {
    const rect = event.target.getBoundingClientRect();
    if (
      event.clientX < rect.left ||
      event.clientX > rect.right ||
      event.clientY < rect.top ||
      event.clientY > rect.bottom
    )
      closeDialog();
  }
});
$("#symbol-dialog").addEventListener("close", () => {
  if (lastDialogTrigger?.isConnected) lastDialogTrigger.focus();
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") closeMenu();
  const tab = event.target instanceof Element ? event.target.closest('[role="tab"]') : null;
  if (tab && ["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) {
    event.preventDefault();
    const pool = event.key === "Home" ? "candidate" : event.key === "End" ? "observation" : state.pool === "candidate" ? "observation" : "candidate";
    setPool(pool);
    document.querySelector(`[role="tab"][data-pool="${pool}"]`)?.focus();
  }
});
window.addEventListener("hashchange", () => {
  closeDialog();
  closeMenu();
  render();
  $("#main").focus({ preventScroll: true });
  window.scrollTo({ top: 0, behavior: "instant" });
});
render();
load();
