import {
  escapeHTML as e,
  number as n,
  percent,
  list,
  object,
  clamp,
  tag,
  icon,
  screenNames,
  readable,
  market,
  marketNames,
  safeUrl,
  empty,
  panel,
} from "./ui.js";
const gateAvailable = (d) =>
  d.gates.status === "approved" &&
  d.gates.reason === "validated" &&
  list(d.gates.approved_screens).length > 0;
const tableAction = (item, kind) =>
  `<button class="row-detail" data-symbol="${e(item.symbol)}" data-kind="${kind}" aria-label="查看 ${e(item.symbol)} 研究证据">查看证据 ${icon("chevron")}</button>`;
export function stockTable(rows, kind = "observation", compact = false) {
  if (!rows.length)
    return empty(
      "没有匹配的标的",
      "尝试其他代码、名称或市场，或清除筛选条件。",
      '<button class="button secondary" data-action="clear-filters">清除筛选</button>',
    );
  return `<div class="table-wrap"><table><thead><tr><th scope="col">标的 / 代码</th><th scope="col">市场</th><th scope="col">${kind === "candidate" ? "公式分" : "公式研究分"}</th>${!compact ? '<th scope="col">量化辅助</th>' : ""}<th scope="col">状态</th><th scope="col"><span class="text-muted">证据链</span></th></tr></thead><tbody>${rows
    .map(
      (r) =>
        `<tr><td><div class="stock-name"><span class="stock-avatar" aria-hidden="true">${e(
          String(r.name || r.symbol)
            .replace(/^示例/, "")
            .slice(0, 1),
        )}</span><span class="stock-title">${e(r.name || r.symbol)}<small class="stock-code">${e(r.symbol)}</small></span></div></td><td class="cell-secondary">${marketNames[market(r.symbol)]}</td><td><span class="number">${n(r.formula_score)}</span><span class="mini-bar" aria-hidden="true"><span data-width="${clamp(r.formula_score)}"></span></span></td>${!compact ? `<td class="cell-secondary">${r.qlib_score == null ? "不可用" : n(r.qlib_score)}</td>` : ""}<td>${tag(kind === "candidate" ? "正式候选" : "仅供研究", kind === "candidate" ? "blue" : "")}</td><td>${tableAction(r, kind)}</td></tr>`,
    )
    .join("")}</tbody></table></div>`;
}
function chart(d) {
  const rows = list(d.research_observations)
    .slice()
    .sort(
      (a, b) => (b.formula_score ?? -Infinity) - (a.formula_score ?? -Infinity),
    )
    .slice(0, 5);
  return rows.length
    ? `<div class="research-chart">${rows.map((r) => `<button class="chart-row chart-button" data-symbol="${e(r.symbol)}" data-kind="observation" aria-label="查看 ${e(r.symbol)} 研究证据"><span class="chart-label">${e(String(r.name || r.symbol).replace(/^示例/, ""))}<small>${e(r.symbol)}</small></span><span class="chart-track" aria-hidden="true"><span class="chart-fill" data-width="${clamp(r.formula_score)}"></span></span><span class="chart-score">${n(r.formula_score)}</span></button>`).join("")}</div><div class="chart-scale" aria-hidden="true"><span>0</span><span>25</span><span>50</span><span>75</span><span>100</span></div><div class="chart-legend"><span class="legend-square"></span>公式研究分 <span>· 多公式平均，不代表收益预测</span></div>`
    : empty(
        "暂无研究分布",
        "生成日报后，显示真实公式研究分，不以模拟收益代替。",
      );
}
function workflow(d) {
  return `<div class="review-status"><div class="review-step"><strong>数据截面</strong>${tag(d.report.asof || "尚无报告", "blue")}<p>${d.service.demo ? "离线虚构样本，与实盘完全隔离" : "只读取当前知识库中的已生成报告"}</p></div><div class="review-step"><strong>公式验收</strong>${tag(gateAvailable(d) ? `${d.gates.approved_screens.length} 个公式已验收` : readable(d.gates.reason), "amber")}<p>仅使用显式发布的验收产物</p></div><div class="review-step"><strong>量化辅助</strong>${tag(d.model.inference_available ? "推理可用" : readable(d.model.reason, "未启用"))}<p>不自动训练，不回退到其他模型</p></div><div class="review-step"><strong>正式候选</strong>${tag(`${d.summary.formal_candidates} 个`, d.summary.formal_candidates ? "blue" : "")}<p>${d.summary.formal_candidates ? "仍需独立复核，不构成交易建议" : "缺少有效验收时，保持关闭"}</p></div></div><div class="side-note">研究观察独立保留。分数较高，不等于已经通过公式验收。</div>`;
}
function events(d, limit = 3) {
  const rows = list(d.macro.calendar_7d).slice(0, limit);
  return rows.length
    ? `<div class="event-list">${rows
        .map((r) => {
          const date = String(r.scheduled_shanghai || r.scheduled_at || "");
          return `<div class="event-item"><div class="event-date"><strong>${e(date.slice(8, 10) || "—")}</strong><small>${e(date.slice(5, 7) || "—")} 月</small></div><div class="event-info">${e(r.name)}<span class="event-region">${e(r.region)}</span><p>${e(date.slice(11, 16))} 上海时间 · ${r.actual == null ? "结果待补充" : "已发布数据"}</p></div></div>`;
        })
        .join("")}</div>`
    : '<p class="empty-small">没有可用的事件日历。缺失数据不代表没有事件风险。</p>';
}
function sources(d) {
  return `<div class="sources-list">${list(d.sources)
    .map(
      (r) =>
        `<div class="source-row"><div>${e({ daily: "研究日报", news: "消息来源", macro: "宏观日历" }[r.id] || r.id)}<small>${e(r.asof || "日期未知")}</small></div>${tag(readable(r.status), r.status === "demo" ? "amber" : "")}</div>`,
    )
    .join("")}</div>`;
}
export function overview(d) {
  const s = d.summary;
  return `<div class="summary-strip"><div class="summary-item"><div class="summary-label">${icon("shield")}正式候选</div><div class="summary-value">${s.formal_candidates}<span>个</span></div><div class="summary-note caution">${gateAvailable(d) ? "仅显示已验收公式内候选" : "验收未就绪，自动关闭"}</div></div><div class="summary-item"><div class="summary-label">${icon("search")}研究观察</div><div class="summary-value">${s.research_observations}<span>个</span></div><div class="summary-note">独立观察池，不等于可交易</div></div><div class="summary-item"><div class="summary-label">${icon("universe")}研究覆盖</div><div class="summary-value">${s.universe_size}<span>个标的</span></div><div class="summary-note">${s.quality_passed} 个通过基础质量筛选</div></div><div class="summary-item"><div class="summary-label">${icon("layers")}数据缺失</div><div class="summary-value">${s.missing_ratio == null ? "—" : percent(s.missing_ratio)}</div><div class="summary-note">缺失不是零值，不补造证据</div></div></div><div class="main-grid"><div class="main-column">${panel("研究线索分布", "先看相对分数，再核对每一条证据。", chart(d), tag("公式研究视图"), '分数来自当前快照 <a class="button link" href="#universe" data-pool="observation">查看全部观察 ' + icon("arrow") + "</a>")}<section class="panel"><div class="panel-head table-heading"><div><h2>研究观察池</h2><p>各公式的研究记录，不直接授予候选资格。</p></div><a class="button link" href="#universe" data-pool="observation">全部 ${s.research_observations} 个 ${icon("arrow")}</a></div>${stockTable(list(d.research_observations).slice(0, 5), "observation", true)}<div class="panel-foot"><span>公式分仅用于比较；量化缺失不按 0 处理</span><span>${e(d.report.asof || "")}</span></div></section>${panel("来源状态", "每一条判断，都应有可以追溯的来源。", sources(d), "", `<span>快照并非实时行情</span><a class="button link" href="#validation">查看完整溯源 ${icon("arrow")}</a>`)}</div><div class="side-column">${panel("研究流程", "当前快照的可用性", workflow(d))}${panel("事件日历", d.service.demo ? "虚构日历样本 · 非实时事件" : "快照内的日历，不自动刷新", events(d), icon("calendar"), '<a class="button link" href="#evidence">查看消息与事件 ' + icon("arrow") + "</a>")}</div></div>`;
}
export function universe(d, state) {
  const isCandidate = state.pool === "candidate";
  const all = isCandidate ? list(d.candidates) : list(d.research_observations);
  const q = state.query.trim().toLowerCase();
  const rows = all
    .filter(
      (r) =>
        (!q || `${r.symbol} ${r.name}`.toLowerCase().includes(q)) &&
        (state.market === "all" || market(r.symbol) === state.market),
    )
    .sort((a, b) =>
      state.sort === "symbol"
        ? String(a.symbol).localeCompare(String(b.symbol))
        : (state.sort === "low" ? 1 : -1) *
          ((a.formula_score ?? -Infinity) - (b.formula_score ?? -Infinity)),
    );
  return `<div class="section-heading"><div class="segmented" role="tablist" aria-label="标的池"><button class="segment" role="tab" aria-selected="${isCandidate}" data-pool="candidate">正式候选 <small>${d.summary.formal_candidates}</small></button><button class="segment" role="tab" aria-selected="${!isCandidate}" data-pool="observation">研究观察 <small>${d.summary.research_observations}</small></button></div><span class="text-muted small-text">截面 ${e(d.report.asof || "未知")} · ${d.service.demo ? "虚构演示样本" : "只读报告"}</span></div>${isCandidate ? `<div class="inline-notice caution">${icon("shield")}<span>正式候选仅来自已发布验收的公式。${d.service.demo ? "当前仍为虚构演示，不能用于真实投资决策。" : "正式候选也不等于交易建议。"}</span></div>` : `<div class="inline-notice">${icon("info")}<span>研究观察视图本身不代表验收资格，可能与正式候选重叠。公式研究分只表达当前样本内的相对排序。</span></div>`}<div class="toolbar"><label class="search-box">${icon("search")}<input id="search" type="search" value="${e(state.query)}" placeholder="搜索名称或代码" aria-label="搜索名称或代码" autocomplete="off"></label><select id="market-filter" aria-label="筛选市场"><option value="all">全部市场</option>${Object.entries(
    marketNames,
  )
    .map(
      ([key, value]) =>
        `<option value="${key}" ${key === state.market ? "selected" : ""}>${value}</option>`,
    )
    .join(
      "",
    )}</select><select id="sort" aria-label="排序方式"><option value="high" ${state.sort === "high" ? "selected" : ""}>公式分：从高到低</option><option value="low" ${state.sort === "low" ? "selected" : ""}>公式分：从低到高</option><option value="symbol" ${state.sort === "symbol" ? "selected" : ""}>代码顺序</option></select></div><section class="panel">${isCandidate && !all.length ? empty("当前没有正式候选", gateAvailable(d) ? "已验收公式在当前截面没有输出正式候选。研究记录仍然保留，供独立复核。" : `公式验收${readable(d.gates.reason, "尚未就绪")}。这是一道保护边界，不是页面故障；研究观察仍然可以继续核对。`, '<button class="button primary" data-pool="observation">查看研究观察 ' + icon("arrow") + '</button><br><a href="#validation" class="button link">了解验收状态</a>') : stockTable(rows, isCandidate ? "candidate" : "observation")}<div class="panel-foot"><span>显示 ${rows.length} / ${all.length} 个标的</span><span>点击「查看证据」展开详情</span></div></section>`;
}
export function evidence(d) {
  const em = d.news.emergency || {};
  const risk =
    em.active === null || em.active === undefined
      ? "应急状态未知"
      : em.active
        ? "快照记录了应急信号"
        : "快照未记录应急触发";
  const cards = list(d.news.clusters);
  return `<div class="inline-notice ${em.active ? "caution" : ""}">${icon("warning")}<span><strong>${e(risk)}</strong>。${d.report.freshness !== "current" || d.service.demo ? "该结果来自历史或演示快照，不能代表当前市场风险。" : "这不是零风险声明；缺少来源时需要重新核验。"} ${list(em.reasons).map(e).join("；")}</span></div><div class="section-heading"><div><h2>消息证据</h2><p class="text-muted small-text description">保留来源级别、时间与反应，不将消息直接升级为结论。</p></div>${tag(`${cards.length} 条线索`)}</div>${
    cards.length
      ? `<div class="news-grid">${cards
          .map(
            (r) =>
              `<article class="panel news-card"><div class="news-tags">${tag(r.source_grade_label || "来源等级未知", r.source_grade === 1 ? "blue" : "")}${tag(r.stale ? "旧闻 / 需复核" : r.freshness || "时效未知", r.stale ? "amber" : "")}${d.service.demo ? tag("虚构样本") : ""}</div><h3>${e(r.headline)}</h3><div class="news-tags">${list(
                r.tags,
              )
                .map((t) => tag(t))
                .join(
                  "",
                )}${r.symbol ? tag(r.symbol) : ""}</div><p class="news-meta">首次出现 ${e(r.first_seen || "未知")}<br>市场反应：${e(r.reaction || "未知")}</p><div class="news-sources">${
                list(r.sources)
                  .map((s) => {
                    const url = safeUrl(s.url);
                    return url
                      ? `<a href="${e(url)}" target="_blank" rel="noopener noreferrer">${e(s.source || "查看来源")} ↗</a>`
                      : `<span class="text-muted">${e(s.source || "来源待核验")}${d.service.demo ? " · 演示链接" : " · 无有效链接"}</span>`;
                  })
                  .join("") || '<span class="text-muted">暂无可追溯链接</span>'
              }</div></article>`,
          )
          .join("")}</div>`
      : panel(
          "暂无消息证据",
          "",
          empty(
            "消息来源未就绪",
            "没有记录不代表没有风险。请先生成同一来源模式的消息报告。",
          ),
        )
  }<section class="panel calendar-panel"><div class="panel-head table-heading"><div><h2>事件与数据日历</h2><p>${d.service.demo ? "虚构演示样本；不是实际日程或交易提示。" : "来自所选快照。历史快照不代表今天的日程。"}</p></div>${tag("上海时间")}</div>${
    list(d.macro.calendar_7d).length
      ? `<div class="table-wrap"><table><thead><tr><th>事件</th><th>时间</th><th>区域</th><th>前值</th><th>预期</th><th>实际</th></tr></thead><tbody>${list(
          d.macro.calendar_7d,
        )
          .map(
            (r) =>
              `<tr><td>${e(r.name)}</td><td>${e(r.scheduled_shanghai || r.scheduled_at || "未知")}</td><td>${e(r.region)}</td><td>${n(r.previous, 2)} ${e(r.unit)}</td><td>${n(r.consensus, 2)}</td><td class="cell-secondary">${r.actual == null ? "尚无数据" : n(r.actual, 2)}</td></tr>`,
          )
          .join("")}</tbody></table></div>`
      : empty("暂无事件日历", "缺少日历数据时，不推断当前没有重要事件。")
  }</section>`;
}
export function validation(d) {
  const g = d.gates;
  const m = d.model;
  const screens = Object.keys(screenNames);
  return `<div class="inline-notice">${icon("shield")}<span>日常运行只消费已经发布的研究产物。这个工作台不会训练模型、重跑回测或修改验收状态。</span></div><div class="validation-grid">${panel("公式验收", "研究结果必须经过显式发布，才进入日常候选。", `<dl class="definition-list"><dt>验收状态</dt><dd>${tag(gateAvailable(d) ? "存在已验收公式" : readable(g.reason), gateAvailable(d) ? "blue" : "amber")}</dd><dt>验收截面</dt><dd>${e(g.evaluated_asof || "尚未发布")}</dd><dt>产物版本</dt><dd>${e(g.version || "无可用版本")}</dd></dl><div class="subsection-title">各公式状态</div>${screens.map((key) => `<div class="status-row"><div>${e(screenNames[key])}<small>${e(key)}</small></div>${tag(list(g.approved_screens).includes(key) ? "已验收" : readable(object(g.statuses)[key], readable(g.reason, "待验收")), list(g.approved_screens).includes(key) ? "blue" : "")}</div>`).join("")}`)}${panel("量化辅助模型", "缺少发布产物时保持不可用，不以其他模型替代。", `<dl class="definition-list"><dt>发布状态</dt><dd>${tag(m.available ? "已发布" : readable(m.reason), m.available ? "blue" : "amber")}</dd><dt>推理状态</dt><dd>${m.inference_available ? "当前快照有推理结果" : "未提供有效推理结果"}</dd><dt>模型类型</dt><dd>${e(m.implementation || "未发布")}</dd><dt>训练截面</dt><dd>${e(m.trained_on || "未知")}</dd><dt>Rank IC</dt><dd>${n(m.rank_ic, 3)} <span class="text-muted">${m.rank_ic == null ? "（没有可用证据）" : ""}</span></dd></dl><div class="muted-card spaced">发布完整性检查不等于样本外有效性。量化分只作为过闸公式内的排序辅助，不单独生成正式候选。</div><h3 class="subsection-title">数据与快照溯源</h3>${sources(d)}<details class="disclosure"><summary>查看快照标识与验收元数据</summary><pre class="technical">${e(JSON.stringify({ snapshot_id: d.report.snapshot_id, source_mode: d.service.mode, asof: d.report.asof, freshness: d.report.freshness, historical: d.report.historical, gates: g, model: m }, null, 2))}</pre></details>`)}<div class="full-width">${panel("运行记录", "用于复核，不隐藏缺失数据与降级原因。", list(d.notes).length ? `<ul class="notes-list">${d.notes.map((note) => `<li>${e(note)}</li>`).join("")}</ul>` : '<p class="empty-small">此快照没有运行备注。</p>')}</div></div>`;
}
export function symbolDetail(d, symbol, kind) {
  const item = list(
    kind === "candidate" ? d.candidates : d.research_observations,
  ).find((r) => r.symbol === symbol);
  if (!item) return "";
  const screens = Object.entries(object(item.screens));
  const audits = list(d.audits).filter((a) => a.symbol === symbol);
  const evidence = object(d.screen_evidence);
  return `<div class="dialog-head"><div>${tag(kind === "candidate" ? "正式候选" : "研究观察", kind === "candidate" ? "blue" : "amber")} ${d.service.demo ? tag("虚构样本") : ""}<h2 id="symbol-title">${e(item.name || symbol)}</h2><p class="dialog-subtitle">${e(symbol)} · ${marketNames[market(symbol)]} · 截面 ${e(d.report.asof)}</p></div><button class="icon-button" data-action="close-dialog" aria-label="关闭详情">${icon("close")}</button></div><div class="dialog-body"><div class="muted-card">${kind === "candidate" ? "该标的属于所选快照的正式候选，仍需独立复核，不构成投资建议。" : "研究分仅反映公式内比较，不证明验收资格。是否属于正式候选，以正式候选池中的记录为准，不能据此执行交易。"}</div><div class="dialog-stats"><div><span>公式研究分</span><strong>${n(item.formula_score)}</strong></div><div><span>量化辅助分</span><strong>${n(item.qlib_score)}</strong></div><div><span>公式覆盖</span><strong>${screens.length}</strong></div></div><h3>公式分解</h3>${screens.length ? screens.map(([key, val]) => `<div class="evidence-metric"><span>${e(screenNames[key] || key)}<small>${e(key)}</small></span><span class="chart-track"><span class="chart-fill" data-width="${clamp(val)}"></span></span><strong>${n(val)}</strong></div>`).join("") : '<p class="empty-small">未提供公式分解。</p>'}<h3 class="subsection-title">AI 审核</h3>${
    audits.length
      ? audits
          .map(
            (a) =>
              `<div class="muted-card"><strong>${e(a.conclusion || "未提供结论")}</strong><p>${e(a.skip_reason || "")}</p>${list(
                a.opinions,
              )
                .map(
                  (op) =>
                    `<p>${e(op.master)} · ${e(op.stance)}：${e(op.rationale)}</p>`,
                )
                .join("")}</div>`,
          )
          .join("")
      : '<div class="muted-card">当前快照没有该标的的 AI 审核证据。未调用模型，不生成替代结论。</div>'
  }<details class="disclosure"><summary>核对原始指标与来源</summary><pre class="technical">${e(
    JSON.stringify(
      {
        source: item.source,
        snapshot_id: d.report.snapshot_id,
        screen_evidence: Object.fromEntries(
          Object.entries(evidence)
            .map(([key, rows]) => [
              key,
              list(rows).filter((r) => r.symbol === symbol),
            ])
            .filter(([, rows]) => rows.length),
        ),
        gates: d.gates,
        model: d.model,
      },
      null,
      2,
    ),
  )}</pre></details></div>`;
}
