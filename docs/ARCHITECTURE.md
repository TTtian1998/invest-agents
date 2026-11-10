# zhiku 架构：研究、发布、每日执行

本地投研与提醒工具，不接券商、不下单。保持现有公式、PIT 回测与因子研究，但不让历史研究或隐式模型切换进入 daily。

## 三条明确路径

1. 研究：`data sync` → `research gates` / `factor train` / `factor research*` / `backtest`。
2. 显式发布：`research publish-gates PATH`、`factor publish PATH`。研究输出不是正式准入，文件名叫 selected/locked 也不会自动生效。
3. 每日执行：当前截面 → 读取兼容、有效、已发布的公式验收与模型 → 正式候选和独立研究观察 → 确定性清单 → 一层证据驱动 AI 研判 → 报告快照。

`daily` 不训练、不回测、不补历史成分、不拟合新闻历史权重。`run-daily` 仅在 live 模式做增量同步，然后调用 daily；demo/fixture/disabled 不同步。`--skip-train` 保留兼容并提示弃用。

## 正式准入是 fail-closed

- 正式候选只能来自已发布且通过验收的公式，量化模型只辅助池内排序。
- 价格/覆盖/半段超额/基准等证据缺失，不等于通过。发布重新检查有限数值、正整数覆盖计数、时点引擎、T+1 成交、净值证据及基准，不能只提交 `passed_gate=true`。
- 缺发布、损坏、未来日期、过期、provider/config/code 不匹配，公式不进入正式池。live 必须有新鲜正数有限价格与估值截面；缺 PIT 指数成分不回退今天名单。历史/未来 live 日报的完整基本面来源尚未验证，因此仅输出研究观察，正式候选关闭。
- 所有公式截面仍可在 `observations` 与各公式榜单查看，明确标 research_only。新闻机会也是研究观察，不替代正式准入。
- 不再以“公式为空”为由把量化分升成 100% 正式分数。`merge.formula_first=false` 仅供显式研究调用，daily 始终强制公式准入。

## 小型文件接口，不引入框架

- `pipeline/research.py`：本地 PIT 历史验收，输入价格截断到 asof，不隐式联网。缺完整 T+1 开盘/时点财务则数据不足。
- `pipeline/artifacts.py`：公式研究结果与显式发布。SHA-256 绑定结果、输入摘要、数据截止日、配置、schema/策略版本及实际策略源码摘要。
- `models/gate-research/<sha>.json`：不可变研究证据。
- `models/approved-gates/<sha>.json` + `approved.json`：发布证据和原子更新的指针。
- `factors/model_artifacts.py`：研究模型身份/校验和/样本数/推理接口验证，版本化发布至 `models/published/<version>/`，`models/approved.json` 指向唯一批准版本。
- `reports/daily-snapshots/<date>-<sha>.json`：每次不同输出保留历史，日期日报与 latest 是便利视图。
- `reports/review-snapshots/<sha>.json`：不可变 AI 输入，日期 review-pack 是最新别名。

两个发布操作都不由 daily 调用。真实发布时间按 Asia/Shanghai 日期约束 live 日报，不能用 --date 回填过去的审批。验收有效期同时约束数据截止日与最后净值证据，不允许重跑旧数据续期；策略和比较基准的 T+1 开盘均要求实际收盘替代次数为零。无效发布不会替换上一个有效版本。有效的负面公式验收可显式发布空准入池。哈希用于检测损坏/误替换，不防御能同时修改本地模型、代码与清单的攻击者。pickle 仅加载可信本地训练产物，不是安全交换格式。

## 模型身份与验证含义

研究训练保留 qlib Alpha158、lgb_lite、面板模型和各轮实验。生产预测只认已发布 manifest，不扫描 selected.pkl、locked_recipe.json 或文件 mtime。元数据与 payload 实际 kind、features、训练配置、来源/配置指纹必须一致。

目前可发布推理实现：lgb_lite、lgb_panel、zscore_panel。qlib_lgb 仍可研究训练，但没有实现等价的生产推理适配，所以拒绝发布/预测，不偷换为 lgb_lite。空训练、未拟合、无样本、非有限探针、过期/未来数据都拒绝；保留上个有效产物。

模型 publication_status=approved 表示身份、数据时间、训练非空和推理接口验收，不代表有盈利或样本外预测力。日报仍明确“量化样本外未验证”。未绑定版本的 live_rank_ic.json 不再自动放大 daily 权重，研究加载接口保留。

## 数据源与研究隔离

CLI 与 run_daily 以 `.zhiku-source-mode` 绑定独立知识库；不同模式不得共享新的报告/日志目录，迁移旧混合数据请参阅 MIGRATION.md。

- live：真实源或本地真实缓存；失败/空值显示缺数据，绝不借用 fixture。
- demo：显式离线虚构数据；fixture 为兼容别名。禁止自动网络 AI、行情同步和交易日历抓取，即使环境设置 ZHIKU_LIVE_CALENDAR=1。
- disabled：空结果，来源明确停用，不读取 fixture、不抓取、不训练历史权重。
- `news.enabled=false`、`macro.enabled=false` 优先停用。demo 新闻研究日志放在 `journal/demo/`，不会污染新 live 研究日志。
- `news --research` 显式计算历史统计/权重；默认新闻/宏观只有确定性分析，使用固定默认权重，不读取未发布的历史拟合权重。历史 live 日期不使用今天的无时间戳新闻/锚点/利率伪装历史；日历过滤未来实际值。

旧版未注明来源的新闻权重/日志无法自动证明来源，升级前应人工核查或归档；不要把曾用于 demo 的旧知识库直接用于 live。

## AI 与决策日志

规则清单保留各投资人思想，但 daily 不再自动调用旧大师 LLM 补充或 TradingAgents。后者留在显式 `debate` 研究命令。

单一 AI 阶段消费保存的证据快照：支持整包后端一次调用，通用后端每候选一次调用。它不能改变确定性分数、公式闸门或触发线。

`export-review --date DATE` 只读保存的快照，不重跑 daily。外部结果必须回传 snapshot_id/asof，候选、rank/score、引用须与快照相符。`ingest-review` 验证后写 journal，并只在当前日期报告仍指向相同快照时重新渲染；否则仅保留历史决策。同一快照重复写回幂等，冲突决定拒绝。daily 只复用完全一致的结果，不借用昨天或同日旧证据的判断。

## 默认有效期与迁移

`artifacts.gate_max_age_days=45`、`model_max_age_days=45`、`price_max_age_days=7` 可在知识库 pipeline.yaml 覆盖。修改策略源码或准入相关配置后需重新研究/发布，不能把旧验证搬到新策略。

详见 [迁移操作](MIGRATION.md)。离线示例的月度收盘数据不足以证明时点/T+1 验收，demo 默认无正式候选是正确结果，不是通过假审批补齐展示。研究观察、新闻、宏观和报告仍可离线演示。
