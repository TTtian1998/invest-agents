# zhiku

把个人投研流程代码化的本地工具：研究/验证 → 显式版本发布 → 每日正式候选与研究观察 → 单次 AI 审核 → 触发线纪律。LLM 只做可选分析层，**不进入触发线执行路径**。

不接任何券商、不下单、不发消息（提醒通道需显式配置且默认跳过）。输出只有本地 markdown / HTML / JSON 和 stdout。

## 产品截图 / Screenshots

以下看板截图来自仓库内离线 demo（`bash examples/run_demo.sh` 后 `zk ui`）。**全部是虚构示例数据**：标的只有 510300、600519、AAPL、0700.HK、2800.HK、000333 等示例代码，页面上的价位、财务、新闻和宏观数字都是编造的。不含真实持仓、数量、成本、账户或个人信息。

同一次 demo 产出的完整日报：[docs/sample_daily_report.md](docs/sample_daily_report.md)。

### 今日候选股

按公式打出的当日候选表；运行备注和未过门栏收在折叠里，表格先露出来。

![今日候选股](docs/screenshots/candidates.png)

### 单票详情

单票分项得分与可选 AI 审核；没配 key 时明确跳过，不编造结论。

![单票详情](docs/screenshots/stock_detail.png)

### 公式回测

各选股公式在虚构价上的月度等权回测与闸门结果。

![公式回测](docs/screenshots/backtest.png)

### 新闻舆情

今日舆情以分析卡片为主体：事实 / 解读 / 推测、失效条件与时间止损，不是标题搬运。

![新闻舆情](docs/screenshots/news_sentiment.png)

### 大V动态

关注名单上的财经账号动态、共振与账号命中率（帖子为虚构样本）。

![大V动态](docs/screenshots/kol_feed.png)

### 多空机会

机会分质量 / 交易 / 只观察三档；应急模式下全部降为只观察。

![多空机会](docs/screenshots/opportunities.png)

### 宏观与事件

宏观日历、预期差、锚定资产与事件剧本（时区 Asia/Shanghai）。

![宏观与事件](docs/screenshots/macro_events.png)

### 应急与拥挤

黑天鹅应急清单与拥挤度反身性；只列受影响代码，不展示数量或成本。

![应急与拥挤](docs/screenshots/crowding_emergency.png)

### 持仓触发线风控

触发线与禁加状态，读取示例知识库而非真实账户。

![持仓触发线风控](docs/screenshots/trigger_risk.png)

### 每周命中率

触发线 T+1 / T+3 命中率、拥挤度分组与大V跟踪复盘。

![每周命中率](docs/screenshots/weekly_hitrate.png)

## 5 分钟上手

需要 Python ≥ 3.11。

```bash
uv venv && uv pip install -e ".[dev]"
bash examples/run_demo.sh
```

demo 使用离线 `FixtureProvider` 和 `examples/` 里的**虚构**标的（510300、600519、AAPL、0700.HK、2800.HK、000333），数量、成本、价位、新闻和宏观数字都是编造的。成功后会在 `examples/kb/reports/` 写下检查 / 收盘 / 暴露 / 周报 / 每日 / 新闻 / 宏观 markdown。

真实规则和持仓请放在**仓库外**：

```bash
export ZHIKU_KB=/path/to/your-kb   # 默认 ./kb
zk --provider live check --at 1455
zk --provider live daily --date 2026-10-20
```

`.gitignore` 已忽略 `/kb/`、`*.local.yaml`、`.env`、本地行情库 `**/data/`。不要把真实持仓提交进 git。

## 真实数据（live）

离线 demo 之后，把股票池从 `fundamentals.yaml` 换成全市场本地库。数据写在 `ZHIKU_KB/data/`（SQLite，可选 parquet），**不进 git**。

```bash
export ZHIKU_KB=/path/to/your-kb          # 仓库外；不要放真实持仓
uv pip install -e ".[dev]"
# 可选：uv pip install -e ".[qlib]"       # pyqlib Alpha158；没有则 pandas+LightGBM
# 日线备用源：pip install baostock        # 东财 hist 常熔断时走 baostock 子进程
zk data sync --universe csi800 --years 3 --workers 4
# 全市场：zk data sync --universe all --years 3
# 中断后续传：同一 ZHIKU_KB 再跑（默认 --resume）；挂死标的记 missing，不插值
zk factor prepare
zk factor train          # 仅生成研究候选，不自动发布
zk research gates --date 2026-10-22  # 本地 PIT 验收，输出不可变 JSON
# 检查证据后：zk research publish-gates /path/to/<hash>.json --date 2026-10-22
# 检查模型后：zk factor publish /path/to/models/lgb_lite.pkl --date 2026-10-22
zk --provider live daily --date 2026-10-22
# 交易日 cron（Asia/Shanghai 15:30）：
# 30 15 * * 1-5 TZ=Asia/Shanghai ZHIKU_KB=/path/to/kb zk --provider live run-daily
```

- A 股：列表（含 ST、停牌、上市/退市日）、指数成分、前复权+不复权日线、东财整表 + **新浪三大表明细**（营业利润/流动资产负债/固定资产，公告日 PIT）、整表分红（股息率用「每10股派息/收盘」或官方股息率，不再把派息当市值比）。
- 港美：`--include-global` 时按配置拉恒生、恒生科技、标普500、纳指100。
- 主源 akshare，日线备用 baostock 子进程 / tushare（`TUSHARE_TOKEN`）。FRED / X 无 token 就跳过并标注。
- live 新闻/宏观**不混入**离线样本；缺真实数据写「缺数据」，利率/地缘/应急冲击不回落 fixture。
- 宏观事件窗口只给核心数据与央行（可配白名单）；会议/展会重要度封顶且不建窗口。公开日程补未来 7 天 FOMC。
- 经典公式（Greenblatt / Piotroski / 分红）只用可交易过滤，不套 ROE/PE 质量门槛；门槛可配。
- live 回测用本地库前复权；增量截面只写不复权，前复权按涨跌平移。回测池用**历史时点成分**（baostock 沪深300+中证500 回查中证800；不够则用月末已上市且有日线的全 A，并在报告标明）。超额和闸门对**同池时点等权**，对照行必须是闸门实际用的 **T+1 开盘** 等权，不是月末收盘等权。过闸前提：覆盖月数 ≥ 80%（不足标「数据不足」）；然后超额 > 0、Sharpe ≥ 等权、前后半段超额都 > 0。禁止用今日静态名单回测（前视）。
- 量化分默认权重 0.10，只做过闸公式内排序辅助；没有可用的前瞻验证时 brief 写「量化样本外未验证」。未绑定版本的 live_rank_ic.json 不再自动放大权重。正式候选只从已发布过闸公式池取；量化不能单独进池。未过闸与数据不足的公式分不计入总分，只在单票详情展示。应急必须有 VIX/指数价格确认，摘要写触发原因。

新版不沿用旧实测表作为准入依据。历史研究仍见 `docs/research/`，实际可用状态以已发布版本为准。升级步骤、边界与兼容参数见 [MIGRATION.md](docs/MIGRATION.md)；完整分层见 [ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## 命令

全局选项：`--provider live|demo|disabled`（fixture 为 demo 兼容别名）、`--kb`、`--fixtures`。

| 命令 | 作用 |
| --- | --- |
| `zk screen [greenblatt\|piotroski\|dividend\|multifactor\|all]` | 经典选股公式 |
| `zk daily` | 读当前数据/已发布验收→正式候选/研究观察→单次 AI 审核 |
| `zk run-daily` | 非交易日跳过；live 增量同步→daily→≤150 行摘要，不重训 |
| `zk news [--kind all\|premarket-ah\|premarket-us\|midday]` | 新闻雷达（分析为主体，链接附后） |
| `zk macro` | 宏观锚点：日历、预期差、剧本、传导链、事件窗口 |
| `zk check --at 1455` | 14:55 判定；close 规则仅 PREVIEW |
| `zk close` | 收盘确认；事件窗口提示新开仓降仓位 |
| `zk exposure` | 成长仓闸门（0.60/0.50 滞回）+ 禁加 |
| `zk review --week` | 触发线命中率 + 事件剧本兑现 + 信源跟踪 |
| `zk ui` | 本地看板（需 `[ui]`），含「宏观与事件」 |
| `zk debate 600519` | 可选 TradingAgents |
| `zk export-review --date DATE` | 只导出已保存证据快照，不重跑 daily |
| `zk ingest-review path.json` | 精确匹配 snapshot_id/asof，记录并更新相同快照日报 |
| `zk review-scorecard` | AI「研究跟进 / 回避」T+5、T+20 对同池等权超额，对照公式排名 |
| `zk data sync` | 全市场增量同步到 `ZHIKU_KB/data`（可断点、限速） |
| `zk backtest <公式\|trigger>` | 月度等权或触发线回测；live 用本地库价格 |
| `zk factor prepare\|train\|predict` | 准备/研究训练/已发布模型预测 |
| `zk factor publish PATH` | 验证身份、非空训练、推理接口并显式发布模型 |
| `zk research gates` / `publish-gates PATH` | 历史 PIT 验证 / 重新检查证据后显式发布公式准入 |

未装可选 extra 时命令降级或友好退出，不甩 traceback。

## 新闻雷达：分析，不是搬运

每条**入选**新闻结构化输出：`so_what` / `priced_in` / `impact` / `counter_view` / `next_check` / `confidence`，并分别标注**事实 / 解读 / 推测**。低价值（纯情绪、旧闻翻炒）只计数，不展开。报告以分析为主体，原文链接放附录。

新闻/宏观默认只做确定性规则分析，标明「未做深度分析」；不再重复调用独立 LLM。历史统计更新需要显式 `zk news --research`，demo 研究日志隔离。

研判层（`docs/ai_layer.md`）：公式/量化初筛，AI 判断候选前 10。后端 `ZHIKU_LLM_BACKEND=auto|openai|cursor|mock|external`。OpenAI 兼容走 `ZHIKU_LLM_KEY`；Cursor 走 `CURSOR_API_KEY` + `cursor-sdk`（local，失败再 Cloud Agents API）。无 key 时导出不可变 review snapshot，`zk ingest-review` 绑定同一证据写回；`zk review-scorecard` 对账 T+5/T+20 超额。结论进 `journal/ai_decisions.jsonl`。密钥不进 git。

### 信源分级（YAML 可配）

配置在包内 `zhiku/news/grades.yaml`，可用 `ZHIKU_KB/config/source_grades.yaml` 覆盖。等级参与打分和标签：

1. **一级·官方第一手**：Fed / BLS / BEA / 美国财政部 / SEC EDGAR / 人民银行 / 国家统计局 / 证监会 / 沪深港交易所 / 巨潮 / 国务院·发改委·商务部 / ECB / BOJ / OPEC / 公司公告。优先抓取；能接官方 RSS/API 就直接接（Fed、BLS、EDGAR）。媒体转述尽量回溯官方原文链接。
2. **二级·大媒体**：Bloomberg、Reuters、WSJ、FT、CNBC、财新、第一财经、证券时报等。RSS/公开标题源；有付费墙只取标题、摘要和链接。彭博/路透独家与快讯加权。「据知情人士」标 **未经官方证实**，并跟踪后续官方确认/否认，计入来源命中率。
3. **三级·投资机构**：券商研报、评级、目标价。必须做辩证审查：`institution_stance` / `track_record` / `key_assumptions` / `consensus_gap` / `counter_view`。**不能单独**触发「事实」标签或质量档机会。
4. 社交/自媒体只作情绪信号。

### X 财经大V

可配置关注名单见仓库 `config/x_accounts.example.yaml`（复制到 `ZHIKU_KB/config/x_accounts.yaml`）。分类：`newswire` 快讯、`macro_fed` 美联储/宏观、`institutional` 机构大佬、`flow_trader` 交易/资金流、`contrarian` 反向参考。

采集走 X API v2，Bearer 读 `TWITTER_BEARER_TOKEN` 或 `X_BEARER_TOKEN`；没配 token 就跳过并在报告里标注。支持限流与 SQLite 缓存。离线测试用 `examples/fixtures/x_posts.yaml` 虚构帖子。

规则：快讯必须匹配到一级或二级信源后才能标「事实」；机构帖按三级做辩证审查（写出已知公开持仓代码与利益关系，不写数量）；`macro_fed` 在 FOMC 事件窗口内加权；帖子里的股票代码进入机会雷达；多账号同向标「共振」，情绪极度一致提示反向风险。每个账号的关键判断计入每周命中率，并按命中率动态调整权重。报告和看板有独立一节「大V动态」。

报告里每条新闻都标信源等级。

### 事件驱动层

打分由**预期差**驱动（一致预期 / 标准化 surprise / 美股隐含涨跌幅），不是消息「好坏」。缺预期写「缺数据」。每条记录 `first_seen` 与新颖度、发布时段（盘前/盘中/盘后/隔夜）和 A 股 T+1 最早可执行时点；超过半衰期标「已过时」。拥挤度综合融资、北向、主力、龙虎榜、换手，以及美股 put/call、做空、零售热度；拥挤时好消息标「利好出尽风险」。产业链 YAML 与地缘传导合并，事件打出一阶/二阶受影响方向。每个机会必带 `invalidation` 与 `time_stop`，到期触发线提示「未兑现，撤」，回测用同一时间止损。日历外冲击必须有价格确认（VIX、指数跌幅）才进入应急；关键词单独不触发，摘要写出指标与阈值。判断写入只追加的 `journal/decisions.jsonl`。回测财务按**公告日**对齐，处理退市幸存者偏差，并有前视偏差测试。

## 宏观锚点

`zk macro` 与每日报告顶部「宏观锚点」：未来 7 天**核心**日历（美国非农/CPI/PCE/FOMC、中国 CPI/PPI/GDP/社融/LPR、各大央行；会议/展会不建窗口）。live 利率无期货则写「缺数据」，不回落 fixture。事件剧本（偏鹰/符合/偏鸽）和对持仓行业暴露（只读仓库外私有文件的**代码与行业**，没有则跳过，绝不打印数量/成本）。

重大事件前后生成事件窗口，触发线与多空机会提示「新开仓降仓位/止损放宽或等落地」；`macro.freeze_adds_in_window` 可配窗口内禁加。

缺一致预期、缺标准差、缺价格一律写「缺数据」，不编造。

## 可选 extra

```bash
uv pip install -e ".[ui]"        # Streamlit 看板
uv pip install -e ".[qlib]"      # Alpha158 + LightGBM，未装则公式分
uv pip install -e ".[backtest]"  # vectorbt，未装则纯 pandas
uv pip install -e ".[masters]"   # openai 大师审核
uv pip install -e ".[debate]"    # TradingAgents @ v0.7.0
uv pip install -e ".[full]"      # ui+backtest+stats+masters（不含 git 辩论）
```

FRED 走环境变量 `FRED_API_KEY`，可选安装 `fredapi` / `pandas-datareader`，未装则跳过。

## 站在谁的肩膀上

| 项目 | 用法 | 许可证 |
| --- | --- | --- |
| [akshare](https://github.com/akfamily/akshare) | A 股行情、宏观、公告 | MIT |
| [yfinance](https://github.com/ranaroussi/yfinance) | 港美股、期货近似 | Apache-2.0 |
| [TradingAgents](https://github.com/TauricResearch/TradingAgents) | 可选辩论 pin v0.7.0 | Apache-2.0 |
| [qlib](https://github.com/microsoft/qlib) | 可选因子 | MIT |
| [vectorbt](https://github.com/polakowo/vectorbt) | 可选回测 | Apache-2.0 |

完整声明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。架构见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## 隐私

仓库内只有通用示例标的和明显虚构的数量、成本、价位、新闻与宏观数字。不要提交真实持仓、账户、券商、人名或个人价位线。新闻/宏观报告最多提到持仓**代码**，不输出数量与成本。

## 免责声明

这是研究与提醒工具，**不构成投资建议**，不自动下单。数据源为非官方接口，仅限个人研究。任何数据源失败都记为「缺数据」，程序不会替你补数。

## 开发

```bash
uv pip install -e ".[dev]"
ruff check src tests
mypy src/zhiku
pytest                 # 默认 -m "not network"
```

## ETF 组合策略研究（实验性）

新增离线 `zk research etf --spec ... --bars ... --split ... --out ...`：固定参数动量轮动与同约束等权再平衡，目标权重/现金、T+1 开盘及每日组合账本、时间顺序样本外报告。研究结果不会自动发布或进入 daily。可运行的**虚构数据工程示例**与实数据限制见 [ETF_STRATEGY_RESEARCH.md](docs/ETF_STRATEGY_RESEARCH.md)。
