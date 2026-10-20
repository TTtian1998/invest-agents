# zhiku

把个人投研流程代码化的本地工具：选股公式 → 因子打分 → 回测闸门 → 新闻/宏观**分析**（不是标题搬运）→ 触发线纪律。LLM 只做可选分析层，**不进入触发线执行路径**。

不接任何券商、不下单、不发消息（提醒通道需显式配置且默认跳过）。输出只有本地 markdown / HTML / JSON 和 stdout。

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

`.gitignore` 已忽略 `/kb/`、`*.local.yaml`、`.env`。不要把真实持仓提交进 git。

## 命令

全局选项：`--provider live|fixture`、`--kb`、`--fixtures`。

| 命令 | 作用 |
| --- | --- |
| `zk screen [greenblatt\|piotroski\|dividend\|multifactor\|all]` | 经典选股公式 |
| `zk daily` | 选股→打分→回测闸门→AI 审核→新闻/宏观分析 |
| `zk news [--kind all\|premarket-ah\|premarket-us\|midday]` | 新闻雷达（分析为主体，链接附后） |
| `zk macro` | 宏观锚点：日历、预期差、剧本、传导链、事件窗口 |
| `zk check --at 1455` | 14:55 判定；close 规则仅 PREVIEW |
| `zk close` | 收盘确认；事件窗口提示新开仓降仓位 |
| `zk exposure` | 成长仓闸门（0.60/0.50 滞回）+ 禁加 |
| `zk review --week` | 触发线命中率 + 事件剧本兑现 + 信源跟踪 |
| `zk ui` | 本地看板（需 `[ui]`），含「宏观与事件」 |
| `zk debate 600519` | 可选 TradingAgents |
| `zk backtest <公式\|trigger>` | 月度等权或触发线回测 |
| `zk factor prepare\|train\|predict` | 可选 qlib |

未装可选 extra 时命令降级或友好退出，不甩 traceback。

## 新闻雷达：分析，不是搬运

每条**入选**新闻结构化输出：`so_what` / `priced_in` / `impact` / `counter_view` / `next_check` / `confidence`，并分别标注**事实 / 解读 / 推测**。低价值（纯情绪、旧闻翻炒）只计数，不展开。报告以分析为主体，原文链接放附录。

分析层：有 `ZHIKU_LLM_KEY` 或 `OPENAI_API_KEY` 时走 LLM；没有 key 则规则模板，并标明「未做深度分析」。测试默认无 key。

### 信源分级（YAML 可配）

配置在包内 `zhiku/news/grades.yaml`，可用 `ZHIKU_KB/config/source_grades.yaml` 覆盖。等级参与打分和标签：

1. **一级·官方第一手**：Fed / BLS / BEA / 美国财政部 / SEC EDGAR / 人民银行 / 国家统计局 / 证监会 / 沪深港交易所 / 巨潮 / 国务院·发改委·商务部 / ECB / BOJ / OPEC / 公司公告。优先抓取；能接官方 RSS/API 就直接接（Fed、BLS、EDGAR）。媒体转述尽量回溯官方原文链接。
2. **二级·大媒体**：Bloomberg、Reuters、WSJ、FT、CNBC、财新、第一财经、证券时报等。RSS/公开标题源；有付费墙只取标题、摘要和链接。彭博/路透独家与快讯加权。「据知情人士」标 **未经官方证实**，并跟踪后续官方确认/否认，计入来源命中率。
3. **三级·投资机构**：券商研报、评级、目标价。必须做辩证审查：`institution_stance` / `track_record` / `key_assumptions` / `consensus_gap` / `counter_view`。**不能单独**触发「事实」标签或质量档机会。
4. 社交/自媒体只作情绪信号。

报告里每条新闻都标信源等级。

## 宏观锚点

`zk macro` 与每日报告顶部「宏观锚点」：未来 7 天核心日历（Asia/Shanghai，标出影响 A/H/美哪个时段）、昨日预期差与锚定资产反应、利率预期（FedWatch 不可用时用联邦基金期货**近似**并注明）、地缘温度（GPR fixture；GDELT 可选）、事件剧本（偏鹰/符合/偏鸽）和对持仓行业暴露（只读仓库外私有文件的**代码与行业**，没有则跳过，绝不打印数量/成本）。

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
