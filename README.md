# zhiku

把个人投研流程里的「价位触发线 + 14:55/收盘确认 + 成长仓上限 + 周五命中率复盘」代码化的个人工具。LLM 多空辩论层可选，**不进入触发线执行路径**。

不接任何券商、不下单、不发消息。输出只有本地 markdown 和 stdout。

## 站在谁的肩膀上

| 项目 | 本仓库怎么用 | 许可证 |
| --- | --- | --- |
| [TradingAgents](https://github.com/TauricResearch/TradingAgents) | 可选 `[debate]` extra，多空辩论（pin `v0.7.0`） | Apache-2.0 |
| [TradingAgents-astock](https://github.com/simonlin1212/TradingAgents-astock) | A 股辩论需**单独 venv**（导入名同为 `tradingagents`，不要和上游装在一起） | 见上游 |
| [akshare](https://github.com/akfamily/akshare) | A 股 / ETF 日线、1 分钟 14:55 价、龙虎榜、两融、解禁 | MIT |
| [a-stock-data](https://github.com/cqu20160901/a-stock-data) | A 股数据参考，本仓库不直接依赖 | Apache-2.0 |
| [yfinance](https://github.com/ranaroussi/yfinance) | 港股、美股日线与财报日期 | Apache-2.0 |
| [daily_stock_analysis](https://github.com/JohnnyWang1994/daily_stock_analysis) | 日报结构参考，本仓库不直接依赖 | 见上游 |
| [vectorbt](https://github.com/polakowo/vectorbt) | 路线图 v0.3 触发线回测，MVP 未接入 | Apache-2.0 |
| [quantstats](https://github.com/ranaroussi/quantstats) | 可选 `[stats]` extra，MVP 未接入执行路径 | Apache-2.0 |

完整第三方声明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 安装与 5 分钟上手

需要 Python ≥ 3.11。

```bash
uv venv && uv pip install -e ".[dev]"
bash examples/run_demo.sh
```

demo 使用离线 `FixtureProvider` 和 `examples/` 里的**虚构**标的（510300、600519、AAPL、0700.HK、2800.HK），数量、成本、价位都是编造的。成功后会在 `examples/kb/reports/` 写下检查 / 收盘 / 暴露 / 周报 markdown。

真实规则和持仓文件请放在**仓库外**，用环境变量指向：

```bash
export ZHIKU_KB=/path/to/your-kb   # 默认 ./kb
zk --provider live check --at 1455
```

`.gitignore` 已忽略用户自己的 `/kb/`、`*.local.yaml` 和 `.env`。不要把真实持仓提交进 git。

## 规则 YAML

`ZHIKU_KB/rules/*.yaml` 每个文件一个标的：

```yaml
symbol: "510300"
market: CN            # CN | HK | US
name: 示例宽基ETF
growth: false         # 是否计入成长仓
add_blocked: true     # 禁加
position: {qty: 10000, cost: 4.00}   # 示例里的数量/成本均为虚构
rules:
  - id: demo-380-2d
    when: {field: close, op: lt, level: 3.80, consecutive: 2}
    action: {type: sell_fraction, fraction: 0.5}
  - id: demo-370-1d
    when: {field: close, op: lt, level: 3.70}
    action: {type: sell_fraction, fraction: 0.5}
    supersedes: [demo-380-2d]
```

语义：

- `field`：`price_1455`（仅 CN）、`close`、`open`、`low`。
- `op`：`lt`、`le`、`gt`、`ge`。
- `buffer_pct`（可选，默认 0）：有效线 = `level × (1 − buffer_pct/100)`（`lt/le`）或 `level × (1 + buffer_pct/100)`（`gt/ge`）。
- `consecutive`（默认 1）：最近 N 个**交易日**都满足才触发；遇非交易日跳过而非重置；任一日缺数据 → **缺数据**，不是「未触发」。不估算、不插值。
- `and`：附加条件列表。目前支持 `{field: volume_ratio, op: ge, level: 1.0}`（量比 = 当日成交量 / 前一交易日成交量）。
- `action.type`：`sell_qty`（`qty`）、`sell_fraction`（`fraction` ∈ (0, 1]）、`sell_all`、`alert`。
- `supersedes`：同一天高优先级规则也命中时，被覆盖的规则不单独输出；动作合并取**更大数量**（跳空一次卖出）。`sell_qty` 可设 `cumulative: true`，表示与被覆盖规则数量相加。
- 结果状态：`TRIGGERED` / `NOT_TRIGGERED` / `MISSING_DATA`（另有盘中 `PREVIEW`、港股 `NOT_CLOSED`），每条都带实际值、有效线、价源、时间。

`ZHIKU_KB/portfolio/positions.yaml` 用于成长仓闸门，可手填 `fx.USD_CNY` / `fx.HKD_CNY`（MVP **不联网取汇率**）。

## 命令参考

全局选项：`--provider live|fixture`（示例和测试走 `fixture`）、`--kb`、`--fixtures`。

| 命令 | 作用 |
| --- | --- |
| `zk check --at 1455 [--date YYYY-MM-DD]` | 只正式评估 `price_1455`；`close` 规则用 14:55 价做 **PREVIEW** 预警，不算触发 |
| `zk close [--date YYYY-MM-DD]` | 评估 `close/open/low`。CN 15:00；HK 16:10（未到则标「未收盘」）；US 用最近一个已完成的美东交易日并注明 |
| `zk exposure` | 成长仓占比。`>=0.60` 冻结；`<0.50` 正常；中间带滞回。`add_blocked` 输出「禁加」 |
| `zk review --week [--end YYYY-MM-DD]` | 本周一至周五命中率（T+1 / T+3），写周报；候选教训只打印 |
| `zk debate 600519` | 可选 TradingAgents。未安装时提示安装并以非零码退出 |

真正 `TRIGGERED` 时追加 `kb/journal/trigger_events.jsonl`（按 `rule_id+date` 幂等）。报告在 `kb/reports/YYYY-MM/`。

时区一律按 **Asia/Shanghai** 显示；美股价格保留其美东交易日。

A 股交易日历来自 akshare，失败则退化为工作日并警告。资金流走东财 `fflow/daykline`，新浪备源，东财请求串行限流（`EM_MIN_INTERVAL`，默认 ≥1s + 抖动）。北向个股净买入固定为「缺数据：交易所已停披露」。

## 路线图

- **v0.2** 辩论接入：把 TradingAgents / TradingAgents-astock 的结论引用进收盘报告（仍不改触发线）。
- **v0.3** 用 vectorbt 对触发线做离线回测。
- **v0.4** 新闻雷达与质量门槛。
- **v0.5** MCP server / 本地推送（仍然不接券商）。

## 隐私

仓库内只有通用示例标的和明显虚构的数量、成本、价位。不要提交真实持仓、账户、券商、人名或个人价位线。

## 免责声明

这是研究与提醒工具，**不构成投资建议**，不自动下单。数据源为非官方接口，仅限个人研究。任何数据源失败都会记为「缺数据」，程序不会替你补数。

## 开发

```bash
uv pip install -e ".[dev]"
ruff check src tests
mypy src/zhiku
pytest                 # 默认 -m "not network"
```

可选 extra：

```bash
uv pip install -e ".[debate]"   # 上游 TradingAgents @ v0.7.0
uv pip install -e ".[stats]"    # quantstats，尚未接入执行路径
```
