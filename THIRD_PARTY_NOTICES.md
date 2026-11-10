# 第三方许可声明

本项目是胶水层，行情与可选辩论能力来自下列开源项目。分发本软件时请一并保留其许可。

| 项目 | 用途 | 许可证 |
| --- | --- | --- |
| [akshare](https://github.com/akfamily/akshare) | A 股 / ETF 日线、分钟线、龙虎榜、两融、解禁 | MIT |
| [yfinance](https://github.com/ranaroussi/yfinance) | 港股、美股日线与财报日期 | Apache-2.0 |
| [TradingAgents](https://github.com/TauricResearch/TradingAgents) | 可选多空辩论层（`[debate]` extra，pin `v0.7.0`） | Apache-2.0 |
| [TradingAgents-astock](https://github.com/simonlin1212/TradingAgents-astock) | A 股辩论（需单独 venv，勿与上游同时安装） | 见上游仓库 |
| [a-stock-data](https://github.com/cqu20160901/a-stock-data) | A 股数据参考实现（本仓库不直接依赖） | Apache-2.0 |
| [pydantic](https://github.com/pydantic/pydantic) | 规则与数据模型 | MIT |
| [typer](https://github.com/fastapi/typer) | CLI | MIT |
| [Jinja2](https://github.com/pallets/jinja) | markdown 报告模板 | BSD-3-Clause |
| [PyYAML](https://github.com/yaml/pyyaml) | 规则 / 持仓 YAML | MIT |
| [pandas](https://github.com/pandas-dev/pandas) | 表格处理 | BSD-3-Clause |
| [quantstats](https://github.com/ranaroussi/quantstats) | 可选绩效统计（`[stats]` extra） | Apache-2.0 |
| [streamlit](https://github.com/streamlit/streamlit) | 旧版兼容看板（`[legacy-ui]` extra） | Apache-2.0 |
| [qlib](https://github.com/microsoft/qlib) | 可选 Alpha158 因子（`[qlib]` extra） | MIT |
| [vectorbt](https://github.com/polakowo/vectorbt) | 可选回测（`[backtest]` extra） | Apache-2.0 |

TradingAgents 与 TradingAgents-astock 在 PyPI / 导入名上都叫 `tradingagents`，**不要**写进同一环境。A 股辩论请另开虚拟环境安装：

```text
pip install git+https://github.com/simonlin1212/TradingAgents-astock
```

## 新版 Web 工作台

- [FastAPI](https://github.com/fastapi/fastapi)（MIT）：只读 HTTP 适配器。
- [Uvicorn](https://github.com/encode/uvicorn)（BSD-3-Clause）：ASGI 服务。
- [Playwright](https://github.com/microsoft/playwright-python)（Apache-2.0）：仅开发/CI 浏览器验证。
- 视觉设计参考 Anthropic frontend-design（Apache-2.0）与 shadcn/ui（MIT）的公开设计原则，未复制其 skill 文本或组件实现。具体链接和应用取舍见 [工作台指南](docs/WORKSPACE.md)。
