# AI 研判层

分工：公式和量化做初筛，AI 做判断。没有 LLM key 时不装成已经判断，导出研判任务包，由外部代理（含每天 15:30 的 Cloud Agent）按本文写回。不是投资建议。

## 1. 发力点与出处

### a. 文本抽取 → 可回测因子

把公告、财报附注、新闻原文抽成：事件类型、影响方向、与预期的差异、新颖度，并带原文摘录。这是文本因子，不是「情绪分装饰」。

公开实践：

- Tetlock (2007) 用媒体悲观措辞解释市场收益，是词典法的基准，不是 LLM。
- Loughran & McDonald (2011) 说明通用情感词典在财报里会误标（比如 liability），领域词表比通用词表重要。
- Lopez-Lira & Tang (2023, arXiv:2304.07619；修订 2024) 用 ChatGPT 给新闻标题打正/负/中，样本从模型训练截止日**之后**起算，分数对后续日收益有样本外预测力，小盘和负面新闻更强。
- Yang, Liu & Wang (2023, arXiv:2306.06031) FinGPT：开源金融 LLM 的数据管线与 LoRA，应用包括情绪分析；我们不训练自己的 FinGPT，只借用「文本 → 结构化任务」的分工。

我们的实现：抽取结果写入决策日志，字段固定，才能在 T+5 / T+20 对同池等权算超额。缺原文就记缺数据，不编。

### b. 前 10 名：清单 + 多空辩论

规则清单（是/否/缺数据 + 字段引用）先跑，AI 只在清单之上做多空辩论，输出「研究跟进 / 观察 / 回避」、置信度、失效条件、时间止损。每一条主张必须引用当时可见的字段或原文，不能空口。

公开实践：

- Xiao, Sun, Luo & Wang (2024/2025, arXiv:2412.20138) TradingAgents：基本面/情绪/新闻/技术分析师 → 多空研究员辩论 → 交易员 → 风控。仓库已有可选 `[debate]` extra，默认**不**把上游图接进触发线。本层只复用「先举证、再辩论、再裁决」的结构，不复制其源码。
- 角色划分对齐 virattt/ai-hedge-fund 的公开框架说明（Graham / Buffett / Munger / Lynch），**未复制**其提示词或代码。清单正文来自各人公开著作，见 `docs/research/investor_principles.md`。

没有 key 时：清单照常跑（已有），辩论留空，任务包交给外部代理。

### c. 市场状态

用宏观锚点（利率预期、地缘温度、硬日历）和近窗风格（动量或反转、价值或成长的简单收益差）建议当前更值得信哪类公式，并写进日志，事后用 scorecard 检验。这是启发式记录，不是 Dalio 全天候，也不是已发表的体制转换模型。

Lopez-Lira & Tang 的结果不能直接当成「A 股该信动量」；LSY (2019) 仍提示中国主异象是反转。状态建议必须事后对账，不能当日改闸门。

## 2. 后端

环境变量（都不进 git）：

| 变量 | 作用 |
| --- | --- |
| `ZHIKU_LLM_BACKEND` | `auto` / `openai` / `cursor` / `mock` / `external` |
| `ZHIKU_LLM_KEY` 或 `OPENAI_API_KEY` | OpenAI 兼容接口的 key |
| `CURSOR_API_KEY` | Cursor 后端专用（只读这个变量，不进仓库） |
| `ZHIKU_LLM_BASE_URL` | OpenAI 兼容基址（DeepSeek、Qwen、本地 vLLM/Ollama 兼容层） |
| `ZHIKU_LLM_MODEL` | 模型名。openai 默认 `gpt-4o-mini`；cursor 默认 `composer-2.5` |
| `ZHIKU_CURSOR_API_BASE` | Cloud Agents API 基址，默认 `https://api.cursor.com` |
| `ZHIKU_CURSOR_TIMEOUT_SEC` | Cursor 超时秒数，默认 300 |

`auto`：显式 `mock` 用模拟；有 `ZHIKU_LLM_KEY` / `OPENAI_API_KEY` 走 OpenAI 兼容；否则有 `CURSOR_API_KEY` 走 Cursor；否则 `external`（只导出任务包）。

### Cursor 后端

Cursor 没有 OpenAI 兼容的 `/v1/chat/completions`。`ZHIKU_LLM_BACKEND=cursor` 时：

1. 优先用官方 Python 包 `cursor-sdk` 的 **local** 运行时：不需要本仓库，把 `export-review` 任务包放进 prompt，要求只返回 `zhiku.review_result.v1` JSON。
2. SDK 未安装或本地运行时不可用时，退到 Cloud Agents API（`POST /v1/agents`，再按 run 轮询）。
3. 解析失败或超时只报错，**不**退回规则模板冒充已经判断。

本机需要：`pip install cursor-sdk`（或 `pip install -e ".[cursor]"`），以及 Cursor Dashboard 签发的用户/服务账号 key。把 key 写进环境变量 `CURSOR_API_KEY`，或复制 `.env.example` 为 `.env`（`.env` 已 gitignore）。日志会脱敏。仓库扫描禁止出现官方密钥前缀字符串。

外部模式：

1. `zk export-review`（日报在无 key 时也会写）产出 `reports/YYYY-MM-DD-review-pack.json`
2. 外部代理按第 4 节填 `review_result.v1`
3. `zk ingest-review path.json` 校验后写入决策日志，并出现在次日报告

每天 15:30 的例行 Cloud Agent 按本文担任这个外部代理：读任务包，只使用包里的候选、截面、新闻摘录和清单，不编造包外数字，写回结果。

## 3. 防止装饰

- 每条结论进 `ZHIKU_KB/journal/ai_decisions.jsonl`，带 asof、时间戳、当时可见字段快照、引用来源、后端名称。
- `zk review-scorecard`：把「研究跟进」和「回避」两组在记录日之后第 5、第 20 个交易日对**同池等权**的超额算出来，并与当日纯公式排名前 N 对照。没到期的标「待定」，不填。
- **前视：** Sarkar & Vafa (2024, SSRN 4754678) 证明预训练语料含未来信息时，用 LLM 做历史文本研究有前视偏差。Lopez-Lira & Tang 因此只用截止日之后的标题。本工具：历史回测文本因子只用「模型截止日之后的样本」，或者只用决策日志里当时写下的前瞻记录；不得拿今天的模型去给 2020 年新闻打分再当样本外。

## 4. 任务包 / 写回格式

任务包 `zhiku.review_pack.v1`：`asof`、候选（代码、公式分、量化分、来源公式）、截面字段、大师清单、新闻原文摘录、宏观锚点、`docs/ai_layer.md` 的约束摘要。不含持仓数量、成本、账户。

写回 `zhiku.review_result.v1`：

- `extractions[]`：`symbol, event_type, direction (-1/0/+1), surprise, novelty, cite`
- `judgments[]`：`symbol, stance (研究跟进|观察|回避), confidence, invalidation, time_stop_days, long[], short[]`；`long`/`short` 每条有 `claim` 和 `cite`
- `regime`：`style (动量|反转|混合), value_or_growth, prefer_screens[], reason, cite`

缺数据写 `"缺数据"`，stance 不要用「买入」。时间止损用交易日。

## 参考文献

1. Tetlock, P. C. (2007). Giving content to investor sentiment: The role of media in the stock market. *Journal of Finance*, 62(3), 1139–1168.
2. Loughran, T., & McDonald, B. (2011). When is a liability not a liability? Textual analysis, dictionaries, and 10-Ks. *Journal of Finance*, 66(1), 35–65.
3. Lopez-Lira, A., & Tang, Y. (2023). Can ChatGPT forecast stock price movements? Return predictability and large language models. arXiv:2304.07619.
4. Yang, H., Liu, X.-Y., & Wang, C. D. (2023). FinGPT: Open-source financial large language models. arXiv:2306.06031.
5. Xiao, Y., Sun, E., Luo, D., & Wang, W. (2024). TradingAgents: Multi-agents LLM financial trading framework. arXiv:2412.20138. https://github.com/TauricResearch/TradingAgents
6. Sarkar, S. K., & Vafa, K. (2024). Lookahead bias in pretrained language models. SSRN 4754678.
7. Liu, J., Stambaugh, R. F., & Yuan, Y. (2019). Size and value in China. *Journal of Financial Economics*, 134(1), 48–69.
8. virattt. *AI Hedge Fund* (public repository). https://github.com/virattt/ai-hedge-fund — 仅对照角色划分，未复制源码。
