# 智库投研工作台

`zk ui` 现在启动一个只读的 Web 工作台。它是既有研究管线的展示层，不会重新训练、回测、采集行情、调用 LLM、发布审批或下单。旧 Streamlit 保留为 `zk ui --legacy`。

## 运行

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,ui]'
bash scripts/run_workspace_demo.sh
```

打开 http://127.0.0.1:8501 。脚本会在临时目录复制**仓库内的虚构规则与样例**，显式使用 demo、external AI 和固定样本日期 2026-10-20。它不会读取已有 `ZHIKU_KB`，不会调用收费模型或行情接口，不会把演示数据写进真实知识库。结束服务后临时报告保留，便于复核；终端会给出目录。

读取自己已经生成的报告：

```bash
ZHIKU_KB=/absolute/path/to/generated-kb zk ui
# 可选端口
ZHIKU_KB=/absolute/path/to/generated-kb zk ui --port 8502
```

UI 的来源以报告和知识库模式标记为准，不会因 CLI 默认 `--provider live` 把 demo 改成 live。打开 UI 不写模式标记、不创建报告，也不从其他知识库或 fixtures 回退。

## 信息结构

- **今日概览**：正式候选与研究观察分别计数，研究分布仅使用实际公式分；流程状态显示缺少什么验收证据。
- **候选与观察**：双视图切换、名称/代码搜索、市场筛选和排序。研究观察覆盖全部公式记录，可能与正式候选重叠；观察分不授予候选资格。缺少量化分显示「不可用」，不补成 0。空正式候选是合法的 fail-closed 结果。
- **消息与证据**：信源级别、时效、事件日历和应急快照。缺少应急结果显示未知；历史或演示结果不能当作实时风险状态。
- **研究验证**：公式验收、模型发布/推理状态、截面、版本标识、原始指标及运行备注。完整性验证不是样本外有效性证明。

点击标的可查看公式分解、原始指标、审核证据与产物标识。缺少 AI 审核时明确标注，不生成替代判断。所有金额、交易执行和持仓编辑都不在这个只读工作台的范围内。

## 历史快照与导出

快照菜单只列出 `reports/YYYY-MM/YYYY-MM-DD-zk-daily.json` 的受限日期注册表，默认读取 `reports/latest-daily.json`。不接受任意路径，不扫描模型文件名。历史候选保留其历史含义，但明确标记不是今天的信号。

「导出快照」导出与页面相同的**验证后公开字段投影**，不是未经筛选的原始日报。导出保留来源、截面、公式/模型状态和研究证据，不包含持仓数量、成本、账户字段或内部任务包路径。私有 KB 中的研究内容依然可能敏感：服务没有身份认证，不能直接公开它。

旧版混合候选、损坏 JSON、NaN/Infinity、来源/日期不一致、无验收出处的正式候选都会被拦截；前端提供可重试的空/错误状态。报告是已生成的受信本地管线产物，不是密码学签名文件；Web 适配器进行结构和语义校验，不重新运行产物审批。

## 内部云端验证与部署

默认只监听 `127.0.0.1`。`/healthz` 仅表示 HTTP 进程存活，**不代表行情源、模型或市场健康**。无公网链接、没有托管资源、没有公网生产部署，都不能被描述成「已上线」。

GitHub Actions 的 `Workspace browser QA` 在 Ubuntu 云 runner 启动真正的 Uvicorn 服务，再用 Chromium 访问并交互。它使用虚构数据，包含桌面 1440px 和手机 390px、截图、页面/控制台错误、请求日志、Playwright trace 与 JSON 结果。Artifact 名称包含精确运行 SHA。浏览器测试只是临时内部云服务验证，不是持久公开部署。

自行容器化运行（以下配置需在支持 Docker 的环境中构建验证）：

```bash
export ZHIKU_WORKSPACE_KB=/absolute/path/to/generated-kb
docker compose -f compose.workspace.yaml up --build
```

容器以非 root 用户运行，KB 只读挂载，宿主端口仅绑定 loopback。不复制 `.env` 或真实知识库。要远程使用，必须自行配置经过身份验证的 HTTPS 反向代理，显式设置 `--allow-host YOUR_VERIFIED_HOST`；不要直接将此无认证服务暴露到公网。CORS 默认关闭，Host 白名单、CSP 与同源限制均保留。本文不提供现成的公开部署声明。

## 验证命令

```bash
ruff check src tests scripts/qa_workspace_browser.py
mypy src/zhiku
pytest -q -rs
node --test tests/frontend/*.test.mjs
pip install -r scripts/requirements-browser.txt
python -m playwright install --with-deps chromium
python scripts/qa_workspace_browser.py --output-dir browser-artifacts
```

若本地浏览器受环境 socket 限制，可先运行 `--http-only` 检查真实 HTTP；这**不能替代浏览器验证**。以 CI 的实际浏览器结果和截图为准。普通 Python/前端单测也不证明视觉或交互通过。

前端无需 npm 构建和运行时 CDN：ES 模块、样式和图标均随 Python wheel 打包。`ui.js` 管理格式化/转义/图标，`views.js` 渲染纯视图，`app.js` 管理请求、导航与交互。所有报告文本转义；链接只允许安全的 HTTP(S)，不使用字符串事件处理器或内联脚本。

## 设计参考与取舍

阅读并应用了这些开源设计方法，而非安装未审查的 skill 脚本：

- [Anthropic frontend-design](https://github.com/anthropics/skills/blob/main/skills/frontend-design/SKILL.md)（[Apache-2.0](https://github.com/anthropics/skills/blob/main/skills/frontend-design/LICENSE.txt)）：为投研场景设定明确的字体、层级和颜色，避免模板化大标题、装饰性收益图和无意义动效。
- [shadcn/ui blocks](https://ui.shadcn.com/blocks) 与 [data table](https://ui.shadcn.com/docs/components/data-table)（[MIT](https://github.com/shadcn-ui/ui/blob/main/LICENSE.md)）：参考侧栏、工具栏、表格和渐进详情的组织方式。没有声称安装 shadcn React 组件，也没有为视觉升级引入 React/Next 重写。

深蓝导航表示工作空间；白色正文与细分隔线承载高密度信息；蓝色表示交互与研究数据，琥珀色表示限制或不确定性。系统中文字体、数值等宽排布、可见键盘焦点、原生 dialog 焦点管理、小屏导航和 reduced-motion 支持优先于装饰。

## 旧版兼容入口

```bash
pip install -e '.[legacy-ui]'
ZHIKU_KB=/absolute/path/to/generated-kb zk ui --legacy
```

旧版用于已有高级回测/触发线工作流，不是新版截图。旧版不再自动跨 KB 回退到示例规则；不在 live/未知来源下用 FixtureProvider 计算真实持仓暴露或命中率。仍建议优先使用新版只读适配器。
