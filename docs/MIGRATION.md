# 从隐式 daily 流程迁移

本重构提交基于 [PR #4 的类型修复](https://github.com/TTtian1998/invest-agents/pull/4)（fix/mypy-ci-types），不替换或合并该 PR。集成时先处理 #4，再将本重构 PR retarget 到 main 并复跑 CI。

## 一次性操作

1. 保留现有知识库；本次改动不删除公式、模型研究、历史报告或 journal。
2. 旧 selected.pkl / locked_recipe.json 不会被 daily 自动采纳。用当前版本重训，检查实际 kind、n_train、日期/来源/配置与研究结果。
3. 显式计算并检查公式验收证据，再分别发布公式和模型。不要把下面占位路径原样执行。

```bash
export ZHIKU_KB=/path/to/private-kb
zk --provider live data sync --universe csi800 --years 3
zk --provider live research gates --date 2026-10-22
# 上一步输出 models/gate-research/<hash>.json；先查看统计、时点池、覆盖与备注
zk --provider live research publish-gates /path/to/gate-research/<hash>.json --date 2026-10-22
zk --provider live factor prepare
zk --provider live factor train
# 仅发布训练命令输出的受支持且上下文完整的研究模型；不会隐式选另一个模型
zk --provider live factor publish /path/to/models/lgb_lite.pkl --date 2026-10-22
zk --provider live daily --date 2026-10-22
```

研究日期和发布日期必须与数据时间相容，过期或未来证据会拒绝。没有足够验证数据时，保留研究输出并补齐数据，不改成裸 passed=true。qlib_lgb 暂无等价生产推理适配，仅保留研究用途；不能发布成其他 kind。

`factor train --model` 旧参数以前被忽略，现在明确报错；用 `factor publish PATH` 选择发布对象。`factor predict --date DATE` 只使用有效发布。历史研究回测命令不等于正式准入发布。

## 每日 cron

```bash
# live：增量同步 + daily，不再按月重训
zk --provider live run-daily
# 已由其他任务同步时
zk --provider live run-daily --skip-sync
# 兼容旧 cron，明确提示无须这个参数
zk --provider live run-daily --skip-train
```

训练和历史验收由你独立安排，不在每日关键路径里自动执行。研究产物过期会产生清晰缺失/过期状态，而不会偷偷重新训练或切换模型。

## 离线演示

```bash
ZHIKU_LLM_BACKEND=external bash examples/run_demo.sh
zk --provider demo run-daily --force --date 2026-10-20
```

demo/fixture 永不自动同步 live，也不自动调用网络 AI。示例股票、财务、新闻均虚构。样本只有稀疏月度收盘数据，缺时点/T+1 验证时正式候选为空，但研究观察和报告完整可看。`news.enabled=false` / `macro.enabled=false` 会真正停用，不再加载虚构新闻作为回退。

## 外部 AI 写回

1. `daily --date DATE` 生成报告与不可变 review snapshot。
2. `export-review --date DATE` 返回已保存快照路径，不抓取、不训练、不调用 AI。
3. 外部 JSON 使用 zhiku.review_result.v1，原样带回 snapshot_id 和 asof，只引用包中字段/摘录。
4. `ingest-review result.json` 校验、记录，并在日报仍匹配该快照时更新原日报显示。不将旧结果放到次日证据上。

旧版没有 snapshot_id 的结果仍留在历史日志供对账，但不能作为新日报审批；请重新审核对应新包。相同导入幂等，冲突导入明确拒绝。快照/日报便利别名可更新，不可变存档不会覆盖。


## 知识库来源隔离

一个 KB 只允许一种来源模式。首次运行创建 `.zhiku-source-mode` 标记，后续 live / demo / disabled 混用会直接报错，不会污染共享 journal、review、AI 决策或 latest 报告。fixture 与 demo 同义。请为演示创建单独目录，仅复制规则/配置，不复制 live 日志或数据；已有混合旧日志请先归档到只读目录，再使用新的专用 KB。不要删除标记来绕过边界。direct `run_daily` API 同样执行这个检查。

live 日报仅接受配置窗口内的有效正数价格和估值截面。指数成分按 asof 读取，缺历史快照不使用今天名单。历史/未来 live 日报完整基本面 PIT 来源尚未完成验证，因此正式候选保持关闭，仍可使用明确的 research/backtest 命令；不会将回放伪装成当时已获准的正式信号。
