#!/usr/bin/env bash
# cron 示例（Asia/Shanghai 每个交易日 15:30）：
#   30 15 * * 1-5 TZ=Asia/Shanghai ZHIKU_KB=/path/to/kb /path/to/scripts/run-daily.sh
# 非交易日 zk run-daily 以退出码 0 跳过。不要把真实持仓放进仓库。
set -euo pipefail
export TZ="${TZ:-Asia/Shanghai}"
export ZHIKU_KB="${ZHIKU_KB:?请设置 ZHIKU_KB 指向仓库外知识库}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if command -v zk >/dev/null 2>&1; then
  ZK=zk
else
  ZK="python -m zhiku"
fi
exec $ZK --provider live run-daily "$@"
