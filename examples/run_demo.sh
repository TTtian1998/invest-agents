#!/usr/bin/env bash
# 用 FixtureProvider 跑一遍全部命令（离线）。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ZHIKU_KB="${ROOT}/examples/kb"
export ZHIKU_FIXTURES="${ROOT}/examples/fixtures"
cd "${ROOT}"

if ! command -v zk >/dev/null 2>&1; then
  echo "找不到 zk。请先：uv pip install -e \".[dev]\"" >&2
  exit 1
fi

zk --provider fixture check --at 1455 --date 2026-10-19
zk --provider fixture close --date 2026-10-20
zk --provider fixture exposure
zk --provider fixture review --week --end 2026-10-23
zk --provider fixture screen all
zk --provider fixture daily --date 2026-10-20
zk --provider fixture news --date 2026-10-20
zk --provider fixture macro --date 2026-10-20

echo
echo "报告目录：${ZHIKU_KB}/reports/"
find "${ZHIKU_KB}/reports" -name '*.md' | sort
