#!/usr/bin/env bash
# Isolated, fictional demo; never reads the caller's ZHIKU_KB or secrets.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEMO_KB="$(mktemp -d "${TMPDIR:-/tmp}/zhiku-workspace-demo.XXXXXX")"
cp -R "${ROOT}/examples/kb/." "${DEMO_KB}/"
export ZHIKU_KB="${DEMO_KB}"
export ZHIKU_FIXTURES="${ROOT}/examples/fixtures"
export ZHIKU_LLM_BACKEND=external
export ZHIKU_ANALYSIS_LLM=0
export ZHIKU_LIVE_CALENDAR=0
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}"
python -m zhiku --provider demo run-daily --force --date 2026-10-20
printf '\n虚构演示知识库：%s\n停止服务不会删除报告；需要时自行删除此临时目录。\n' "${DEMO_KB}"
exec python -m zhiku --provider demo ui --host 127.0.0.1 --port "${PORT:-8501}"
