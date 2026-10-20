"""TradingAgents 可选适配。未安装时给出友好提示，不抛 traceback。"""

from __future__ import annotations

INSTALL_HINT = """未安装辩论层 TradingAgents。

上游包不在 PyPI，请在单独虚拟环境中安装（已 pin 到 v0.7.0）：
  uv pip install -e ".[debate]"

A 股多空辩论请勿与上游同时安装（导入名都是 tradingagents），需另开 venv：
  pip install git+https://github.com/simonlin1212/TradingAgents-astock

LLM 辩论不进入 MVP 的触发线执行路径。
"""


class DebateNotInstalled(RuntimeError):
    """未安装可选 extra ``[debate]``。"""


def run_debate(symbol: str) -> str:
    """尝试导入 TradingAgents。未安装则抛 DebateNotInstalled。"""
    try:
        import tradingagents  # noqa: F401
    except ImportError as exc:
        raise DebateNotInstalled(INSTALL_HINT.strip()) from exc
    return (
        f"已检测到 tradingagents，但 MVP 默认不调用 LLM。"
        f"请使用 TradingAgents 自己的 CLI 对 {symbol} 做多空辩论；"
        f"本工具的触发线判定始终是确定性纯 Python。"
    )
