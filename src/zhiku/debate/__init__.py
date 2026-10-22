"""可选辩论层适配。默认不安装、默认测试不依赖。"""

from zhiku.debate.tradingagents_adapter import DebateNotInstalled, run_debate

__all__ = ["DebateNotInstalled", "run_debate"]
