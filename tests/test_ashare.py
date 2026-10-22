"""akshare 函数名兜底：在单测里 mock，不访问网络。"""

from __future__ import annotations

from types import SimpleNamespace

from zhiku.data.ashare import _call_first_available
from zhiku.data.fundflow import northbound_stock_net_buy
from zhiku.models import MissingData


def test_call_first_available_skips_missing_then_failing() -> None:
    def boom(**kwargs):
        raise RuntimeError("第一候选失败")

    def ok(**kwargs):
        return {"ok": kwargs.get("symbol")}

    module = SimpleNamespace(first=boom, second=ok)
    result = _call_first_available(
        ["ghost", "first", "second"],
        symbol="600519",
        module=module,
    )
    assert result == {"ok": "600519"}


def test_call_first_available_all_fail_returns_missing() -> None:
    def boom(**kwargs):
        raise RuntimeError("挂了")

    module = SimpleNamespace(a=boom)
    result = _call_first_available(["missing", "a"], module=module)
    assert isinstance(result, MissingData)
    assert "调用失败" in result.reason


def test_northbound_hardcoded_missing() -> None:
    miss = northbound_stock_net_buy("600519")
    assert isinstance(miss, MissingData)
    assert "交易所已停披露" in miss.reason
