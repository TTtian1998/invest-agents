"""可插拔 LLM：OpenAI 兼容、模拟、外部（只导出任务包）。"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Protocol


def llm_key() -> str | None:
    return os.environ.get("ZHIKU_LLM_KEY") or os.environ.get("OPENAI_API_KEY") or None


def cursor_key() -> str | None:
    from zhiku.ai.secrets import cursor_key as _cursor_key

    return _cursor_key()


def backend_name() -> str:
    raw = (os.environ.get("ZHIKU_LLM_BACKEND") or "auto").strip().lower()
    if raw in {"auto", "openai", "mock", "external", "cursor"}:
        return raw
    return "auto"


class LLMBackend(Protocol):
    name: str

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        """返回一个 JSON 对象。失败应抛异常，由调用方降级。"""


class MockBackend:
    """离线测试用。只根据任务包里已有的清单/原文做确定性输出，不访问网络。"""

    name = "mock"

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        task = _task_kind(system)
        if task == "extract":
            return _mock_extract(user)
        if task == "judge":
            return _mock_judge(user)
        if task == "regime":
            return _mock_regime(user)
        return {"error": "unknown_task"}


class OpenAICompatBackend:
    name = "openai"

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        key = llm_key()
        if not key:
            raise RuntimeError("缺 ZHIKU_LLM_KEY / OPENAI_API_KEY")
        from openai import OpenAI

        kwargs: dict[str, Any] = {"api_key": key}
        base = os.environ.get("ZHIKU_LLM_BASE_URL")
        if base:
            kwargs["base_url"] = base
        client = OpenAI(**kwargs)
        model = os.environ.get("ZHIKU_LLM_MODEL", "gpt-4o-mini")
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
        )
        text = (resp.choices[0].message.content or "").strip()
        parsed = parse_json_object(text)
        if parsed is None:
            raise RuntimeError("模型没有返回 JSON 对象")
        return parsed


class ExternalBackend:
    """不调用模型。调用方应导出任务包。"""

    name = "external"

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        raise RuntimeError("external 后端不调用模型，请 zk export-review / ingest-review")


def resolve_backend(name: str | None = None) -> LLMBackend:
    chosen = (name or backend_name()).lower()
    if chosen == "auto":
        env = (os.environ.get("ZHIKU_LLM_BACKEND") or "").strip().lower()
        if env == "mock":
            return MockBackend()
        if llm_key():
            return OpenAICompatBackend()
        if cursor_key():
            from zhiku.ai.cursor_backend import CursorBackend

            return CursorBackend()
        return ExternalBackend()
    if chosen == "mock":
        return MockBackend()
    if chosen == "openai":
        return OpenAICompatBackend()
    if chosen == "cursor":
        from zhiku.ai.cursor_backend import CursorBackend

        return CursorBackend()
    if chosen == "external":
        return ExternalBackend()
    raise ValueError(f"未知后端 {name}")


def _task_kind(system: str) -> str:
    if "抽取" in system or "extract" in system.lower():
        return "extract"
    if "辩论" in system or "研判" in system or "judge" in system.lower():
        return "judge"
    if "市场状态" in system or "regime" in system.lower():
        return "regime"
    return "unknown"


def parse_json_object(text: str) -> dict | None:
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.S)
    if fence:
        try:
            data = json.loads(fence.group(1))
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None


def _mock_extract(user: str) -> dict[str, Any]:
    text = user
    direction = 0
    event = "其它"
    surprise = 0.0
    if any(k in text for k in ("超预期", "增长", "上调", "中标")):
        direction = 1
        event = "业绩/订单"
        surprise = 0.4
    if any(k in text for k in ("下调", "亏损", "造假", "立案")):
        direction = -1
        event = "负面公告"
        surprise = 0.6
    novelty = 0.7 if "首次" in text or "NEW" in text else 0.3
    cite = "headline"
    for marker in ("headline", "原文", "摘录"):
        if marker in text:
            cite = marker
            break
    return {
        "event_type": event,
        "direction": direction,
        "surprise": surprise,
        "novelty": novelty,
        "cite": cite,
    }


def _mock_judge(user: str) -> dict[str, Any]:
    buys = user.count("买入")
    avoids = user.count("回避")
    if avoids > buys:
        stance = "回避"
        confidence = 0.55
    elif buys >= 2:
        stance = "研究跟进"
        confidence = 0.6
    else:
        stance = "观察"
        confidence = 0.45
    cite = "checklist"
    if "roe" in user.lower():
        cite = "roe"
    elif "momentum_12_1" in user:
        cite = "momentum_12_1"
    return {
        "stance": stance,
        "confidence": confidence,
        "invalidation": "清单引用字段翻号或新闻被官方否认",
        "time_stop_days": 20,
        "long": [{"claim": "规则清单偏多", "cite": cite}],
        "short": [{"claim": "规则清单有回避项或字段缺失", "cite": cite}],
    }


def _mock_regime(user: str) -> dict[str, Any]:
    mom = "动量" if ("动量为正" in user or "12-1>0" in user or "mom_pos" in user) else "反转"
    prefer = ["livermore", "canslim"] if mom == "动量" else ["graham", "greenblatt"]
    return {
        "style": mom,
        "value_or_growth": "价值" if mom == "反转" else "成长",
        "prefer_screens": prefer,
        "reason": "近窗收益差的启发式，不是已发表体制模型",
        "cite": "universe_12_1",
    }
