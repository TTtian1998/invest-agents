"""Cursor 后端：只测 mock，不真调 SDK / Cloud Agents API。虚构标的。"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from zhiku.ai.backend import resolve_backend
from zhiku.ai.cursor_backend import CursorBackend, CursorUnavailable
from zhiku.ai.pack import RESULT_SCHEMA
from zhiku.ai.runner import run_ai_layer
from zhiku.ai.secrets import redact_secret
from zhiku.config import Paths
from zhiku.pipeline.settings import PipelineSettings
from tests.test_screens import _row

_FICTIONAL_RESULT = {
    "schema": RESULT_SCHEMA,
    "asof": "2026-10-21",
    "backend": "cursor",
    "judgments": [
        {
            "symbol": "000333",
            "name": "虚构家电",
            "stance": "观察",
            "confidence": 0.4,
            "invalidation": "ROE 转负",
            "time_stop_days": 10,
            "long": [{"claim": "ROE 仍为正", "cite": "roe"}],
            "short": [{"claim": "缺少当日公告原文", "cite": "缺数据"}],
            "formula_rank": 1,
        }
    ],
    "regime": {
        "style": "混合",
        "value_or_growth": "缺数据",
        "prefer_screens": ["piotroski"],
        "reason": "虚构近窗",
        "cite": "universe_12_1",
    },
}


def test_auto_openai_beats_cursor(monkeypatch) -> None:
    monkeypatch.setenv("ZHIKU_LLM_KEY", "sk-test")
    monkeypatch.setenv("CURSOR_API_KEY", "not-a-real-key")
    monkeypatch.delenv("ZHIKU_LLM_BACKEND", raising=False)
    assert resolve_backend("auto").name == "openai"


def test_auto_cursor_when_only_cursor_key(monkeypatch) -> None:
    monkeypatch.delenv("ZHIKU_LLM_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("CURSOR_API_KEY", "not-a-real-key")
    monkeypatch.delenv("ZHIKU_LLM_BACKEND", raising=False)
    assert resolve_backend("auto").name == "cursor"


def test_explicit_cursor_backend(monkeypatch) -> None:
    monkeypatch.delenv("CURSOR_API_KEY", raising=False)
    assert resolve_backend("cursor").name == "cursor"


def test_redact_does_not_leak_prefix() -> None:
    prefix = "crs" + "r_"
    raw = f"Authorization: Bearer {prefix}abcd1234 CURSOR_API_KEY={prefix}zzzz"
    out = redact_secret(raw)
    assert prefix + "abcd" not in out
    assert prefix + "zzzz" not in out
    assert "***" in out


def test_sdk_pack_success_no_network() -> None:
    import json

    backend = CursorBackend(
        sdk_call=lambda prompt: json.dumps(_FICTIONAL_RESULT, ensure_ascii=False),
        prefer_sdk=True,
    )
    monkey_key = "CURSOR_API_KEY"
    # 无真实 key 时注入 sdk_call 仍应工作
    result = backend.complete_review_pack({"schema": "zhiku.review_pack.v1", "asof": "2026-10-21"})
    assert result.judgments[0].symbol == "000333"
    assert result.judgments[0].long_case[0].cite
    assert result.backend == "cursor"
    del monkey_key


def test_sdk_unavailable_falls_to_cloud() -> None:
    import json

    calls: list[str] = []

    def boom(_prompt: str) -> str:
        raise CursorUnavailable("未安装 cursor-sdk")

    def cloud(prompt: str) -> str:
        calls.append(prompt)
        return json.dumps(_FICTIONAL_RESULT, ensure_ascii=False)

    backend = CursorBackend(sdk_call=boom, cloud_call=cloud, prefer_sdk=True)
    parsed = backend.complete_json("任务=抽取", "虚构家电首次超预期")
    assert parsed.get("schema") == RESULT_SCHEMA or "stance" in str(parsed) or parsed
    assert calls


def test_parse_failure_does_not_use_rules() -> None:
    backend = CursorBackend(sdk_call=lambda _p: "这不是 JSON", prefer_sdk=True)
    with pytest.raises(RuntimeError, match="JSON"):
        backend.complete_json("任务=抽取", "下调亏损")


def test_timeout_does_not_use_rules() -> None:
    def boom(_prompt: str) -> str:
        raise TimeoutError("Cursor SDK 超时")

    backend = CursorBackend(sdk_call=boom, prefer_sdk=True)
    with pytest.raises(TimeoutError):
        backend.complete_json("任务=研判辩论", "买入 买入")


def test_runner_uses_pack_not_per_symbol(tmp_path: Path, examples_dir: Path) -> None:
    calls = {"json": 0, "pack": 0}

    class FakeCursor:
        name = "cursor"

        def complete_json(self, system: str, user: str) -> dict:
            calls["json"] += 1
            raise AssertionError("打包路径不应逐票 complete_json")

        def complete_review_pack(self, pack: dict) -> object:
            calls["pack"] += 1
            from zhiku.ai.pack import review_result_from_obj

            from copy import deepcopy

            raw = deepcopy(_FICTIONAL_RESULT)
            raw["snapshot_id"] = pack["snapshot_id"]
            raw["regime"]["cite"] = "缺数据"
            return review_result_from_obj(raw)

    paths = Paths.load(kb=tmp_path, fixtures=examples_dir / "fixtures")
    paths.ensure_runtime_dirs()
    result, pack, notes = run_ai_layer(
        asof=date(2026, 10, 21),
        paths=paths,
        settings=PipelineSettings(),
        candidates=[{"symbol": "000333", "name": "虚构家电", "formula_score": 80}],
        rows=[_row(symbol="000333", name="虚构家电", pe=8, roe=0.2)],
        news={"clusters": []},
        macro={},
        backend=FakeCursor(),
    )
    assert calls["pack"] == 1 and calls["json"] == 0
    assert result is not None and result.judgments[0].stance == "观察"
    assert pack is not None and pack.is_file()
    del notes


def test_runner_cursor_failure_is_honest(tmp_path: Path, examples_dir: Path) -> None:
    class FakeCursor:
        name = "cursor"

        def complete_review_pack(self, pack: dict) -> object:
            raise TimeoutError("Cursor SDK 超时")

    paths = Paths.load(kb=tmp_path, fixtures=examples_dir / "fixtures")
    paths.ensure_runtime_dirs()
    result, _pack, notes = run_ai_layer(
        asof=date(2026, 10, 21),
        paths=paths,
        settings=PipelineSettings(),
        candidates=[{"symbol": "000333", "name": "虚构家电", "formula_score": 80}],
        rows=[_row(symbol="000333", name="虚构家电", pe=8, roe=0.2)],
        news={"clusters": []},
        macro={},
        backend=FakeCursor(),
    )
    assert result is None
    assert any("不装成已判断" in n and "超时" in n for n in notes)
