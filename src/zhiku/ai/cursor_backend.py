"""Cursor 研判后端：官方 SDK local 优先，不可用再走 Cloud Agents API。

不是 OpenAI 兼容 chat/completions。解析失败或超时只报错，不退回规则模板。
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import urllib.error
import urllib.request
from typing import Any, Callable

from zhiku.ai.backend import parse_json_object
from zhiku.ai.pack import review_result_from_obj
from zhiku.ai.schema import ReviewResult
from zhiku.ai.secrets import cursor_key, redact_secret

DEFAULT_MODEL = "composer-2.5"
DEFAULT_API_BASE = "https://api.cursor.com"
DEFAULT_TIMEOUT_SEC = 300.0

PACK_PROMPT = (
    "你是投研助手，只根据下面的研判任务包做判断。不要编造包外数字，不要写持仓数量、成本、账户。"
    "stance 只能是 研究跟进 / 观察 / 回避。每条主张必须有 cite。缺数据写「缺数据」。"
    "只输出一个 JSON 对象，schema 为 zhiku.review_result.v1，字段见 docs/ai_layer.md："
    "extractions[], judgments[]（long/short 各含 claim+cite）, regime。"
    "不要输出 markdown 说明。"
)


class CursorUnavailable(RuntimeError):
    """SDK 未安装或本地运行时不可用，调用方可以改走 Cloud Agents API。"""


class CursorBackend:
    """ZHIKU_LLM_BACKEND=cursor。密钥只来自 CURSOR_API_KEY。"""

    name = "cursor"

    def __init__(
        self,
        *,
        sdk_call: Callable[[str], str] | None = None,
        cloud_call: Callable[[str], str] | None = None,
        prefer_sdk: bool | None = None,
    ) -> None:
        self._sdk_call = sdk_call
        self._cloud_call = cloud_call
        self._prefer_sdk = prefer_sdk

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        prompt = (
            f"{system}\n\n{user}\n\n"
            "只输出一个 JSON 对象。不要编造包外数字。解析失败由调用方报错，不要用规则模板凑答案。"
        )
        text = self._complete_text(prompt)
        parsed = parse_json_object(text)
        if parsed is None:
            raise RuntimeError("Cursor 未返回 JSON 对象")
        return parsed

    def complete_review_pack(self, pack: dict[str, Any]) -> ReviewResult:
        payload = json.dumps(pack, ensure_ascii=False)
        prompt = f"{PACK_PROMPT}\n\n任务包 JSON：\n{payload}"
        text = self._complete_text(prompt)
        parsed = parse_json_object(text)
        if parsed is None:
            raise RuntimeError("Cursor 未返回 zhiku.review_result.v1 JSON 对象")
        result = review_result_from_obj(parsed)
        result.backend = self.name
        return result

    def _complete_text(self, prompt: str) -> str:
        if not cursor_key() and self._sdk_call is None and self._cloud_call is None:
            raise RuntimeError("缺 CURSOR_API_KEY")
        use_sdk = self._prefer_sdk if self._prefer_sdk is not None else _sdk_importable()
        if use_sdk:
            try:
                return self._run_sdk(prompt)
            except CursorUnavailable as exc:
                return self._run_cloud(prompt, sdk_note=str(exc))
        return self._run_cloud(prompt)

    def _run_sdk(self, prompt: str) -> str:
        if self._sdk_call is not None:
            return self._sdk_call(prompt)
        return _sdk_local_prompt(prompt)

    def _run_cloud(self, prompt: str, *, sdk_note: str | None = None) -> str:
        try:
            if self._cloud_call is not None:
                return self._cloud_call(prompt)
            return _cloud_agents_prompt(prompt)
        except Exception as exc:  # noqa: BLE001
            extra = f"SDK 不可用（{redact_secret(sdk_note)}）；" if sdk_note else ""
            raise RuntimeError(redact_secret(f"{extra}Cloud Agents API 失败：{exc}")) from None


def _sdk_importable() -> bool:
    try:
        import cursor_sdk  # noqa: F401
    except ImportError:
        return False
    return True


def _timeout_sec() -> float:
    raw = os.environ.get("ZHIKU_CURSOR_TIMEOUT_SEC")
    try:
        return float(raw) if raw else DEFAULT_TIMEOUT_SEC
    except ValueError:
        return DEFAULT_TIMEOUT_SEC


def _model_id() -> str:
    return (os.environ.get("ZHIKU_LLM_MODEL") or DEFAULT_MODEL).strip() or DEFAULT_MODEL


def _api_base() -> str:
    return (os.environ.get("ZHIKU_CURSOR_API_BASE") or DEFAULT_API_BASE).rstrip("/")


def _sdk_local_prompt(prompt: str) -> str:
    try:
        from cursor_sdk import Agent, AgentOptions, LocalAgentOptions
    except ImportError as exc:
        raise CursorUnavailable("未安装 cursor-sdk") from exc
    key = cursor_key()
    if not key:
        raise RuntimeError("缺 CURSOR_API_KEY")
    tmp = tempfile.mkdtemp(prefix="zhiku-cursor-")
    try:
        options = AgentOptions(
            model=_model_id(),
            api_key=key,
            local=LocalAgentOptions(cwd=tmp),
        )
        result = Agent.prompt(prompt, options)
        text = _result_text(result)
        if not text:
            raise RuntimeError("Cursor SDK 返回空文本")
        return text
    except CursorUnavailable:
        raise
    except ImportError as exc:
        raise CursorUnavailable(redact_secret(f"Cursor SDK 导入失败：{exc}")) from None
    except Exception as exc:  # noqa: BLE001
        msg = redact_secret(str(exc))
        lowered = msg.lower()
        if any(k in lowered for k in ("not installed", "未安装", "no such", "cannot find", "runtime")):
            raise CursorUnavailable(f"Cursor SDK 本地运行时不可用：{msg}") from None
        if "timeout" in lowered or "超时" in msg:
            raise TimeoutError("Cursor SDK 超时") from None
        raise RuntimeError(f"Cursor SDK 失败：{msg}") from None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _result_text(result: Any) -> str:
    if result is None:
        return ""
    for attr in ("result", "text"):
        value = getattr(result, attr, None)
        if callable(value):
            try:
                value = value()
            except TypeError:
                value = None
        if isinstance(value, str) and value.strip():
            return value
    return str(result).strip()


def _cloud_agents_prompt(prompt: str) -> str:
    key = cursor_key()
    if not key:
        raise RuntimeError("缺 CURSOR_API_KEY")
    created = _http_json(
        "POST",
        f"{_api_base()}/v1/agents",
        key,
        body={
            "prompt": {"text": prompt},
            "name": "zhiku-review",
            "model": {"id": _model_id()},
        },
    )
    agent = created.get("agent") if isinstance(created.get("agent"), dict) else created
    run = created.get("run") if isinstance(created.get("run"), dict) else {}
    agent_id = agent.get("id") if isinstance(agent, dict) else None
    run_id = (run.get("id") if isinstance(run, dict) else None) or (
        agent.get("latestRunId") if isinstance(agent, dict) else None
    )
    if not agent_id or not run_id:
        raise RuntimeError("Cloud Agents API 未返回 agent/run id")
    deadline = time.monotonic() + _timeout_sec()
    while time.monotonic() < deadline:
        info = _http_json("GET", f"{_api_base()}/v1/agents/{agent_id}/runs/{run_id}", key)
        status = str(info.get("status") or "").upper()
        if status == "FINISHED":
            text = info.get("result")
            if not isinstance(text, str) or not text.strip():
                raise RuntimeError("Cloud Agents API 结束但结果为空")
            return text
        if status in {"ERROR", "CANCELLED", "EXPIRED"}:
            raise RuntimeError(f"Cloud Agents API run 状态={status}")
        time.sleep(2.0)
    raise TimeoutError("Cursor Cloud Agents API 超时")


def _http_json(method: str, url: str, key: str, *, body: dict | None = None) -> dict[str, Any]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    timeout = min(_timeout_sec(), 60.0)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}") from None
    except TimeoutError:
        raise TimeoutError("Cursor HTTP 超时") from None
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(redact_secret(f"HTTP 失败：{exc}")) from None
    if not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Cloud Agents API 返回了非 JSON") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("Cloud Agents API 返回了非对象")
    return parsed
