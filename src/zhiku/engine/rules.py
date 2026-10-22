"""读取并校验 ``rules/*.yaml`` 与 ``portfolio/positions.yaml``。"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from zhiku.models import Portfolio, SymbolBook


class RulesSchemaError(ValueError):
    """规则 YAML 不合法时的可读错误。"""


def _load_yaml(path: Path) -> object:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RulesSchemaError(f"无法读取 {path}: {exc}") from exc
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise RulesSchemaError(f"{path.name} YAML 无法解析: {exc}") from exc


def _format_validation(path: Path, exc: ValidationError | ValueError) -> str:
    if isinstance(exc, ValueError) and not isinstance(exc, ValidationError):
        return f"{path.name}: {exc}"
    parts: list[str] = []
    if isinstance(exc, ValidationError):
        for err in exc.errors():
            loc = ".".join(str(x) for x in err.get("loc", ()))
            msg = err.get("msg", str(err))
            parts.append(f"{loc}: {msg}" if loc else msg)
    text = "; ".join(parts) if parts else str(exc)
    return f"{path.name}: {text}"


def load_symbol_book(path: Path) -> SymbolBook:
    data = _load_yaml(path)
    if not isinstance(data, dict):
        raise RulesSchemaError(f"{path.name}: 顶层必须是 mapping")
    try:
        return SymbolBook.model_validate(data)
    except (ValidationError, ValueError) as exc:
        raise RulesSchemaError(_format_validation(path, exc)) from exc


def load_symbol_books(rules_dir: Path) -> list[SymbolBook]:
    if not rules_dir.is_dir():
        raise RulesSchemaError(f"规则目录不存在: {rules_dir}")
    files = sorted(rules_dir.glob("*.yaml")) + sorted(rules_dir.glob("*.yml"))
    # glob 可能对同一文件匹配两次（当后缀就是 yaml 时不会），去重
    uniq: list[Path] = []
    seen: set[Path] = set()
    for path in files:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        uniq.append(path)
    if not uniq:
        raise RulesSchemaError(f"规则目录为空: {rules_dir}")
    books = [load_symbol_book(path) for path in uniq]
    ids: list[str] = []
    for book in books:
        ids.extend(r.id for r in book.rules)
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise RulesSchemaError(f"规则 id 重复: {', '.join(dupes)}")
    return books


def load_portfolio(path: Path) -> Portfolio:
    data = _load_yaml(path)
    if not isinstance(data, dict):
        raise RulesSchemaError(f"{path.name}: 顶层必须是 mapping")
    try:
        return Portfolio.model_validate(data)
    except (ValidationError, ValueError) as exc:
        raise RulesSchemaError(_format_validation(path, exc)) from exc
