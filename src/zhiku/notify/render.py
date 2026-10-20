"""Jinja2 渲染 markdown 报告。"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from zhiku.models import STATUS_CN

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"

_ENV = Environment(
    loader=FileSystemLoader(TEMPLATE_DIR),
    autoescape=select_autoescape(enabled_extensions=()),
    trim_blocks=True,
    lstrip_blocks=True,
)
_ENV.globals["status_cn"] = lambda s: STATUS_CN.get(s, getattr(s, "value", str(s)))
_ENV.filters["pct"] = lambda v: "" if v is None else f"{v * 100:.1f}%"
_ENV.filters["num"] = lambda v: "缺数据" if v is None else f"{v:g}"


def render(name: str, **ctx: Any) -> str:
    return _ENV.get_template(name).render(**ctx)


def report_path(reports_root: Path, on: date, kind: str) -> Path:
    folder = reports_root / on.strftime("%Y-%m")
    return folder / f"{on.isoformat()}-zk-{kind}.md"


def write_report(reports_root: Path, on: date, kind: str, markdown: str) -> Path:
    path = report_path(reports_root, on, kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")
    return path


def latest_json(reports_root: Path, name: str) -> Path | None:
    latest = reports_root / f"latest-{name}.json"
    if latest.is_file():
        return latest
    files = sorted(reports_root.glob(f"*/*-zk-{name}.json"))
    return files[-1] if files else None


def latest_daily_json(reports_root: Path) -> Path | None:
    return latest_json(reports_root, "daily")


