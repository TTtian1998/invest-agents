"""非法 YAML 必须给出清晰报错。"""

from __future__ import annotations

from pathlib import Path

import pytest

from zhiku.engine.rules import RulesSchemaError, load_symbol_book, load_symbol_books
from zhiku.models import SymbolBook


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


VALID = """
symbol: "510300"
market: CN
name: 示例
growth: false
add_blocked: true
position: {qty: 10000, cost: 4.00}
rules:
  - id: demo-380-2d
    when: {field: close, op: lt, level: 3.80, consecutive: 2}
    action: {type: sell_fraction, fraction: 0.5}
"""


def test_valid_example_parses() -> None:
    book = SymbolBook.model_validate(
        {
            "symbol": "510300",
            "market": "CN",
            "position": {"qty": 10000, "cost": 4.00},
            "rules": [
                {
                    "id": "demo-380-2d",
                    "when": {"field": "close", "op": "lt", "level": 3.80, "consecutive": 2},
                    "action": {"type": "sell_fraction", "fraction": 0.5},
                }
            ],
        }
    )
    assert book.symbol == "510300"


def test_duplicate_rule_id(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "dup.yaml",
        VALID
        + """
  - id: demo-380-2d
    when: {field: close, op: lt, level: 3.70}
    action: {type: sell_fraction, fraction: 0.5}
""",
    )
    with pytest.raises(RulesSchemaError, match="规则 id 重复"):
        load_symbol_book(path)


def test_supersedes_missing_id(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "bad.yaml",
        VALID
        + """
  - id: demo-370-1d
    when: {field: close, op: lt, level: 3.70}
    action: {type: sell_fraction, fraction: 0.5}
    supersedes: [does-not-exist]
""",
    )
    with pytest.raises(RulesSchemaError, match="supersedes 指向不存在的 id"):
        load_symbol_book(path)


@pytest.mark.parametrize("fraction", [0, -0.1, 1.01, 2])
def test_fraction_out_of_range(tmp_path: Path, fraction: float) -> None:
    path = _write(
        tmp_path / "frac.yaml",
        f"""
symbol: "510300"
market: CN
position: {{qty: 10000, cost: 4.00}}
rules:
  - id: bad-frac
    when: {{field: close, op: lt, level: 3.80}}
    action: {{type: sell_fraction, fraction: {fraction}}}
""",
    )
    with pytest.raises(RulesSchemaError, match=r"fraction 必须在 \(0, 1\]"):
        load_symbol_book(path)


def test_duplicate_id_across_files(tmp_path: Path) -> None:
    rules = tmp_path / "rules"
    rules.mkdir()
    _write(rules / "a.yaml", VALID)
    _write(
        rules / "b.yaml",
        VALID.replace('symbol: "510300"', 'symbol: "600519"'),
    )
    with pytest.raises(RulesSchemaError, match="规则 id 重复"):
        load_symbol_books(rules)
