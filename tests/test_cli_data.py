"""zk data / factor CLI：离线假数据，不访问网络。"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from zhiku.cli import app

RUNNER = CliRunner()


def test_data_sync_help() -> None:
    result = RUNNER.invoke(app, ["data", "sync", "--help"])
    assert result.exit_code == 0
    assert "universe" in result.output


def test_factor_predict_untrained(tmp_path: Path, examples_dir: Path) -> None:
    env = {
        "ZHIKU_KB": str(tmp_path / "kb"),
        "ZHIKU_FIXTURES": str(examples_dir / "fixtures"),
    }
    result = RUNNER.invoke(app, ["--provider", "fixture", "factor", "predict"], env=env)
    assert result.exit_code == 0
    text = result.output + (result.stderr or "")
    assert "not_published" in text
    assert "factor publish" in text
