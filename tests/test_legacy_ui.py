"""Keep legacy risk messaging truthful while new workspace is the default."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("streamlit")
from zhiku.app import streamlit_app


@pytest.mark.parametrize("news", [None, {}, {"emergency": {}}])
def test_missing_emergency_never_claims_normal(monkeypatch, news):
    st = SimpleNamespace(subheader=Mock(), warning=Mock(), info=Mock())
    monkeypatch.setattr(streamlit_app, "st", st)
    streamlit_app._emergency_panel(news)
    st.warning.assert_called_once()
    assert "未知" in st.warning.call_args.args[0]
    st.info.assert_not_called()


def test_historical_nontrigger_is_scoped_to_snapshot(monkeypatch):
    st = SimpleNamespace(subheader=Mock(), warning=Mock(), info=Mock(), caption=Mock(), write=Mock())
    monkeypatch.setattr(streamlit_app, "st", st)
    streamlit_app._emergency_panel({"asof": "2020-01-01", "emergency": {"active": False}})
    assert "历史快照" in st.warning.call_args.args[0]
    assert "该快照" in st.info.call_args.args[0]
    assert "当前未进入" not in st.info.call_args.args[0]
