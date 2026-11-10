import json

import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from zhiku.cli import app
from zhiku.strategies.research import load_bars, run_research


def inputs(tmp_path):
    dates = pd.bdate_range('2024-01-02', periods=100)
    rows = [{'date': d.date().isoformat(), 'symbol': s, 'open': 10+i/100+j,
             'close': 10+i/100+j, 'tradable': 1, 'limit_up': 0, 'limit_down': 0}
            for i, d in enumerate(dates) for j, s in enumerate(['A', 'B'])]
    bars = tmp_path/'bars.csv'
    pd.DataFrame(rows).to_csv(bars, index=False)
    spec = tmp_path/'spec.yaml'
    spec.write_text(yaml.safe_dump({'universe': ['A', 'B'], 'universe_known_at': '2024-01-01',
                                   'lookback': 5, 'rebalance_every': 10}), encoding='utf-8')
    return spec, bars, dates


def test_reproducible_research_only_and_holdout_independent(tmp_path):
    spec, bars, dates = inputs(tmp_path)
    split = dates[30].date()
    first = run_research(spec, bars, split, tmp_path/'reports')
    payload = json.loads(first.read_text())
    assert first == run_research(spec, bars, split, tmp_path/'reports')
    assert payload['acceptance']['production_eligible'] is False
    assert payload['status'] == 'RESEARCH_ONLY'
    dev = payload['periods']['development']
    test = payload['periods']['holdout']['strategies']['momentum']['backtest']
    assert test['nav'][split.isoformat()] == 100000
    assert test['trades'][0]['date'] > split.isoformat()
    df = pd.read_csv(bars)
    df.loc[df.date >= split.isoformat(), ['open', 'close']] *= 2
    df.to_csv(bars, index=False)
    second = run_research(spec, bars, split, tmp_path/'reports')
    assert second != first
    assert json.loads(second.read_text())['periods']['development'] == dev


@pytest.mark.parametrize('issue', ['duplicate', 'unknown', 'bad_mask', 'inf', 'negative'])
def test_bars_reject_unsafe_inputs(tmp_path, issue):
    spec, bars, dates = inputs(tmp_path)
    df = pd.read_csv(bars)
    if issue == 'duplicate':
        df = pd.concat([df, df.iloc[:1]])
    elif issue == 'unknown':
        df.loc[0, 'symbol'] = 'C'
    elif issue == 'bad_mask':
        df.loc[0, 'tradable'] = 2
    elif issue == 'inf':
        df.loc[0, 'open'] = np.inf
    else:
        df.loc[0, 'close'] = -1
    df.to_csv(bars, index=False)
    with pytest.raises(ValueError):
        load_bars(bars, ['A', 'B'])


def test_split_and_universe_knowledge_validated(tmp_path):
    spec, bars, dates = inputs(tmp_path)
    with pytest.raises(ValueError, match='interior'):
        run_research(spec, bars, dates[-1].date(), tmp_path/'out')
    raw = yaml.safe_load(spec.read_text())
    raw['universe_known_at'] = '2024-02-01'
    spec.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match='universe_known_at'):
        run_research(spec, bars, dates[30].date(), tmp_path/'out')


def test_cli_writes_only_research_and_no_network(tmp_path, monkeypatch):
    spec, bars, dates = inputs(tmp_path)
    import socket
    def forbidden(*args, **kwargs):
        raise AssertionError('network forbidden')
    monkeypatch.setattr(socket, 'create_connection', forbidden)
    result = CliRunner().invoke(app, ['--provider', 'demo', '--kb', str(tmp_path/'kb'),
        'research', 'etf', '--spec', str(spec), '--bars', str(bars),
        '--split', dates[30].date().isoformat(), '--out', str(tmp_path/'out')])
    assert result.exit_code == 0, result.output
    assert 'RESEARCH_ONLY' in result.output
    assert len(list((tmp_path/'out').glob('*.json'))) == 1
    assert not list((tmp_path/'kb').rglob('*manifest*'))


def test_artifact_tamper_rejected_without_overwrite(tmp_path):
    spec, bars, dates = inputs(tmp_path)
    dest = run_research(spec, bars, dates[30].date(), tmp_path/'out')
    dest.write_text('tampered')
    with pytest.raises(ValueError, match='content hash'):
        run_research(spec, bars, dates[30].date(), tmp_path/'out')
    assert dest.read_text() == 'tampered'


def test_provenance_hashes_exact_parsed_snapshot(tmp_path, monkeypatch):
    from hashlib import sha256
    import zhiku.strategies.research as module
    spec, bars, dates = inputs(tmp_path)
    original_spec, original_bars = spec.read_bytes(), bars.read_bytes()
    original_schedule = module.schedule_decisions
    def mutate_after_parse(*args, **kwargs):
        spec.write_text('changed after parse')
        bars.write_text('changed after parse')
        return original_schedule(*args, **kwargs)
    monkeypatch.setattr(module, 'schedule_decisions', mutate_after_parse)
    dest = run_research(spec, bars, dates[30].date(), tmp_path/'out')
    result = json.loads(dest.read_text())
    assert result['provenance']['spec_sha256'] == sha256(original_spec).hexdigest()
    assert result['provenance']['bars_sha256'] == sha256(original_bars).hexdigest()
