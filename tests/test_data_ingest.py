import os
from pathlib import Path

import pytest
import pandas as pd

from src import data_ingest


def test_read_config_defaults(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("start_date: '2005-01-01'\nend_date: '2005-12-31'\ndata_dir: 'data'\n")
    cfg = data_ingest.read_config(str(config_file))

    assert cfg["start_date"] == "2005-01-01"
    assert cfg["end_date"] == "2005-12-31"
    assert cfg["data_dir"] == "data"


def test_load_tickers(tmp_path):
    tickers_file = tmp_path / "yahoo_tickers.txt"
    tickers_file.write_text("Symbol,Name\nAAPL,Apple\nMSFT,Microsoft\n")
    tickers = data_ingest.load_tickers(str(tickers_file))

    assert tickers == ["AAPL", "MSFT"]


def test_clean_price_series_and_detect_anomalies(tmp_path):
    df = pd.DataFrame({
        "Date": ["2021-01-01", "2021-01-04", "2021-01-05"],
        "Adj Close": [100.0, 150.0, 90.0],
    })
    cleaned = data_ingest.clean_price_series(df, max_jump_pct=0.5)

    assert "2021-01-01" in cleaned.index.astype(str).tolist()
    assert "2021-01-04" in cleaned.index.astype(str).tolist()

    anomalies = data_ingest.detect_anomalies(cleaned, max_jump_pct=0.3)
    assert not anomalies.empty
    assert (anomalies["type"] == "large_jump").any()


def test_append_future_prices(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    ticker = "TEST"
    csv_path = data_dir / f"{ticker}.csv"
    pd.DataFrame({"Date": ["2021-01-01"], "Adj Close": [100.0]}).to_csv(csv_path, index=False)

    result = data_ingest.append_future_prices(ticker, {"2021-01-02": 102.0, "2021-01-03": 105.0}, data_dir=str(data_dir))

    assert len(result) >= 3
    assert pd.Timestamp("2021-01-03") in result.index

    with pytest.raises(ValueError):
        data_ingest.append_future_prices(ticker, {"2021-01-01": 110.0}, data_dir=str(data_dir))


def test_download_benchmark_monkeypatch(monkeypatch):
    called = {}

    def fake_download_prices(tickers, start="2000-01-01", end=None, data_dir="data", max_jump_pct=0.2):
        called['tickers'] = tickers
        return {tickers[0]: {'data': pd.DataFrame({}), 'anomalies': pd.DataFrame({})}}

    monkeypatch.setattr(data_ingest, 'download_prices', fake_download_prices)
    out = data_ingest.download_benchmark('SPY', start='2000-01-01', end='2000-01-10', data_dir='data')

    assert called['tickers'] == ['SPY']
    assert 'SPY' in out
