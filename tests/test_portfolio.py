import pandas as pd
import pytest

from src import portfolio


def test_normalize_weights_equal():
    w = portfolio.normalize_weights(None, ["A", "B", "C"])
    assert all(w == 1 / 3)


def test_normalize_weights_custom():
    w = portfolio.normalize_weights([1, 2], ["A", "B"])
    assert pytest.approx(w["A"] + w["B"]) == 1.0
    assert pytest.approx(w["A"]) == 1 / 3


def test_calculate_portfolio_returns(tmp_path):
    df = pd.DataFrame({"A": [100, 110, 121], "B": [100, 105, 110.25]}, index=pd.date_range("2021-01-01", periods=3))
    port_returns, asset_returns, weights = portfolio.calculate_portfolio_returns(df, weights=[0.5, 0.5])
    assert len(port_returns) == 2
    assert not asset_returns.isna().any().any()
    assert pytest.approx(weights.sum()) == 1.0
