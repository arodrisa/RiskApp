import numpy as np
import pandas as pd
import pytest

from src import risk_metrics


def test_cumulative_and_annualized_return():
    returns = pd.Series([0.01, 0.02, 0.0])
    cum = risk_metrics.calculate_cumulative_return(returns)
    ann = risk_metrics.calculate_annualized_return(returns, periods_per_year=252)

    assert pytest.approx(cum, rel=1e-6) == 0.0302
    assert ann > 0


def test_sharpe_sortino():
    returns = pd.Series([0.01, 0.02, -0.005, 0.015])
    shr = risk_metrics.calculate_sharpe(returns, risk_free_rate=0.02)
    srt = risk_metrics.calculate_sortino(returns, risk_free_rate=0.02)

    assert np.isfinite(shr)
    assert np.isfinite(srt)


def test_var_cvar():
    returns = pd.Series([-0.1, -0.05, 0.0, 0.05, 0.1])
    var = risk_metrics.calculate_var(returns, alpha=0.05)
    cvar = risk_metrics.calculate_cvar(returns, alpha=0.05)

    assert pytest.approx(var, rel=1e-6) == -0.09
    assert cvar <= var


def test_beta_alpha_tracking():
    port = pd.Series([0.01, 0.02, -0.01, 0.03])
    bench = pd.Series([0.015, 0.025, -0.005, 0.02])
    beta, alpha = risk_metrics.calculate_beta_alpha(port, bench, risk_free_rate=0.02)

    assert np.isfinite(beta)
    assert np.isfinite(alpha)
    te = risk_metrics.calculate_tracking_error(port, bench)
    assert te >= 0


def test_performance_attribution():
    asset_returns = pd.DataFrame({"A": [0.01, 0.02, -0.005], "B": [0.005, 0.015, 0.01]})
    weights = pd.Series([0.6, 0.4], index=["A", "B"])
    benchmark = pd.Series([0.008, 0.018, 0.002])
    attribution = risk_metrics.calculate_performance_attribution(asset_returns, weights, benchmark_returns=benchmark)

    assert "selection" in attribution.columns
    assert "allocation" in attribution.columns
    assert "total" in attribution.columns
    assert np.isclose(attribution["total"].sum(), attribution["selection"].sum() + attribution["allocation"].sum() + attribution["interaction"].sum())
