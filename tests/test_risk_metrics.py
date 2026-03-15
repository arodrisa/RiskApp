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


def test_risk_contribution_alignment():
    asset_returns = pd.DataFrame({"A": [0.01, 0.02, -0.005], "B": [0.005, 0.015, 0.01]})
    cov = asset_returns.cov()
    weights = pd.Series([0.5, 0.5], index=["A", "B"])
    rc = risk_metrics.calculate_risk_contribution(cov, weights)
    assert list(rc.index) == ["A", "B"]
    assert np.isclose(rc.sum(), 1.0, atol=1e-6) or np.isfinite(rc.sum())


def test_portfolio_report_includes_weights():
    price_df = pd.DataFrame({"A": [100, 101, 102], "B": [200, 202, 204]}, index=pd.date_range("2023-01-01", periods=3))
    weights = pd.Series([0.5, 0.5], index=["A", "B"])
    metrics = risk_metrics.portfolio_report(price_df, weights=weights)
    assert "weights" in metrics
    assert isinstance(metrics["weights"], pd.Series)
    assert np.isclose(metrics["weights"].sum(), 1.0)


def test_performance_attribution():
    asset_returns = pd.DataFrame({"A": [0.01, 0.02, -0.005], "B": [0.005, 0.015, 0.01]})
    weights = pd.Series([0.6, 0.4], index=["A", "B"])
    benchmark = pd.Series([0.008, 0.018, 0.002])
    attribution = risk_metrics.calculate_performance_attribution(asset_returns, weights, benchmark_returns=benchmark)

    assert "selection" in attribution.columns
    assert "allocation" in attribution.columns
    assert "total" in attribution.columns
    assert np.isclose(attribution["total"].sum(), attribution["selection"].sum() + attribution["allocation"].sum() + attribution["interaction"].sum())


def test_performance_attribution_summary():
    asset_returns = pd.DataFrame({"A": [0.01, 0.02, -0.005], "B": [0.005, 0.015, 0.01]})
    weights = pd.Series([0.6, 0.4], index=["A", "B"])
    benchmark = pd.Series([0.008, 0.018, 0.002])
    summary = risk_metrics.calculate_performance_attribution_summary(asset_returns, weights, benchmark_returns=benchmark)

    assert isinstance(summary, dict)
    assert set(summary.keys()) == {"allocation", "selection", "interaction", "total"}
    assert np.isclose(summary["total"], summary["allocation"] + summary["selection"] + summary["interaction"])
