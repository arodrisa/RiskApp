import numpy as np
import pandas as pd


def calculate_cumulative_return(returns):
    returns = returns.dropna()
    return (1 + returns).prod() - 1


def calculate_annualized_return(returns, periods_per_year=252):
    if isinstance(returns, pd.Series):
        cum = calculate_cumulative_return(returns)
        n = len(returns)
        if n == 0:
            return np.nan
        return (1 + cum) ** (periods_per_year / n) - 1
    else:
        return returns.apply(lambda x: calculate_annualized_return(x, periods_per_year))


def calculate_annualized_volatility(returns, periods_per_year=252):
    std = returns.std()
    return std * np.sqrt(periods_per_year)


def calculate_max_drawdown(nav):
    nav = nav.dropna()
    highwater = nav.cummax()
    drawdown = nav / highwater - 1
    return drawdown.min()


def calculate_drawdown_series(nav):
    nav = nav.dropna()
    highwater = nav.cummax()
    drawdown = nav / highwater - 1
    return drawdown


def calculate_var(returns, alpha=0.05):
    returns = returns.dropna()
    return np.percentile(returns, alpha * 100)


def calculate_cvar(returns, alpha=0.05):
    returns = returns.dropna()
    var = calculate_var(returns, alpha)
    tail = returns[returns <= var]
    return tail.mean() if len(tail) > 0 else np.nan


def calculate_sharpe(returns, risk_free_rate=0.02, periods_per_year=252):
    if isinstance(returns, pd.DataFrame):
        return returns.apply(lambda x: calculate_sharpe(x, risk_free_rate, periods_per_year))

    rf_daily = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess = returns - rf_daily
    ann_excess = excess.mean() * periods_per_year
    ann_vol = returns.std() * np.sqrt(periods_per_year)
    return ann_excess / ann_vol if ann_vol != 0 else np.nan


def calculate_sortino(returns, risk_free_rate=0.02, periods_per_year=252):
    if isinstance(returns, pd.DataFrame):
        return returns.apply(lambda x: calculate_sortino(x, risk_free_rate, periods_per_year))

    rf_daily = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess = returns - rf_daily
    downside = returns[returns < rf_daily]
    if len(downside) == 0:
        return np.nan
    downside_std = np.sqrt((downside ** 2).mean()) * np.sqrt(periods_per_year)
    ann_excess = excess.mean() * periods_per_year
    return ann_excess / downside_std if downside_std != 0 else np.nan


def calculate_beta_alpha(portfolio_returns, benchmark_returns, risk_free_rate=0.02, periods_per_year=252):
    common = portfolio_returns.dropna().index.intersection(benchmark_returns.dropna().index)
    p = portfolio_returns.loc[common]
    b = benchmark_returns.loc[common]

    x = b.values.reshape(-1, 1)
    y = p.values
    beta = np.nan
    alpha = np.nan
    if len(x) > 1:
        beta = np.cov(y, x.squeeze(), ddof=1)[0, 1] / np.var(x, ddof=1)
        rf = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
        alpha = p.mean() - beta * (b.mean() - rf)
    return beta, alpha


def calculate_tracking_error(portfolio_returns, benchmark_returns):
    returns = portfolio_returns.dropna().index.intersection(benchmark_returns.dropna().index)
    diff = portfolio_returns.loc[returns] - benchmark_returns.loc[returns]
    return diff.std() * np.sqrt(252)


def calculate_performance_contribution(asset_returns, weights):
    contribution = (asset_returns.mean() * weights).rename("contribution")
    return contribution


def calculate_risk_contribution(cov_matrix, weights):
    cov_matrix = pd.DataFrame(cov_matrix)

    if isinstance(weights, (pd.Series, pd.DataFrame)):
        weights = pd.Series(weights).reindex(cov_matrix.index).fillna(0.0)
    else:
        weights = pd.Series(np.asarray(weights), index=cov_matrix.index)

    if len(weights) != len(cov_matrix.index):
        raise ValueError("Weights length must match covariance matrix dimension")

    w = weights.values.astype(float)
    cov_vals = cov_matrix.values
    port_vol = np.sqrt(w.dot(cov_vals).dot(w))
    if port_vol == 0 or np.isnan(port_vol):
        return pd.Series(np.nan, index=cov_matrix.index)

    mrc = cov_vals.dot(w) / port_vol
    rc = (w * mrc) / port_vol
    return pd.Series(rc, index=cov_matrix.index)


def calculate_performance_attribution(asset_returns, weights, benchmark_returns=None):
    asset_mean = asset_returns.mean()
    if benchmark_returns is None:
        benchmark_mean = asset_mean.mean()
    else:
        benchmark_mean = benchmark_returns.mean()

    n = len(weights)
    equal_w = 1.0 / n

    selection = weights * (asset_mean - benchmark_mean)
    allocation = (weights - equal_w) * benchmark_mean
    interaction = weights * (asset_mean - benchmark_mean) - selection

    attribution_df = pd.DataFrame({
        "selection": selection,
        "allocation": allocation,
        "interaction": interaction,
        "total": selection + allocation + interaction,
    })
    return attribution_df


def calculate_effective_diversification(weights, cov_matrix):
    cov_matrix = pd.DataFrame(cov_matrix)

    if isinstance(weights, (pd.Series, pd.DataFrame)):
        w = pd.Series(weights).reindex(cov_matrix.index).fillna(0.0).values.astype(float)
    else:
        w = np.asarray(weights, dtype=float)

    if len(w) != len(cov_matrix.index):
        raise ValueError("Weights length must match covariance matrix dimension")

    port_vol = np.sqrt(w.dot(cov_matrix.values).dot(w))
    if port_vol == 0 or np.isnan(port_vol):
        return np.nan
    mrc = cov_matrix.values.dot(w) / port_vol
    dc = (w * mrc) / port_vol
    return 1.0 / np.sum(np.square(dc))


def portfolio_report(price_df, weights=None, benchmark_returns=None, risk_free_rate=0.02):
    asset_returns = price_df.pct_change().dropna()
    weights = weights if weights is not None else pd.Series(np.repeat(1 / len(price_df.columns), len(price_df.columns)), index=price_df.columns)

    weights = weights.reindex(price_df.columns).fillna(0.0)
    if weights.sum() == 0:
        raise ValueError("Weights sum to zero after alignment with price columns")
    weights = weights / weights.sum()

    port_returns = asset_returns.dot(weights)
    metrics = {}
    metrics["annualized_return"] = calculate_annualized_return(port_returns)
    metrics["annualized_volatility"] = calculate_annualized_volatility(port_returns)
    metrics["sharpe"] = calculate_sharpe(port_returns, risk_free_rate)
    metrics["sortino"] = calculate_sortino(port_returns, risk_free_rate)
    metrics["VaR_95"] = calculate_var(port_returns, alpha=0.05)
    metrics["CVaR_95"] = calculate_cvar(port_returns, alpha=0.05)

    nav = (1 + port_returns).cumprod()
    metrics["max_drawdown"] = calculate_max_drawdown(nav)
    if benchmark_returns is not None:
        metrics["tracking_error"] = calculate_tracking_error(port_returns, benchmark_returns)
        beta, alpha = calculate_beta_alpha(port_returns, benchmark_returns, risk_free_rate)
        metrics["beta"] = beta
        metrics["alpha"] = alpha
    else:
        metrics["tracking_error"] = np.nan
        metrics["beta"] = np.nan
        metrics["alpha"] = np.nan

    metrics["correlation"] = asset_returns.corr()
    cov = asset_returns.cov()
    metrics["risk_contribution"] = calculate_risk_contribution(cov, weights)
    metrics["effective_diversification"] = calculate_effective_diversification(weights, cov)
    metrics["performance_contribution"] = calculate_performance_contribution(asset_returns, weights)
    metrics["performance_attribution"] = calculate_performance_attribution(asset_returns, weights, benchmark_returns)

    metrics["weights"] = weights
    metrics["portfolio_return"] = calculate_cumulative_return(port_returns)
    metrics["benchmark_return"] = benchmark_returns.mean() if benchmark_returns is not None else np.nan
    metrics["active_return"] = metrics["portfolio_return"] - metrics["benchmark_return"] if benchmark_returns is not None else np.nan

    return metrics


if __name__ == "__main__":
    print("Risk metrics module")
