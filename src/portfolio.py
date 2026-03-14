import os
from pathlib import Path

import numpy as np
import pandas as pd


def normalize_weights(weights, tickers):
    if weights is None or len(weights) == 0:
        n = len(tickers)
        return pd.Series(np.repeat(1.0 / n, n), index=tickers)

    if isinstance(weights, (list, tuple, np.ndarray)):
        weights = pd.Series(weights, index=tickers)

    weights = weights.dropna()
    if set(weights.index) != set(tickers):
        missing = set(tickers) - set(weights.index)
        for m in missing:
            weights.loc[m] = 0.0

    weights = weights.reindex(tickers).fillna(0.0)
    total = weights.sum()
    if total == 0:
        raise ValueError("weights sum to zero")
    return weights / total


def load_price_dataframe(tickers, data_dir="data"):
    dfs = []
    for t in tickers:
        path = Path(data_dir) / f"{t}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing price file {path}")
        df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
        if "Adj Close" not in df.columns and "Close" in df.columns:
            df["Adj Close"] = df["Close"]
        dfs.append(df["Adj Close"].rename(t))

    all_prices = pd.concat(dfs, axis=1).sort_index()
    all_prices = all_prices.dropna(how="all")
    return all_prices


def calculate_returns(price_df, kind="daily"):
    if kind == "daily":
        returns = price_df.pct_change().dropna()
    elif kind == "log":
        returns = np.log(price_df / price_df.shift(1)).dropna()
    else:
        raise ValueError("Unsupported returns kind")
    return returns


def calculate_portfolio_returns(price_df, weights=None, use_log=False):
    tickers = list(price_df.columns)
    weights = normalize_weights(weights, tickers)
    asset_returns = calculate_returns(price_df, kind="log" if use_log else "daily")
    port_returns = asset_returns.dot(weights)
    return port_returns, asset_returns, weights


def weights_from_csv(path):
    df = pd.read_csv(path)
    if "Symbol" not in df.columns or "Weight" not in df.columns:
        raise ValueError("weights CSV must have Symbol and Weight columns")
    series = pd.Series(df["Weight"].values, index=df["Symbol"].astype(str).values)
    return series


if __name__ == "__main__":
    print("Portfolio helper module")
