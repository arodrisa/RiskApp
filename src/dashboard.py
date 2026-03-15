import os
import sys

# Ensure local package path for relative imports in scripts
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

try:
    from src.portfolio import load_price_dataframe, calculate_portfolio_returns, normalize_weights
    from src import risk_metrics
except ImportError:
    from portfolio import load_price_dataframe, calculate_portfolio_returns, normalize_weights
    import risk_metrics


def merge_benchmark(benchmark_symbol, data_dir="data"):
    path = f"{data_dir}/{benchmark_symbol}.csv"
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    if "Adj Close" not in df.columns and "Close" in df.columns:
        df["Adj Close"] = df["Close"]
    return df["Adj Close"].sort_index()


def app():
    try:
        import streamlit as st
    except Exception as e:
        raise RuntimeError("Streamlit is required to run dashboard") from e

    st.set_page_config(page_title="RiskApp MVP Dashboard", layout="wide")
    st.title("RiskApp MVP Dashboard")

    data_dir = st.sidebar.text_input("Data directory", "data")
    benchmark_symbol = st.sidebar.text_input("Benchmark ticker", "SPY")
    risk_free_rate = st.sidebar.number_input("Risk-free rate", value=0.02, step=0.005, min_value=0.0, max_value=1.0)

    try:
        tickers_csv = pd.read_csv("yahoo_tickers.txt")
    except Exception:
        st.error("Could not load yahoo_tickers.txt")
        return

    tickers_default = tickers_csv["Symbol"].dropna().astype(str).tolist()
    selected_tickers = st.sidebar.multiselect("Select tickers", tickers_default, default=tickers_default[:6])

    weights_input = st.sidebar.text_area("Weights (comma-separated or JSON)", value="")
    weights = None
    if weights_input.strip():
        try:
            if "," in weights_input and "{" not in weights_input:
                vals = [float(x.strip()) for x in weights_input.split(",") if x.strip()]
                weights = pd.Series(vals, index=selected_tickers)
            else:
                weights = pd.Series(pd.read_json(weights_input, typ="series"))
        except Exception as e:
            st.sidebar.error(f"Could not parse weights: {e}")

    if st.sidebar.button("Load data and compute"):
        if not selected_tickers:
            st.error("Select at least one ticker")
            return

        try:
            price_df = load_price_dataframe(selected_tickers, data_dir=data_dir)
            port_returns, asset_returns, weights = calculate_portfolio_returns(price_df, weights=weights)

            # Keep weights aligned to actually loaded tickers (after load_price_dataframe may drop unavailable symbols)
            loaded_tickers = list(price_df.columns)
            if weights is None or weights.sum() == 0:
                weights = normalize_weights(None, loaded_tickers)
            else:
                weights = normalize_weights(weights, loaded_tickers)

            benchmark = None
            benchmark_returns = None
            try:
                benchmark_series = merge_benchmark(benchmark_symbol, data_dir=data_dir)
                benchmark_returns = benchmark_series.pct_change().dropna()
            except Exception:
                st.warning("Benchmark not available or missing data")

            metrics = risk_metrics.portfolio_report(price_df, weights=weights, benchmark_returns=benchmark_returns, risk_free_rate=risk_free_rate)

            st.subheader("Portfolio performance metrics")
            scalar_metrics = {
                k: float(v)
                for k, v in metrics.items()
                if isinstance(v, (int, float, np.integer, np.floating)) and pd.notnull(v)
            }
            if scalar_metrics:
                st.dataframe(
                    pd.DataFrame.from_dict(scalar_metrics, orient="index", columns=["value"]).reset_index().rename(columns={"index": "metric"})
                )

            if "weights" in metrics and isinstance(metrics["weights"], (pd.Series, pd.DataFrame)):
                st.subheader("Weights")
                w = metrics["weights"]
                if isinstance(w, pd.Series):
                    w = w.rename("weight").reset_index().rename(columns={"index": "ticker"})
                st.dataframe(w)

            for key in ["correlation", "risk_contribution", "performance_contribution", "performance_attribution"]:
                if key in metrics:
                    st.subheader(key.replace("_", " ").title())
                    st.dataframe(metrics[key])

            # drawdown plot
            st.subheader("Drawdown")
            drawdown = risk_metrics.calculate_drawdown_series((1 + asset_returns.dot(weights)).cumprod())
            st.line_chart(drawdown)
            st.metric("Max drawdown", f"{drawdown.min():.2%}")

            st.subheader("Weights")
            st.dataframe(weights)

            st.subheader("Historical prices")
            st.line_chart(price_df)

        except Exception as e:
            st.error(f"Error during compute: {e}")


if __name__ == "__main__":
    app()
