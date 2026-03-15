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

    # Absolute date selection for timeframe
    available_min = pd.Timestamp("2000-01-01")
    available_max = pd.Timestamp.today()
    start_date = st.sidebar.date_input("Start date", value=available_min.date(), min_value=available_min.date(), max_value=available_max.date())
    end_date = st.sidebar.date_input("End date", value=available_max.date(), min_value=available_min.date(), max_value=available_max.date())
    if start_date > end_date:
        end_date = start_date

    # Maximum ticker limit
    st.sidebar.markdown("**Max tickers:** 20")
    and_dark = st.sidebar.checkbox("Dark mode", value=False)
    if and_dark:
        st.markdown("<style>body { background-color: #0e1117; color: #f5f5f5; }</style>", unsafe_allow_html=True)

    try:
        tickers_csv = pd.read_csv("yahoo_tickers.txt")
    except Exception:
        st.error("Could not load yahoo_tickers.txt")
        return

    tickers_default = tickers_csv["Symbol"].dropna().astype(str).tolist()
    selected_tickers = st.sidebar.multiselect("Select tickers", tickers_default, default=tickers_default[:6])
    if len(selected_tickers) > 20:
        st.sidebar.error("Choose at most 20 tickers")
        return

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

            # Apply absolute date filter to all computations
            price_df = price_df.loc[pd.to_datetime(start_date):pd.to_datetime(end_date)]
            if price_df.empty:
                st.error("No price data in selected date range")
                return

            # Keep weights aligned to actually loaded tickers (after load_price_dataframe may drop unavailable symbols)
            loaded_tickers = list(price_df.columns)
            if weights is None or weights.sum() == 0:
                weights = normalize_weights(None, loaded_tickers)
            else:
                weights = normalize_weights(weights, loaded_tickers)

            benchmark_returns = None
            try:
                benchmark_series = merge_benchmark(benchmark_symbol, data_dir=data_dir)
                benchmark_series = benchmark_series.loc[pd.to_datetime(start_date):pd.to_datetime(end_date)]
                benchmark_returns = benchmark_series.pct_change().dropna()
            except Exception:
                st.warning("Benchmark not available or missing data")

            port_returns, asset_returns, _ = calculate_portfolio_returns(price_df, weights=weights)

            metrics = risk_metrics.portfolio_report(price_df, weights=weights, benchmark_returns=benchmark_returns, risk_free_rate=risk_free_rate)

            st.subheader("Portfolio performance metrics")
            scalar_metrics = {
                k: float(v)
                for k, v in metrics.items()
                if isinstance(v, (int, float, np.integer, np.floating)) and pd.notnull(v)
            }
            if scalar_metrics:
                portfolio_table = pd.DataFrame.from_dict(scalar_metrics, orient="index", columns=["value"]).reset_index().rename(columns={"index": "metric"})
                st.dataframe(portfolio_table)
                st.download_button("Download portfolio metrics CSV", portfolio_table.to_csv(index=False), "portfolio_metrics.csv", "text/csv")

            st.subheader("Individual securities metrics")
            security_table = pd.DataFrame(
                {
                    "mean_return": asset_returns.mean(),
                    "volatility": asset_returns.std() * np.sqrt(252),
                    "cumulative_return": (1 + asset_returns).prod() - 1,
                }
            )
            st.dataframe(security_table)
            st.download_button("Download securities metrics CSV", security_table.to_csv(index=True), "securities_metrics.csv", "text/csv")

            st.subheader("Correlation matrix")
            correlation = metrics.get("correlation")
            if correlation is not None:
                st.dataframe(correlation)

            st.subheader("Risk contribution")
            risk_contrib = metrics.get("risk_contribution")
            if isinstance(risk_contrib, (pd.Series, pd.DataFrame)):
                st.dataframe(risk_contrib)

            st.subheader("Performance contribution")
            perf_contrib = metrics.get("performance_contribution")
            if isinstance(perf_contrib, (pd.Series, pd.DataFrame)):
                st.dataframe(perf_contrib)

            st.subheader("Performance attribution")
            perf_attr = metrics.get("performance_attribution")
            if isinstance(perf_attr, pd.DataFrame):
                st.dataframe(perf_attr)

            st.subheader("Portfolio attribution summary")
            pa_summary = metrics.get("performance_attribution_summary")
            if isinstance(pa_summary, dict):
                pa_df = pd.DataFrame(list(pa_summary.items()), columns=["item", "value"])
                st.dataframe(pa_df)

            st.subheader("Cumulative portfolio NAV")
            nav = (1 + port_returns).cumprod()
            st.line_chart(nav)

            st.subheader("Cumulative portfolio returns")
            st.line_chart(port_returns.cumsum())

            st.subheader("Rolling volatility (30-day)")
            rolling_vol = port_returns.rolling(30).std() * np.sqrt(252)
            st.line_chart(rolling_vol)

            st.subheader("Drawdown")
            drawdown = risk_metrics.calculate_drawdown_series(nav)
            st.line_chart(drawdown)
            st.metric("Max drawdown", f"{drawdown.min():.2%}")

            st.subheader("Weights")
            weights_df = weights.rename("weight").reset_index().rename(columns={"index": "ticker"})
            st.dataframe(weights_df)

            st.subheader("Allocation (weights)")
            st.bar_chart(weights_df.set_index("ticker"))

            st.subheader("Historical prices")
            st.line_chart(price_df)

        except Exception as e:
            st.error(f"Error during compute: {e}")


if __name__ == "__main__":
    app()
