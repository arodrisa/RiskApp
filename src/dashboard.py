import pandas as pd
import streamlit as st

from portfolio import load_price_dataframe, calculate_portfolio_returns, normalize_weights
from risk_metrics import (
    calculate_annualized_return,
    calculate_annualized_volatility,
    calculate_sharpe,
    calculate_sortino,
    calculate_var,
    calculate_cvar,
    calculate_drawdown_series,
    calculate_tracking_error,
    calculate_beta_alpha,
    calculate_risk_contribution,
    calculate_effective_diversification,
)


def merge_benchmark(benchmark_symbol, data_dir="data"):
    path = f"{data_dir}/{benchmark_symbol}.csv"
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    if "Adj Close" not in df.columns and "Close" in df.columns:
        df["Adj Close"] = df["Close"]
    return df["Adj Close"].sort_index()


def app():
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

            if weights is None or weights.sum() == 0:
                weights = normalize_weights(None, selected_tickers)
            else:
                weights = normalize_weights(weights, selected_tickers)

            benchmark = None
            benchmark_returns = None
            try:
                benchmark_series = merge_benchmark(benchmark_symbol, data_dir=data_dir)
                benchmark_returns = benchmark_series.pct_change().dropna()
            except Exception:
                st.warning("Benchmark not available or missing data")

            st.subheader("Portfolio metrics")
            st.write("Annualized return:", calculate_annualized_return(port_returns))
            st.write("Annualized volatility:", calculate_annualized_volatility(port_returns))
            st.write("Sharpe ratio:", calculate_sharpe(port_returns, risk_free_rate))
            st.write("Sortino ratio:", calculate_sortino(port_returns, risk_free_rate))
            st.write("VaR 95%:", calculate_var(port_returns, 0.05))
            st.write("CVaR 95%:", calculate_cvar(port_returns, 0.05))

            st.subheader("Drawdown")
            drawdown = calculate_drawdown_series((1 + port_returns).cumprod())
            st.line_chart(drawdown)
            st.metric("Max drawdown", float(drawdown.min()))

            if benchmark_returns is not None:
                st.subheader("Benchmark comparison")
                st.write("Tracking error:", calculate_tracking_error(port_returns, benchmark_returns))
                beta, alpha = calculate_beta_alpha(port_returns, benchmark_returns, risk_free_rate)
                st.write("Beta:", beta)
                st.write("Alpha:", alpha)

            st.subheader("Correlation matrix")
            st.write(asset_returns.corr())

            st.subheader("Risk contribution")
            st.write(calculate_risk_contribution(asset_returns.cov(), weights))

            st.subheader("Effective diversification")
            st.write(calculate_effective_diversification(weights.values, asset_returns.cov()))

            st.subheader("Weights")
            st.write(weights)

            st.subheader("Historical prices")
            st.line_chart(price_df)

        except Exception as e:
            st.error(f"Error during compute: {e}")


if __name__ == "__main__":
    app()
