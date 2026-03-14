import argparse
import json
import os

import pandas as pd

from data_ingest import baseline_ingest, download_prices, load_tickers, append_future_prices, read_config
from portfolio import load_price_dataframe, calculate_portfolio_returns, normalize_weights, weights_from_csv
from risk_metrics import (calculate_annualized_return, calculate_annualized_volatility, calculate_sharpe,
                          calculate_sortino, calculate_var, calculate_cvar, calculate_drawdown_series,
                          calculate_tracking_error, calculate_beta_alpha, calculate_risk_contribution,
                          calculate_effective_diversification)


def main():
    parser = argparse.ArgumentParser(description="RiskApp MVP CLI")
    sub = parser.add_subparsers(dest="command")

    ingest = sub.add_parser("ingest", help="Download and clean data")
    ingest.add_argument("--tickers", default="yahoo_tickers.txt", help="Ticker list file")
    ingest.add_argument("--config", default="config.yaml", help="Config file")

    add = sub.add_parser("add-future", help="Add future price points")
    add.add_argument("ticker", help="Ticker symbol")
    add.add_argument("--future", required=True, help="Future data as JSON, e.g. \"{\\\"2026-12-01\\\":300}\"")
    add.add_argument("--data-dir", default="data")

    metrics = sub.add_parser("metrics", help="Compute portfolio metrics")
    metrics.add_argument("--tickers", nargs="*", help="Tickers to include")
    metrics.add_argument("--weights", help="JSON weights string or CSV path")
    metrics.add_argument("--benchmark", default="SPY")
    metrics.add_argument("--data-dir", default="data")
    metrics.add_argument("--risk-free", type=float, default=0.02)

    dashboard = sub.add_parser("dashboard", help="Run Streamlit dashboard")

    args = parser.parse_args()
    if args.command == "ingest":
        baseline_ingest(tickers_path=args.tickers, config_path=args.config)

    elif args.command == "add-future":
        future_map = json.loads(args.future)
        append_future_prices(args.ticker, future_map, data_dir=args.data_dir)

    elif args.command == "metrics":
        cfg = read_config()
        tickers = args.tickers
        if not tickers:
            tickers = load_tickers("yahoo_tickers.txt")
        price_df = load_price_dataframe(tickers, data_dir=args.data_dir)

        weights = None
        if args.weights:
            if os.path.exists(args.weights):
                weights = weights_from_csv(args.weights)
            else:
                weights = pd.Series(json.loads(args.weights))

        port_returns, asset_returns, w = calculate_portfolio_returns(price_df, weights=weights)
        benchmark = None
        benchmark_returns = None
        bm_path = os.path.join(args.data_dir, f"{args.benchmark}.csv")
        if os.path.exists(bm_path):
            bm_df = pd.read_csv(bm_path, parse_dates=["Date"], index_col="Date")
            bm_series = bm_df["Adj Close"].fillna(method="ffill").dropna()
            benchmark_returns = bm_series.pct_change().dropna()

        print("Annualized return:", calculate_annualized_return(port_returns))
        print("Annualized volatility:", calculate_annualized_volatility(port_returns))
        print("Sharpe:", calculate_sharpe(port_returns, args.risk_free))
        print("Sortino:", calculate_sortino(port_returns, args.risk_free))
        print("VaR 95:", calculate_var(port_returns, 0.05))
        print("CVaR 95:", calculate_cvar(port_returns, 0.05))
        print("Max Drawdown:", calculate_drawdown_series((1 + port_returns).cumprod()).min())

        if benchmark_returns is not None:
            print("Tracking Error:", calculate_tracking_error(port_returns, benchmark_returns))
            beta, alpha = calculate_beta_alpha(port_returns, benchmark_returns, args.risk_free)
            print("Beta:", beta)
            print("Alpha:", alpha)

        cov = asset_returns.cov()
        print("Risk contribution:", calculate_risk_contribution(cov, w))
        print("Effective diversification:", calculate_effective_diversification(w.values, cov))

    elif args.command == "dashboard":
        os.system("streamlit run src/dashboard.py")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
