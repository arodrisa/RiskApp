# RiskApp MVP

## Overview
RiskApp MVP is a CLI and Streamlit dashboard for portfolio risk and performance analysis using historical prices from Yahoo Finance.

## Setup
1. Create a Python virtual environment:
   - `python -m venv venv`
   - `venv\Scripts\activate`
2. Install dependencies:
   - `pip install -r requirements.txt`

## Data ingestion
- Edit `config.yaml` to set `start_date`, `end_date`, `data_dir`, `anomaly_threshold_pct`, and `risk_free_rate`.
- Run:
  - `python -m src.cli ingest` to download and clean all tickers from `yahoo_tickers.txt`.

## Adding future prices
- `python -m src.cli add-future TICKER --future '{"2026-01-01": 295.0, "2026-01-02": 297.2}'`

## Metrics
- `python -m src.cli metrics --benchmark SPY --risk-free 0.02`
- or specify tickers and weights: `python -m src.cli metrics --tickers EI.PA ALV.DE --weights "{\"EI.PA\":0.5, \"ALV.DE\":0.5}"`

## Dashboard
- `python -m src.cli dashboard` (requires `streamlit`)

## Files
- `src/data_ingest.py` ingests and cleans the data.
- `src/portfolio.py` handles weight normalization and returns.
- `src/risk_metrics.py` computes risk and performance measures (Sharpe, Sortino, VaR/CVaR, drawdown, alpha/beta, etc.).
- `src/dashboard.py` streamlit app.
- `.gitignore` includes `data/` to keep local datasets out of repo.
