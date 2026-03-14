import os
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf
import yaml


def configure_logger(log_path="data_ingest.log", level=logging.INFO):
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    logging.basicConfig(
        filename=log_path,
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = logging.StreamHandler()
    console.setLevel(level)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)


def read_config(config_path="config.yaml"):
    default = {
        "start_date": "2000-01-01",
        "end_date": None,
        "data_dir": "data",
        "anomaly_threshold_pct": 0.2,
        "risk_free_rate": 0.02,
        "default_weights": [],
        "debug": False,
    }
    if not os.path.exists(config_path):
        return default

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    cfg = default.copy()
    cfg.update({k: data[k] for k in data if k in cfg})
    return cfg


def load_tickers(tickers_path="yahoo_tickers.txt"):
    df = pd.read_csv(tickers_path)
    if "Symbol" not in df.columns:
        raise ValueError("ticker file requires a 'Symbol' column")
    tickers = [s.strip() for s in df["Symbol"].dropna().unique()]
    return tickers


def list_existing_tickers(data_dir="data"):
    p = Path(data_dir)
    if not p.exists():
        return []
    return [f.stem for f in p.glob("*.csv")]


def ensure_data_dir(data_dir="data"):
    Path(data_dir).mkdir(parents=True, exist_ok=True)


def clean_price_series(df, max_jump_pct=0.2):
    df = df.copy()

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")

    df.index = pd.to_datetime(df.index)
    start = df.index.min()
    end = df.index.max()

    # Use daily frequency to allow future date append including weekends
    full_index = pd.date_range(start=start, end=end, freq="D")
    df = df.reindex(full_index)

    # Forward fill empty values; if first day is NaN, backfill then forward.
    df = df.fillna(method="ffill").fillna(method="bfill")

    if "Adj Close" in df.columns:
        df = df[df["Adj Close"] > 0]

    df.index.name = "Date"
    return df


def detect_anomalies(df, max_jump_pct=0.2):
    df = df.copy()
    anomalies = []

    if "Adj Close" not in df.columns:
        return pd.DataFrame(anomalies)

    s = df["Adj Close"].astype(float)

    if s.isna().any():
        for date in s[s.isna()].index:
            anomalies.append({"Date": date, "type": "missing_price", "value": None})

    zero_or_neg = s <= 0
    if zero_or_neg.any():
        for date, value in s[zero_or_neg].items():
            anomalies.append({"Date": date, "type": "nonpositive_price", "value": float(value)})

    returns = s.pct_change().dropna()
    if not returns.empty:
        large_moves = returns.abs() > max_jump_pct
        for date, value in returns[large_moves].items():
            anomalies.append({
                "Date": date,
                "type": "large_jump",
                "return": float(value),
                "threshold": float(max_jump_pct),
            })

    return pd.DataFrame(anomalies)


def save_ticker_csv(df, ticker, data_dir="data"):
    ensure_data_dir(data_dir)
    path = Path(data_dir) / f"{ticker}.csv"
    df.to_csv(path, index=True)
    logging.info(f"Saved {ticker} data to {path}")
    return path


def download_prices(tickers, start="2000-01-01", end=None, data_dir="data", max_jump_pct=0.2):
    if not tickers:
        return {}

    if isinstance(tickers, str):
        tickers = [tickers]

    end_date = end or datetime.today().strftime("%Y-%m-%d")
    logging.info(f"Downloading prices for {len(tickers)} tickers from {start} to {end_date}")

    prices = yf.download(tickers, start=start, end=end_date, group_by="ticker", auto_adjust=False, progress=False)

    results = {}
    if len(tickers) == 1:
        ticker = tickers[0]
        df_out = prices.copy()
        if df_out.empty:
            logging.warning(f"No data downloaded for {ticker}")
            return results

        if "Adj Close" not in df_out.columns and "Close" in df_out.columns:
            df_out["Adj Close"] = df_out["Close"]

        cleaned = clean_price_series(df_out)
        anomalies = detect_anomalies(cleaned, max_jump_pct=max_jump_pct)
        save_ticker_csv(cleaned, ticker, data_dir=data_dir)
        results[ticker] = {"data": cleaned, "anomalies": anomalies}
        return results

    # Multiple tickers result in multiindex columns
    for ticker in tickers:
        try:
            subset = prices[ticker].copy()
            if subset.empty:
                logging.warning(f"No data for {ticker}")
                continue

            if "Adj Close" not in subset.columns and "Close" in subset.columns:
                subset["Adj Close"] = subset["Close"]

            cleaned = clean_price_series(subset)
            anomalies = detect_anomalies(cleaned, max_jump_pct=max_jump_pct)
            save_ticker_csv(cleaned, ticker, data_dir=data_dir)
            results[ticker] = {"data": cleaned, "anomalies": anomalies}
        except Exception as e:
            logging.error(f"Mixer error for {ticker}: {e}")

    return results


def download_benchmark(benchmark="SPY", start="2000-01-01", end=None, data_dir="data", max_jump_pct=0.2):
    """Download benchmark price data and store it as benchmark.csv."""
    result = download_prices([benchmark], start=start, end=end, data_dir=data_dir, max_jump_pct=max_jump_pct)
    return result


def append_future_prices(ticker, future_price_map, data_dir="data"):
    # future_price_map: {"YYYY-MM-DD": price, ...}
    path = Path(data_dir) / f"{ticker}.csv"
    if not path.exists():
        raise FileNotFoundError(f"CSV for {ticker} not found at {path}")

    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    last_date = df.index.max()

    new_records = []
    for raw_date, price in future_price_map.items():
        date = pd.to_datetime(raw_date)
        if date <= last_date:
            raise ValueError(f"Future date {date.date()} must be after last known date ({last_date.date()})")
        new_records.append({"Date": date, "Close": price, "Adj Close": price})

    if not new_records:
        logging.warning("No future price records provided")
        return df

    df_future = pd.DataFrame(new_records).set_index("Date").sort_index()
    df = pd.concat([df, df_future])
    df = clean_price_series(df)
    save_ticker_csv(df, ticker, data_dir=data_dir)
    logging.info(f"Appended future prices for {ticker}: {len(df_future)} rows")
    return df


def baseline_ingest(tickers_path="yahoo_tickers.txt", config_path="config.yaml"):
    cfg = read_config(config_path)
    configure_logger(level=logging.DEBUG if cfg.get("debug") else logging.INFO)
    tickers = load_tickers(tickers_path)

    start = cfg.get("start_date", "2000-01-01")
    end = cfg.get("end_date")
    data_dir = cfg.get("data_dir", "data")
    threshold = cfg.get("anomaly_threshold_pct", 0.2)

    return download_prices(tickers=tickers, start=start, end=end, data_dir=data_dir, max_jump_pct=threshold)


if __name__ == "__main__":
    baseline_ingest()
