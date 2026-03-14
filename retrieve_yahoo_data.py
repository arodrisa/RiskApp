# download_and_save.py
import yfinance as yf
import pandas as pd
import os
from datetime import datetime




def descargar_y_guardar(tickers, start="2020-01-01", end=None, carpeta="data"):
    # if isinstance(tickers, str):
    #     tickers = [tickers]

    precios = yf.download(tickers, start=start, end=end)["Adj Close"]
    
    # Crear carpeta si no existe
    os.makedirs(carpeta, exist_ok=True)

    # Guardar cada activo individualmente
    for ticker in tickers:
        df_ticker = precios[[ticker]].dropna()
        file_path = os.path.join(carpeta, f"{ticker}.csv")
        df_ticker.to_csv(file_path)
        print(f"✅ Guardado: {file_path}")

    return precios

def read_file(filepath):
    file_dt = pd.read_csv(filepath)
    return file_dt


filename = 'yahoo_tickers.txt'
symbol_data = read_file(filename)
symbols = [value for value in symbol_data['Symbol']]

yf_symbols = yf.Tickers(symbols)
data = yf_symbols.history(period = '10y')
data = descargar_y_guardar(yf.Tickers(symbols))