from typing import Tuple
import numpy as np
import pandas as pd
import yfinance as yf


def fetch_and_process_returns(
    ticker: str = "SUZB3.SA",
    start: str = "2020-01-01",
    end: str = "2025-12-31"
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Baixa preços ajustados e calcula os log-retornos diários."""
    df = yf.download(ticker, start=start, end=end, auto_adjust=True).dropna()
    
    if isinstance(df["Close"], pd.DataFrame):
        close_prices = df["Close"].iloc[:, 0].values
    else:
        close_prices = df["Close"].values

    log_returns = np.diff(np.log(close_prices), prepend=np.log(close_prices[0]))
    return df, close_prices, log_returns