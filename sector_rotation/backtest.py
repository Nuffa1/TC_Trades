
from pathlib import Path
import logging
from typing import Iterable, List

import pandas as pd
import yfinance as yf

from sector import (
    run_pipeline,
    SectorMapper,
)

MACRO_FILE = Path("macro.csv")
EQUITY_OUTPUT = Path("equity_curve.csv")
REGIME_OUTPUT = Path("regimes.csv")
TRADE_LOG_OUTPUT = Path("trade_log.csv")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def gather_sector_tickers(mapper: SectorMapper) -> List[str]:
    """Collect unique tickers referenced by the sector mapper."""
    tickers: Iterable[str] = (
        ticker
        for allocation in mapper.mapping.values()
        for ticker in allocation
    )
    return sorted(set(tickers))


def download_sector_prices(tickers: List[str]) -> pd.DataFrame:
    """Download price history for the provided sector tickers."""
    if not tickers:
        raise ValueError("No tickers provided for price download.")

    logging.info("Downloading sector price history from Yahoo Finance...")
    data = yf.download(
        tickers,
        period="20y",
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    if data.empty:
        raise RuntimeError("Failed to download price data.")

    # MultiIndex columns -> select Adj Close, else assume already single level.
    if isinstance(data.columns, pd.MultiIndex):
        level0 = data.columns.get_level_values(0)
        if "Adj Close" in level0:
            data = data.xs("Adj Close", axis=1)
        elif "Close" in level0:
            data = data.xs("Close", axis=1)
        else:
            raise ValueError("Downloaded data missing Close/Adj Close levels.")
    data = data.ffill().dropna(how="all")
    logging.info("Price history shape: %s", data.shape)
    return data


def load_macro_data() -> pd.DataFrame:
    if not MACRO_FILE.exists():
        raise FileNotFoundError(f"{MACRO_FILE} not found.")
    return pd.read_csv(MACRO_FILE, index_col=0, parse_dates=True).sort_index()


def run_backtest():
    macro_df = load_macro_data()
    mapper = SectorMapper()
    tickers = gather_sector_tickers(mapper)
    prices = download_sector_prices(tickers)

    # Align price history to macro date range and fill forward daily.
    prices = prices.resample("D").ffill()
    prices = prices.loc[macro_df.index.min(): macro_df.index.max()]
    prices = prices.dropna(how="all")

    equity_curve, regime_series, trade_log = run_pipeline(macro_df, prices)

    EQUITY_OUTPUT.write_text(equity_curve.to_csv())
    REGIME_OUTPUT.write_text(regime_series.to_frame(name="regime").to_csv())
    
    # Save trade log, handle empty DataFrame
    if not trade_log.empty:
        TRADE_LOG_OUTPUT.write_text(trade_log.to_csv())
    else:
        TRADE_LOG_OUTPUT.write_text("No trades executed.\n")

    logging.info("Backtest complete. Results saved to %s, %s, and %s", EQUITY_OUTPUT, REGIME_OUTPUT, TRADE_LOG_OUTPUT)
    print("\n=== EQUITY CURVE (last 10) ===")
    print(equity_curve.tail(10))
    print("\n=== TRADES EXECUTED ===")
    if not trade_log.empty:
        print(trade_log.to_string())
    else:
        print("No trades executed.")


if __name__ == "__main__":
    run_backtest()
