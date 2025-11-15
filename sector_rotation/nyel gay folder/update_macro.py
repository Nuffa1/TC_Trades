import logging
from pathlib import Path
import pandas as pd
import yfinance as yf

OUTPUT_FILE = Path("macro.csv")


YF_TICKERS = {
    "consumer": "XRT",   
    "labor": "VTI",      
    "inflation": "TIP", 
    "credit": "HYG",   
}

def transform(name, series: pd.Series) -> pd.Series:
    """Simple, safe transformations only."""
    if name == "consumer":
        return series.pct_change(63) * 100  # 3-month momentum

    if name == "labor":
        return series.pct_change(252) * 100  # yearly trend

    if name == "inflation":
        return series.pct_change(252) * 100  # yearly inflation expectations

    if name == "credit":
        return -series.pct_change(63) * 100  # falling HYG = rising credit stress

    return series


logging.basicConfig(
    filename="update_macro.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
console = logging.StreamHandler()
console.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
logging.getLogger().addHandler(console)


def download_series(ticker: str) -> pd.Series:
    """Download a valid Yahoo Finance series."""
    df = yf.download(ticker, period="20y", interval="1d", progress=False)
    
    if df is None or df.empty:
        raise RuntimeError(f"Could not download {ticker}")

    return df["Close"]


def build_macro():
    logging.info("Starting Yahoo macro update...")

    macro = {}

    for name, ticker in YF_TICKERS.items():
        try:
            logging.info(f"Downloading {name} ({ticker})...")
            raw = download_series(ticker)

            transformed = transform(name, raw)
            transformed.name = name

            macro[name] = transformed

        except Exception as e:
            logging.error(f"Failed to download {name}: {e}")

    if not macro:
        logging.critical("No macro series downloaded — aborting.")
        return

    # Combine into a DataFrame
    df = pd.concat(macro.values(), axis=1)

    # Clean & resample for your model
    df = df.ffill().dropna()
    df = df.resample("D").ffill()

    df.to_csv(OUTPUT_FILE)
    logging.info(f"macro.csv updated successfully.")
    print(df.tail())


if __name__ == "__main__":
    try:
        build_macro()
    except Exception:
        logging.exception("Fatal error")
