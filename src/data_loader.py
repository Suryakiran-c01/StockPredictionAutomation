import yfinance as yf

from config import TICKER, PERIOD, DATA_PATH


def download_data():

    print(f"Downloading data for {TICKER}...")

    data = yf.download(
        TICKER,
        period=PERIOD,
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        raise ValueError("No stock data was downloaded.")

    # Handle MultiIndex columns if returned by yfinance
    if hasattr(data.columns, "levels"):
        data.columns = data.columns.get_level_values(0)

    data = data.dropna()

    data.to_csv(DATA_PATH)

    print(f"Data saved to: {DATA_PATH}")
    print(f"Rows downloaded: {len(data)}")

    return data