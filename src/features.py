import pandas as pd

from config import DATA_PATH


def create_features():

    # Load stock data
    data = pd.read_csv(DATA_PATH)

    # Convert Date column
    data["Date"] = pd.to_datetime(data["Date"])

    # Sort by date
    data = data.sort_values("Date").reset_index(drop=True)

    # ------------------------------------------------
    # BASIC FEATURES
    # ------------------------------------------------

    # Previous day's closing price
    data["Previous_Close"] = data["Close"].shift(1)

    # Daily return
    data["Daily_Return"] = (
        data["Close"].pct_change()
    )

    # High - Low difference
    data["High_Low_Difference"] = (
        data["High"] - data["Low"]
    )

    # Open - Close difference
    data["Open_Close_Difference"] = (
        data["Close"] - data["Open"]
    )

    # ------------------------------------------------
    # MOVING AVERAGES
    # ------------------------------------------------

    data["MA5"] = (
        data["Close"]
        .rolling(window=5)
        .mean()
    )

    data["MA10"] = (
        data["Close"]
        .rolling(window=10)
        .mean()
    )

    data["MA20"] = (
        data["Close"]
        .rolling(window=20)
        .mean()
    )

    # ------------------------------------------------
    # MOMENTUM
    # ------------------------------------------------

    data["Momentum"] = (
        data["Close"] - data["Close"].shift(5)
    )

    # ------------------------------------------------
    # VOLATILITY
    # ------------------------------------------------

    data["Volatility"] = (
        data["Daily_Return"]
        .rolling(window=10)
        .std()
    )

    # ------------------------------------------------
    # VOLUME CHANGE
    # ------------------------------------------------

    data["Volume_Change"] = (
        data["Volume"].pct_change()
    )

    # ------------------------------------------------
    # RSI
    # ------------------------------------------------

    delta = data["Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    average_gain = (
        gain.rolling(window=14).mean()
    )

    average_loss = (
        loss.rolling(window=14).mean()
    )

    rs = average_gain / average_loss

    data["RSI"] = 100 - (
        100 / (1 + rs)
    )

    # ------------------------------------------------
    # TARGET
    # ------------------------------------------------

    # Tomorrow's closing price is higher than
    # today's closing price → 1 (UP)
    # Otherwise → 0 (DOWN)

    data["Target"] = (
        data["Close"].shift(-1) > data["Close"]
    ).astype(int)

    # Remove rows containing NaN values
    data = data.replace([float("inf"), float("-inf")], float("nan"))

    data = data.dropna().reset_index(drop=True)

    print("Features created successfully.")
    print("Rows available:", len(data))

    return data