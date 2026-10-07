import os
import pandas as pd
import matplotlib.pyplot as plt

from config import DATA_PATH, CHART_PATH


def create_chart():

    # Load stock data
    data = pd.read_csv(DATA_PATH)

    # Make sure Date is datetime
    data["Date"] = pd.to_datetime(data["Date"])

    # Calculate moving averages here
    data["MA_20"] = data["Close"].rolling(window=20).mean()

    plt.figure(figsize=(12, 6))

    # Actual closing price
    plt.plot(
        data["Date"],
        data["Close"],
        label="Close Price"
    )

    # 20-day moving average
    plt.plot(
        data["Date"],
        data["MA_20"],
        label="20-Day Moving Average"
    )

    plt.title("RELIANCE Stock Price")
    plt.xlabel("Date")
    plt.ylabel("Price")

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    # Create output directory if necessary
    os.makedirs(os.path.dirname(CHART_PATH), exist_ok=True)

    plt.savefig(CHART_PATH)
    plt.close()

    print("\nChart saved to:")
    print(CHART_PATH)