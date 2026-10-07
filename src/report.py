import pandas as pd
import os

from config import DATA_PATH, PREDICTION_PATH


def create_report():

    # Load stock data
    stock_data = pd.read_csv(DATA_PATH)

    # Load prediction
    prediction_data = pd.read_csv(PREDICTION_PATH)

    # Latest stock information
    latest = stock_data.iloc[-1]

    # Latest prediction
    prediction = prediction_data.iloc[-1]

    stock_name = "RELIANCE.NS"

    latest_date = latest["Date"]
    latest_close = float(latest["Close"])

    predicted_direction = prediction["Prediction"]
    up_probability = float(prediction["UP_Probability"])
    down_probability = float(prediction["DOWN_Probability"])
    confidence = float(prediction["Confidence"])

    # Calculate moving averages
    stock_data["MA_20"] = stock_data["Close"].rolling(20).mean()
    stock_data["MA_50"] = stock_data["Close"].rolling(50).mean()

    ma20 = float(stock_data["MA_20"].iloc[-1])
    ma50 = float(stock_data["MA_50"].iloc[-1])

    # Create report
    report = f"""
============================================================
                 STOCK PREDICTION REPORT
============================================================

Stock              : {stock_name}
Latest Date        : {latest_date}
Latest Close       : ₹{latest_close:.2f}

------------------------------------------------------------
PREDICTION
------------------------------------------------------------

Prediction          : {predicted_direction}

UP Probability      : {up_probability * 100:.2f}%
DOWN Probability    : {down_probability * 100:.2f}%
Model Confidence    : {confidence * 100:.2f}%

------------------------------------------------------------
TECHNICAL INDICATORS
------------------------------------------------------------

20-Day Moving Avg   : ₹{ma20:.2f}
50-Day Moving Avg   : ₹{ma50:.2f}

------------------------------------------------------------

Note:
This is a machine-learning prediction and is not guaranteed
to represent the actual future movement of the stock.

============================================================
"""

    print(report)

    # Save report
    output_directory = os.path.dirname(PREDICTION_PATH)

    report_path = os.path.join(
        output_directory,
        "prediction_report.txt"
    )

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(report)

    print("Report saved to:")
    print(report_path)

    return report_path