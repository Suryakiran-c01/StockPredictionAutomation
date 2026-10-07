import os


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)


DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "stock_data.csv"
)


MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "stock_model.pkl"
)


OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "output"
)


PREDICTION_PATH = os.path.join(
    OUTPUT_PATH,
    "prediction.csv"
)


CHART_PATH = os.path.join(
    OUTPUT_PATH,
    "stock_chart.png"
)


TICKER = "RELIANCE.NS"

PERIOD = "5y"

FEATURE_COLUMNS = [
    "Previous_Close",
    "Daily_Return",
    "High_Low_Difference",
    "Open_Close_Difference",
    "MA5",
    "MA10",
    "MA20",
    "Momentum",
    "Volatility",
    "Volume_Change",
    "RSI"
]