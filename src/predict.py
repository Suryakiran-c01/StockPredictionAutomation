import os
import joblib
import pandas as pd

from features import create_features
from config import MODEL_PATH, PREDICTION_PATH, FEATURE_COLUMNS, OUTPUT_PATH


def predict_next_day():

    print("\n========================================")
    print("        STOCK MARKET PREDICTION")
    print("========================================")

    # -----------------------------------------
    # 1. Create latest features
    # -----------------------------------------

    data = create_features()

    # -----------------------------------------
    # 2. Load trained model
    # -----------------------------------------

    print("\nLoading trained model...")

    model = joblib.load(MODEL_PATH)

    print("Model loaded successfully.")

    # -----------------------------------------
    # 3. Get latest row
    # -----------------------------------------

    latest_data = data.iloc[-1:]

    # -----------------------------------------
    # 4. Select same features used for training
    # -----------------------------------------

    X_latest = latest_data[FEATURE_COLUMNS]

    # -----------------------------------------
    # 5. Make prediction
    # -----------------------------------------

    prediction = model.predict(X_latest)[0]

    probabilities = model.predict_proba(X_latest)[0]

    down_probability = probabilities[0]
    up_probability = probabilities[1]

    # -----------------------------------------
    # 6. Convert prediction
    # -----------------------------------------

    if prediction == 1:
        prediction_text = "UP"
        confidence = up_probability
    else:
        prediction_text = "DOWN"
        confidence = down_probability

    # -----------------------------------------
    # 7. Display result
    # -----------------------------------------

    latest_date = latest_data["Date"].iloc[0]
    latest_close = latest_data["Close"].iloc[0]

    print("\n========================================")
    print("             PREDICTION")
    print("========================================")

    print(f"Latest Date       : {latest_date}")
    print(f"Latest Close      : {latest_close:.2f}")

    print(f"\nPrediction        : {prediction_text}")

    print(f"UP Probability    : {up_probability:.2%}")
    print(f"DOWN Probability  : {down_probability:.2%}")

    print(f"Model Confidence  : {confidence:.2%}")

    print("========================================")

    # -----------------------------------------
    # 8. Save prediction
    # -----------------------------------------

    result = pd.DataFrame({
        "Date": [latest_date],
        "Latest_Close": [latest_close],
        "Prediction": [prediction_text],
        "UP_Probability": [up_probability],
        "DOWN_Probability": [down_probability],
        "Confidence": [confidence]
    })

    result.to_csv(
        PREDICTION_PATH,
        index=False
    )
    
    # Save prediction history
    history_path = os.path.join(
        OUTPUT_PATH,
        "prediction_history.csv"
    )

    if os.path.exists(history_path) and os.path.getsize(history_path) > 0:
        history = pd.read_csv(history_path)
    else:
        history = pd.DataFrame()

# Add the latest prediction
    history = pd.concat([history, result], ignore_index=True)

# Normalize all dates to YYYY-MM-DD
    history["Date"] = pd.to_datetime(
        history["Date"],
        errors="coerce"
    ).dt.strftime("%Y-%m-%d")

# Remove invalid dates
    history = history.dropna(subset=["Date"])

# Keep only the latest prediction for each date
    history = history.drop_duplicates(
        subset=["Date"],
        keep="last"
    )

    history.to_csv(
        history_path,
        index=False
    )

    print("\nPrediction history saved to:")
    print(history_path)

    print("\nPrediction saved to:")
    print(PREDICTION_PATH)

    return result


if __name__ == "__main__":
    predict_next_day()