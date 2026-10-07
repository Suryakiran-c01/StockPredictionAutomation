import os
import joblib

from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from features import create_features
from config import MODEL_PATH, FEATURE_COLUMNS


def train_model():

    print("\n========================================")
    print("        MODEL TRAINING")
    print("========================================")

    # -----------------------------------------
    # 1. Create features
    # -----------------------------------------

    data = create_features()

    print("\nFeature columns:")
    print(FEATURE_COLUMNS)

    # -----------------------------------------
    # 2. Separate X and y
    # -----------------------------------------

    X = data[FEATURE_COLUMNS]
    y = data["Target"]

    print("\nX shape:", X.shape)
    print("y shape:", y.shape)

    # -----------------------------------------
    # 3. Train/test split
    # -----------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        shuffle=False
    )

    print("\nTraining rows:", len(X_train))
    print("Testing rows :", len(X_test))

    # -----------------------------------------
    # 4. Create model
    # -----------------------------------------

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    # -----------------------------------------
    # 5. Train
    # -----------------------------------------

    print("\nTraining model...")

    model.fit(X_train, y_train)

    print("Training completed.")

    # -----------------------------------------
    # 6. Test model
    # -----------------------------------------

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("\n========================================")
    print("          MODEL PERFORMANCE")
    print("========================================")

    print(
        f"\nAccuracy: {accuracy:.2%}"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=["DOWN", "UP"]
        )
    )

    print("Confusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )
    print("\nFeature Importance:")
    print("=" * 40)

    importance = model.feature_importances_

    feature_importance = sorted(
        zip(FEATURE_COLUMNS, importance),
        key=lambda x: x[1],
        reverse=True
    )

    for feature, score in feature_importance:
        print(f"{feature:25} {score:.4f}")

    # -----------------------------------------
    # 7. Save model
    # -----------------------------------------

    os.makedirs(
        os.path.dirname(MODEL_PATH),
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_PATH
    )

    print("\n========================================")
    print("Model saved to:")
    print(MODEL_PATH)
    print("========================================")

    return model


if __name__ == "__main__":
    train_model()