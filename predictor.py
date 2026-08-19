import joblib
import pandas as pd

MODEL_PATH = "predictive_model.joblib"

def _load_model():
    return joblib.load(MODEL_PATH)

def predict_engagement(caption, posting_hour, day_of_week, media_type,
                       product_type, width, height, has_audio,
                       paid_partnership, carousel_count):
    model = _load_model()

    row = pd.DataFrame([{
        "caption": caption,
        "posting_hour": posting_hour,
        "day_of_week": day_of_week,
        "media_type": media_type,
        "product_type": product_type,
        "width": width,
        "height": height,
        "has_audio": int(has_audio),
        "paid_partnership": int(paid_partnership),
        "carousel_count": carousel_count
    }])

    try:
        pred = float(model.predict(row)[0])
    except Exception:
        # Fall back to a compatible generic prediction path if the saved
        # pipeline expects only its original feature columns.
        pred = float(model.predict(row[["caption", "posting_hour", "day_of_week",
                                        "media_type", "product_type", "width",
                                        "height", "has_audio", "paid_partnership",
                                        "carousel_count"]])[0])

    level = "HIGH" if pred >= 10000 else ("MEDIUM" if pred >= 3000 else "LOW")
    return {"predicted_engagement": max(0, pred), "level": level}
