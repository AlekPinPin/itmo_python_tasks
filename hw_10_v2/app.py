from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_PATH = Path(__file__).with_name("realty_model.joblib")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Файл realty_model.joblib не найден. Сначала запустите train.py."
    )

saved = joblib.load(MODEL_PATH)
model = saved["model"]
features = saved["features"]

app = FastAPI(title="Оценка стоимости квартиры")


class RealtyFeatures(BaseModel):
    lat: float
    lon: float
    total_square: float
    rooms: float
    floor: float


def predict_price(data: RealtyFeatures) -> float:
    """Получает предсказание модели по признакам недвижимости."""
    row = pd.DataFrame(
        [[getattr(data, feature) for feature in features]],
        columns=features
    )
    return float(model.predict(row)[0])


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/predict_get")
def predict_get(
    lat: float,
    lon: float,
    total_square: float,
    rooms: float,
    floor: float
):
    data = RealtyFeatures(
        lat=lat,
        lon=lon,
        total_square=total_square,
        rooms=rooms,
        floor=floor
    )
    return {"predicted_price": predict_price(data)}


@app.post("/predict_post")
def predict_post(data: RealtyFeatures):
    return {"predicted_price": predict_price(data)}