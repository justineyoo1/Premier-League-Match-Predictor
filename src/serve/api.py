"""FastAPI application for serving predictions.

Loads a trained model pipeline from ``artifacts/`` and, for each request,
rebuilds features from the cleaned historical dataset up to the requested date.
This keeps prediction-time features aligned with training.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ..config import load_config
from ..features.rolling import compute_team_rolling_features
from ..features.elo import compute_elo_features
from joblib import load


app = FastAPI(title="Premier League Match Predictor API")


class PredictionRequest(BaseModel):
    """Schema for prediction requests.

    Replace fields with the actual features required by the model.
    """

    home_team: str
    away_team: str
    match_date: str  # YYYY-MM-DD


class PredictionResponse(BaseModel):
    """Schema for prediction responses."""

    home_win_probability: float
    draw_probability: float
    away_win_probability: float


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


def _load_artifacts(model_type: str = "random_forest"):
    artifacts_dir = Path("artifacts").resolve()
    model_path = artifacts_dir / f"{model_type}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Train the model first.")
    clf = load(model_path)
    cols_meta_path = artifacts_dir / "columns.json"
    classes: Optional[list[str]] = None
    if cols_meta_path.exists():
        meta = json.loads(cols_meta_path.read_text())
        classes = meta.get("classes")
    return clf, classes


def _build_single_row_features(clean_csv: Path, home: str, away: str, date_str: str) -> pd.DataFrame:
    df = pd.read_csv(clean_csv)
    df["Date"] = pd.to_datetime(df["Date"])  # ensure datetime
    request_date = pd.to_datetime(date_str)
    # Use all history up to and including the requested date to compute features
    hist = df[df["Date"] <= request_date].copy()
    if hist.empty:
        raise ValueError("No historical data available before the requested date.")

    feats = compute_team_rolling_features(hist)
    feats = compute_elo_features(feats)

    # Select the match row on the requested date
    row = feats[(feats["Date"] == request_date) & (feats["HomeTeam"] == home) & (feats["AwayTeam"] == away)]
    if row.empty:
        # If not found exactly on that date, try the most recent match between these teams on/before date
        pair = feats[(feats["HomeTeam"] == home) & (feats["AwayTeam"] == away) & (feats["Date"] <= request_date)]
        if pair.empty:
            raise ValueError("Requested fixture not found in historical data.")
        row = pair.sort_values("Date").tail(1)

    # Drop non-feature columns; model pipeline will handle preprocessing
    non_features = {"Date", "HomeTeam", "AwayTeam", "FTR"}
    X = row.drop(columns=[c for c in row.columns if c in non_features])
    return X


_CONFIG = load_config("configs/runtime.yaml")
_CLEAN_CSV = Path(_CONFIG["data"]["cleaned_csv"]).resolve()
_MODEL, _CLASSES = _load_artifacts(model_type="random_forest")


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest) -> PredictionResponse:
    try:
        X = _build_single_row_features(_CLEAN_CSV, req.home_team, req.away_team, req.match_date)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    proba = _MODEL.predict_proba(X)[0]
    try:
        classes = list(_MODEL.named_steps["model"].classes_)  # type: ignore[attr-defined]
    except Exception:
        classes = _CLASSES or ["H", "D", "A"]

    # Map class probs
    prob_map = {c: float(p) for c, p in zip(classes, proba)}
    home_p = prob_map.get("H", 0.0)
    draw_p = prob_map.get("D", 0.0)
    away_p = prob_map.get("A", 0.0)
    total = home_p + draw_p + away_p
    if total > 0:
        home_p, draw_p, away_p = home_p / total, draw_p / total, away_p / total

    return PredictionResponse(
        home_win_probability=home_p, draw_probability=draw_p, away_win_probability=away_p
    )


