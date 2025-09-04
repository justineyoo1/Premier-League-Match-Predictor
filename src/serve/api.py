"""FastAPI application for serving predictions.

Provides a minimal stub endpoint `/predict` that returns a placeholder
prediction. Replace with real feature building and model inference.
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(title="Premier League Match Predictor API")


class PredictionRequest(BaseModel):
    """Schema for prediction requests.

    Replace fields with the actual features required by the model.
    """

    home_team: str
    away_team: str
    match_date: str


class PredictionResponse(BaseModel):
    """Schema for prediction responses."""

    home_win_probability: float
    draw_probability: float
    away_win_probability: float


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(_: PredictionRequest) -> PredictionResponse:
    """Return a placeholder prediction response."""
    # Placeholder probabilities; replace with model inference
    return PredictionResponse(
        home_win_probability=0.33, draw_probability=0.34, away_win_probability=0.33
    )


