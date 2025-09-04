"""Streamlit dashboard for Premier League Match Predictor.

Provides a simple form to select teams and match date, calls the FastAPI
`/predict` endpoint if available (http://localhost:8000), and displays the
predicted probabilities. If the API is not reachable, it falls back to a
local prediction using the same feature pipeline and trained model.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

from src.config import load_config
from src.features.rolling import compute_team_rolling_features
from src.features.elo import compute_elo_features
from joblib import load


st.set_page_config(page_title="Premier League Match Predictor", layout="wide")
st.title("Premier League Match Predictor Dashboard")

CONFIG = load_config("configs/runtime.yaml")
CLEAN_CSV = Path(CONFIG["data"]["cleaned_csv"]).resolve()
ARTIFACTS = Path("artifacts").resolve()
MODEL_PATH = ARTIFACTS / "random_forest.joblib"
COLUMNS_META_PATH = ARTIFACTS / "columns.json"
CALIB_PNG = ARTIFACTS / "random_forest_calibration.png"


@st.cache_data(show_spinner=False)
def load_teams() -> list[str]:
    df = pd.read_csv(CLEAN_CSV)
    teams = sorted(set(df["HomeTeam"]).union(set(df["AwayTeam"])))
    return teams


def call_api(home: str, away: str, date_str: str) -> dict:
    try:
        resp = requests.post(
            "http://localhost:8000/predict",
            json={"home_team": home, "away_team": away, "match_date": date_str},
            timeout=5,
        )
        if resp.status_code == 200:
            return resp.json()
        else:
            st.warning(f"API responded with status {resp.status_code}: {resp.text}")
    except Exception as e:
        st.info(f"API not reachable, using local fallback. Details: {e}")
    return {}


def local_fallback(home: str, away: str, date_str: str) -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Trained model not found. Please run `make train` first.")
    clf = load(MODEL_PATH)
    meta_classes = None
    if COLUMNS_META_PATH.exists():
        meta_classes = json.loads(COLUMNS_META_PATH.read_text()).get("classes")

    df = pd.read_csv(CLEAN_CSV)
    df["Date"] = pd.to_datetime(df["Date"])  # ensure datetime
    request_date = pd.to_datetime(date_str)
    hist = df[df["Date"] <= request_date].copy()
    if hist.empty:
        raise ValueError("No historical data before the requested date.")

    feats = compute_team_rolling_features(hist)
    feats = compute_elo_features(feats)
    row = feats[(feats["Date"] == request_date) & (feats["HomeTeam"] == home) & (feats["AwayTeam"] == away)]
    if row.empty:
        pair = feats[(feats["HomeTeam"] == home) & (feats["AwayTeam"] == away) & (feats["Date"] <= request_date)]
        if pair.empty:
            raise ValueError("Fixture not found in historical data.")
        row = pair.sort_values("Date").tail(1)
    X = row.drop(columns=[c for c in ["Date", "HomeTeam", "AwayTeam", "FTR"] if c in row.columns])

    proba = clf.predict_proba(X)[0]
    try:
        classes = list(clf.named_steps["model"].classes_)  # type: ignore[attr-defined]
    except Exception:
        classes = meta_classes or ["H", "D", "A"]
    prob_map = {c: float(p) for c, p in zip(classes, proba)}
    total = sum(prob_map.get(k, 0.0) for k in ("H", "D", "A"))
    if total > 0:
        for k in ("H", "D", "A"):
            prob_map[k] = prob_map.get(k, 0.0) / total
    return {
        "home_win_probability": prob_map.get("H", 0.0),
        "draw_probability": prob_map.get("D", 0.0),
        "away_win_probability": prob_map.get("A", 0.0),
    }


teams = load_teams()
col1, col2, col3 = st.columns(3)
with col1:
    home = st.selectbox("Home team", teams, index=teams.index("Arsenal") if "Arsenal" in teams else 0)
with col2:
    away = st.selectbox("Away team", teams, index=teams.index("Chelsea") if "Chelsea" in teams else 1)
with col3:
    date_str = st.text_input("Match date (YYYY-MM-DD)", value="2023-10-01")

if st.button("Predict"):
    if home == away:
        st.error("Home and away teams must be different.")
    else:
        result = call_api(home, away, date_str)
        if not result:
            try:
                result = local_fallback(home, away, date_str)
            except Exception as e:
                st.error(str(e))
                result = {}

        if result:
            st.subheader("Predicted probabilities")
            c1, c2, c3 = st.columns(3)
            c1.metric("Home win", f"{result['home_win_probability']:.2%}")
            c2.metric("Draw", f"{result['draw_probability']:.2%}")
            c3.metric("Away win", f"{result['away_win_probability']:.2%}")

st.divider()
st.subheader("Model calibration")
if CALIB_PNG.exists():
    st.image(str(CALIB_PNG), caption="Calibration curve (home win)")
else:
    st.info("Calibration plot not found. Run evaluation to generate it.")


