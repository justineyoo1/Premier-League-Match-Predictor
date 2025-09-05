PYTHON?=python3
VENV?=.venv
ACTIVATE=. $(VENV)/bin/activate

.PHONY: help venv install lint format train eval predict api dashboard test

help:
	@echo "Available targets:"
	@echo "  venv       - Create a virtual environment"
	@echo "  install    - Install dependencies"
	@echo "  lint       - Run linter (ruff)"
	@echo "  format     - Format code (black)"
	@echo "  train      - Train model"
	@echo "  eval       - Evaluate model"
	@echo "  predict    - Run a sample prediction"
	@echo "  ingest     - Aggregate EO CSVs into data/raw/matches.csv"
	@echo "  clean      - Clean raw matches into data/processed/matches_clean.csv"
	@echo "  features   - Build features and time-based splits"
	@echo "  api        - Run FastAPI server"
	@echo "  dashboard  - Run Streamlit dashboard"
	@echo "  test       - Run tests"

venv:
	$(PYTHON) -m venv $(VENV)

install: venv
	$(ACTIVATE) && pip install --upgrade pip && pip install -r requirements.txt

lint:
	$(ACTIVATE) && ruff check src tests

format:
	$(ACTIVATE) && black src tests

train:
	$(ACTIVATE) && $(PYTHON) -m src.models.train --config configs/model/random_forest.yaml

eval:
	$(ACTIVATE) && $(PYTHON) -m src.models.evaluate --config configs/model/random_forest.yaml

predict:
	$(ACTIVATE) && $(PYTHON) -m src.serve.cli --config configs/runtime.yaml

ingest:
	$(ACTIVATE) && $(PYTHON) -m src.ingest.aggregate_eo_csvs --eo_dir EO --out_csv data/raw/matches.csv

clean:
	$(ACTIVATE) && $(PYTHON) -m src.ingest.clean_cli --in_csv data/raw/matches.csv --out_csv data/processed/matches_clean.csv

features:
	$(ACTIVATE) && $(PYTHON) -m src.features.build_features_cli --in_csv data/processed/matches_clean.csv --out_dir data/processed --windows 3 5 10 --valid_frac 0.15 --test_frac 0.15

api:
	$(ACTIVATE) && uvicorn src.serve.api:app --reload --host 0.0.0.0 --port 8000

dashboard:
	$(ACTIVATE) && streamlit run dashboard/app.py

test:
	$(ACTIVATE) && pytest -q --disable-warnings --maxfail=1


