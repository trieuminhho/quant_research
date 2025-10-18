# Data Pipeline Guide

This guide explains how market data moves through QuantSearch, from external vendors into processed datasets consumed by features, models, and dashboards.

## 1. Ingestion

| Phase | Module | Notes |
|-------|--------|-------|
| Universe selection | `data/ingestion.py::get_sp500_tickers` | Scrapes the current S&P 500 constituents from Wikipedia (with fallback symbols for offline runs). |
| Download | `DataIngestion.download_all_tickers` | Parallelises Yahoo Finance requests with retry logic, normalises columns, and writes `{ticker}.csv` to `data/raw/ohlcv/`. |
| Metadata | `DataIngestion.save_ticker_data` | Ensures the `date` column is saved as `%Y-%m-%d` strings and logs download counts. |

Key configuration knobs live under `data:` in `config.yaml`:

- `start_date` / `end_date` &mdash; time horizon for historical pulls.
- `update_frequency` &mdash; hints at how often ingestion should run (informational).
- `validation.providers` &mdash; toggles auxiliary vendors for reconciliation checks.

## 2. Validation

`data/validation.py` offers the `DataValidator` class, which orchestrates external comparisons:

1. The validator inspects `config.yaml` to determine which providers are enabled.
2. Every `_fetch_<provider>()` helper returns a standardised OHLCV DataFrame for the recent window (default 10 business days).
3. `validate_price_data()` compares the local CSV against provider quotes, calculates differences, and records consensus/confidence metrics.
4. `validate_ticker_multisource()` wraps `validate_price_data()` and formats a report that the dashboard can render.

Validation results, summaries, and raw pulls are written next to the source CSVs to make audit trails easy to follow.

## 3. Universe curation

`utils/data_loader.UniverseFilter` builds on the ingested data to keep downstream steps focused on liquid, well-covered tickers. The typical call sequence is:

```python
from utils.data_loader import DataLoader, UniverseFilter

loader = DataLoader()
universe = UniverseFilter(loader)
quality = universe.filter_by_data_quality(min_data_points=504)
coverage = universe.filter_by_date_availability("2018-01-01", "2023-12-31")
eligible = sorted(set(quality) & set(coverage))
```

By default, anything stored under `data/raw/ohlcv/` is discoverable. `get_available_tickers()` excludes helper CSVs such as `sp500_tickers.csv`.

## 4. Processed outputs

### Feature matrices
Feature generators write CSV artefacts to `data/processed/features/`. Naming conventions typically follow `<TICKER>_features.csv` or include the feature set label. Each file is indexed by `date` and stores engineered columns ready for modelling.

### Predictions
`ModelTrainer` can persist prediction tables for every symbol it scores. These live under `data/processed/predictions/` and capture predicted returns, confidence intervals, and any auxiliary metrics emitted during training.

### Audit logs
- Download summaries and validation reports accompany the raw data files.
- LightGBM/XGBoost training metadata is stored as JSON alongside pickled models within `models/saved/`.
- Backtest equity curves and trade ledgers are written to `backtest/results/` for replay inside the UI.

## 5. Automation tips

- Schedule ingestion and validation scripts via cron or an orchestrator, and keep an eye on vendor rate limits specified in `config.yaml`.
- Use the `DataLoader.load_multiple_tickers(combine=True)` helper to build consolidated DataFrames that are ready for feature pipelines or exploratory notebooks.
- When adding a new processed dataset, update `Config.get_path(...)` calls instead of hard-coding paths so deployments stay portable.

Following this pipeline ensures the rest of the platform always consumes clean, well-versioned inputs.
