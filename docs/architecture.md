# Architecture Overview

QuantSearch is organised as a set of focused Python packages that cooperate through lightweight interfaces. This document describes the major building blocks and how data flows between them.

## Layered structure

- **Configuration (`utils/config.py`)**  
  Loads `config.yaml`, bootstraps project directories, and exposes convenience accessors for downstream modules. Most components accept either raw parameters or a `Config` instance.

- **Data layer (`data/`)**  
  - `ingestion.py` retrieves OHLCV histories (defaulting to S&P 500 constituents), normalises column names, and stores one CSV per symbol under `data/raw/ohlcv/`.  
  - `validation.py` can reconcile recent price bars across multiple providers (Yahoo Finance, Twelve Data, Alpha Vantage, Polygon, Finnhub) and surfaces tolerance breaches.  
  - `DataLoader` and `UniverseFilter` (in `utils/data_loader.py`) provide read-only access to raw/processed data and help curate tradable universes based on quality metrics.

- **Feature engineering (`features/`)**  
  - `base_feature.py` defines the `BaseFeature` abstraction plus a `FeatureRegistry` that supports plug-and-play feature generators.  
  - `FeatureEngine` orchestrates registered features, returning a consolidated DataFrame ready for labelling.  
  - `technical_indicators.py` hosts built-in momentum, trend, and volatility indicators; you can extend this module or create new files and register them.

- **Model training (`models/trainer.py`)**  
  Handles walk-forward splits using expanding or rolling windows, instantiates LightGBM/XGBoost/CatBoost models, logs each run, and saves estimators/feature importances under `models/saved/`.

- **Strategy & portfolio layer (`strategies/`, `backtest/portfolio.py`)**  
  `BaseStrategy` and `MLStrategy` convert signals into target weights while enforcing universe, turnover, and position limits. `PortfolioConstructor` adds shared filters, ranking, and weighting schemes.

- **Execution simulator (`backtest/engine.py`)**  
  Applies T+1 execution, slippage in basis points, commissions, and tracks cash, equity curve, trades, and daily metrics. Designed to work with weight targets emitted by strategies.

- **User interface (`ui/dashboard.py`, `ui/pages/`)**  
  Streamlit dashboard that imports modular pages (`home`, `data_management`, `stock_viewer`, `data_validation`, `feature_engineering`, `model_training`, `strategy_backtesting`, `performance_analysis`, `system_configuration`). Each page reads from shared utilities rather than duplicating logic.

- **Testing (`test_*.py`)**  
  System-level checks (e.g., `test_system_comprehensive.py`) exercise cross-module workflows; QA suites (`test_comprehensive_*.py`) focus on data integrity and UI smoke tests.

## Data flow
1. **Ingest** &mdash; Historical OHLCV data is fetched and persisted to `data/raw/ohlcv/`.
2. **Validate** &mdash; Selected tickers are reconciled against alternative providers; validation summaries are saved alongside raw data.
3. **Feature pipeline** &mdash; `FeatureEngine` loads OHLCV data via `DataLoader`, applies feature generators, and writes results to `data/processed/features/`.
4. **Model training** &mdash; Curated datasets (features + target) are fed into `ModelTrainer` to produce predictions and metadata in `data/processed/predictions/` plus trained models in `models/saved/`.
5. **Strategy simulation** &mdash; Strategies combine predictions with portfolio rules; the backtest engine produces transaction logs and equity curves in `backtest/results/`.
6. **Visualisation** &mdash; The Streamlit UI surfaces status dashboards, validation diagnostics, signal charts, and backtest summaries.

## Extensibility guidelines
- Prefer registering new feature classes through `FeatureRegistry.register("name")` so they become discoverable across the platform.
- Strategy templates can live in `strategies/templates/`; loaders in the UI expect XML or Python definitions that map to `BaseStrategy` subclasses.
- When adding new persistence outputs, piggyback on the existing directory structure managed by `Config.get_path(...)` to keep file locations configurable.
- For additional API providers, extend `DataValidator` with a `_fetch_<provider>()` helper and wire it into the provider map in `config.yaml`.

Understanding these layers makes it straightforward to swap or extend components without rewriting the full stack.
