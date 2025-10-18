# QuantSearch Research Platform

QuantSearch is an end-to-end research environment for systematic equity strategies. The project bundles data acquisition, feature engineering, model training, portfolio simulation, and a Streamlit dashboard into a single workspace so you can iterate on trading ideas without wiring the plumbing each time.

## Why QuantSearch matters
- **S&P 500 data pipeline** &mdash; `data/ingestion.py` pulls and stores OHLCV histories locally, keeping everything in CSV for reproducibility.
- **Data quality guardrails** &mdash; `data/validation.py` can reconcile multiple market data vendors and highlight mismatches before models see the inputs.
- **Pluggable features** &mdash; the `features` package exposes a registry-based system for technical indicators and custom alpha factors.
- **Walk-forward modelling** &mdash; `models/trainer.py` handles expanding-window splits and persists fitted LightGBM/XGBoost/CatBoost models.
- **Realistic backtesting** &mdash; `backtest/engine.py` and `backtest/portfolio.py` simulate execution with slippage, commissions, and detailed metrics.
- **Interactive UI** &mdash; `ui/dashboard.py` stitches together page modules for monitoring data health, exploring signals, and reviewing performance.

## Getting started

### Prerequisites
- Python 3.10+ (project is tested on 3.12)
- Virtual environment manager (`python -m venv`, `conda`, or similar)
- Streamlit for the UI (installed via `requirements.txt`)

### Environment setup
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Configure paths and providers
All runtime options live in `config.yaml`. Update the `paths` block if you need to store data or models elsewhere and review the `data.validation.providers` section to toggle third-party APIs or add keys.

### Acquire historical data
```python
from data.ingestion import DataIngestion

ingest = DataIngestion(
    output_dir="data/raw/ohlcv",
    start_date="2015-01-01"
)
status = ingest.download_all_tickers(max_workers=5)
print(sum(status.values()), "tickers downloaded successfully")
```

### Validate recent data points
```python
from data.validation import DataValidator
from utils.data_loader import DataLoader

loader = DataLoader()
validator = DataValidator()
tickers = loader.get_available_tickers()[:10]
reports = [validator.validate_ticker_multisource(t) for t in tickers]
```

### Engineer features and train models
```python
from features.base_feature import FeatureEngine
from models.trainer import ModelTrainer

engine = FeatureEngine([
    {"name": "sma", "periods": [20, 50]},
    {"name": "rsi", "period": 14},
])
ohlcv = loader.load_ticker_ohlcv("AAPL")
feature_frame = engine.compute_features(ohlcv) if ohlcv is not None else None

trainer = ModelTrainer(model_type="lightgbm")
# Prepare your combined dataset with targets, then:
# trainer.train(X_train, y_train)
```

### Backtest a strategy
```python
from backtest.engine import BacktestEngine

engine = BacktestEngine(initial_capital=500_000)
# Assuming you created a strategy instance and prepared market_data (ticker->DataFrame):
# import pandas as pd
# results = engine.run_backtest(
#     strategy=my_strategy,
#     market_data=market_data,
#     start_date=pd.Timestamp("2020-01-01"),
#     end_date=pd.Timestamp("2023-12-31")
# )
```

### Launch the dashboard
```bash
streamlit run ui/dashboard.py
```

## Repository tour
```
backtest/           Execution engine, portfolio accounting, and results storage
data/               Data ingestion and validation utilities
features/           Feature registry, base class, and indicator implementations
models/             Walk-forward trainer and model persistence helpers
strategies/         Strategy abstractions and templates
ui/                 Streamlit dashboard with modular pages
utils/              Shared helpers, including configuration and data loaders
config.yaml         Centralised runtime configuration
```

## Typical research workflow
1. **Sync market data** using the ingestion module and confirm integrity with validation checks.
2. **Craft feature sets** by registering new indicators or combining existing ones via the feature engine.
3. **Assemble datasets** with labelled targets and run walk-forward modelling to avoid look-ahead bias.
4. **Derive trading rules** (in `strategies/`) that transform signals into target weights.
5. **Simulate execution** through the backtest engine to measure risk-adjusted performance.
6. **Review insights** and diagnostics in the Streamlit UI or export results for further analysis.

## Testing and linting
- System-level smoke tests live in `test_system_*.py`.
- QA-focused regression tests live in `test_comprehensive_*.py`.

Run the full suite with:
```bash
pytest
```

## Contributing ideas
- Capture reusable changes as scripts in `scripts/`.
- Prefer small, composable feature generators and register them via `FeatureRegistry`.
- When extending the UI, add a module to `ui/pages` and wire it into `ui/dashboard.py`.

If you discover mismatches between the docs and the codebase, open an issue or drop a note in the comments for follow-up.
