# Dashboard Reference

The Streamlit dashboard (`ui/dashboard.py`) exposes the platform’s capabilities to non-developers. Each page module inside `ui/pages/` focuses on a specific workflow. This guide highlights what you can expect on each page and how the pieces interact with the backend code.

## Navigation map

| Page | Module | Status | Focus |
|------|--------|--------|-------|
| Home | `ui/pages/home.py` | Ready | Program overview, quick stats, and shortcuts into common tasks. |
| Data Management | `ui/pages/data_management.py` | Ready | Download, refresh, and inspect S&P 500 OHLCV files stored in `data/raw/ohlcv/`. |
| Stock Data Viewer | `ui/pages/stock_viewer.py` | Ready | Plot OHLCV series, overlay indicators, and pivot between analysis charts and validation diagnostics. |
| Feature Engineering | `ui/pages/feature_engineering.py` | Ready | Explore available feature generators, configure parameters, and run feature computations. |
| Model Training | `ui/pages/model_training.py` | Roadmap | Placeholder that previews planned walk-forward training controls. |
| Strategy Backtesting | `ui/pages/strategy_backtesting.py` | Roadmap | Placeholder outlining upcoming execution and portfolio simulation tooling. |
| Performance Analysis | `ui/pages/performance_analysis.py` | Roadmap | Placeholder describing planned analytics and reporting views. |
| System Configuration | `ui/pages/system_configuration.py` | Ready | Surface `config.yaml` settings, highlight key paths, and summarise costs/limits. |

## Shared utilities

All pages share the same supporting objects:

- `get_config()` from `utils.config` for reading the active configuration.
- `DataLoader` and `UniverseFilter` for retrieving CSV data and curating the tradable universe.
- `FeatureEngine`, `ModelTrainer`, `PortfolioConstructor`, `BacktestEngine`, and `DataValidator` depending on the page’s action.

Streamlit’s caching decorators are used sparingly (see `@st.cache_resource` in the Data Management page) to avoid stale data after downloads or validation runs.

## Running the dashboard

```bash
streamlit run ui/dashboard.py
```

By default the dashboard uses the browser’s local session state to remember page selections (e.g., tickers picked for download). To reset state, rerun the app or use Streamlit’s “Clear cache” menu item.

## Tips for extending the UI

- Add a new page by creating a module under `ui/pages/` with a `show()` function and importing it in `ui/pages/__init__.py`. Wire it into the `page_mapping` in `ui/dashboard.py`.
- Keep heavy computation out of the request cycle; prefer precomputing results via scripts stored in `scripts/` and loading artefacts with `DataLoader`.
- When visualising large tables, rely on pagination or summarised metrics to maintain UI responsiveness.

The dashboard is meant to complement, not replace, notebook-driven exploration. Use it as a control centre for monitoring data health, running controlled experiments, and presenting results to collaborators.
