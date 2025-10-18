"""
Quick sanity check for the validation summary helper used in the Streamlit pages.

Run with: python scripts/test_validation_summary.py
"""

from __future__ import annotations

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover - helper script
    print("Skipping summary smoke test because pandas is not installed:", exc)
    raise SystemExit(0)

from ui.pages.data_validation import summarize_validation


def main() -> None:
    sample_results = pd.DataFrame(
        [
            {"ticker": "AAPL", "is_valid": True, "confidence_score": 98.5},
            {"ticker": "MSFT", "is_valid": True, "confidence_score": 95.2},
            {"ticker": "TSLA", "is_valid": False, "confidence_score": 72.1},
        ]
    )

    summary = summarize_validation(sample_results)
    print("Summary:", summary)


if __name__ == "__main__":
    main()
