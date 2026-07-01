from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from prediction import PredictionError, Predictor
from utils.logging import configure_logging


log = logging.getLogger(__name__)


def _prompt_value(feature: str) -> Any:
    label = feature.replace("_", " ").title()
    return input(f"{label}: ").strip()


def _print_single_prediction(result: dict[str, Any]) -> None:
    currency = result.get("currency", "USD")
    symbol = "$" if currency == "USD" else ""
    print()
    print("-----------------------------")
    print("Predicted Financial Loss")
    print("-----------------------------")
    print(f"{symbol}{result['predicted_loss']:,.2f} {currency}")
    print()
    print(f"Risk Level: {result['risk_level']}")
    print(f"Confidence Score: {result['confidence_score']:.2f}%")
    interval = result["prediction_interval_95"]
    print(f"95% Prediction Interval: {symbol}{interval['lower']:,.2f} to {symbol}{interval['upper']:,.2f} {currency}")
    print()
    print("Top Factors")
    for item in result.get("top_features", [])[:10]:
        print(f"- {item['label']} ({item['impact']}, {item['contribution']:,.6f})")
    print()
    print("SHAP plot: plots/prediction_shap.png")


def _run_interactive(predictor: Predictor) -> None:
    record = {feature: _prompt_value(feature) for feature in predictor.feature_columns}
    result = predictor.predict_single(record)
    _print_single_prediction(result)


def _run_csv(predictor: Predictor, input_path: Path, output_path: Path) -> None:
    if not input_path.exists():
        raise PredictionError(f"CSV input file does not exist: {input_path}")
    if input_path.suffix.lower() != ".csv":
        raise PredictionError("CSV prediction requires a .csv input file.")

    df = pd.read_csv(input_path)
    predictions = predictor.predict_dataframe(df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_columns = [
        "predicted_loss",
        "currency",
        "risk_level",
        "confidence_score",
        "prediction_interval_lower",
        "prediction_interval_upper",
        "top_feature_1",
        "top_feature_2",
        "top_feature_3",
        "timestamp",
    ]
    feature_columns = [column for column in predictions.columns if column not in prediction_columns]
    export_columns = feature_columns + prediction_columns
    predictions[export_columns].to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict cyber financial loss from trained production artifacts.")
    parser.add_argument("--input", type=Path, help="CSV file containing required model features.")
    parser.add_argument("--output", type=Path, default=Path("outputs/predictions.csv"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--risk-thresholds", type=Path, default=Path("config/risk_thresholds.yaml"))
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure_logging(args.log_level)
    try:
        predictor = Predictor(
            model_dir=args.model_dir,
            risk_thresholds_path=args.risk_thresholds,
            plots_dir=Path("plots"),
        ).load()
        if args.input:
            _run_csv(predictor, args.input, args.output)
        else:
            _run_interactive(predictor)
    except PredictionError as exc:
        log.error("%s", exc)
        raise SystemExit(1) from exc
    except Exception as exc:
        log.exception("Unexpected prediction failure.")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
