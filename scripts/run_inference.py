import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger

from src.inference.predict import FraudDetectionPredictor
from src.models.stacking.meta_learner import apply_decision_thresholds


def batch_predict(predictor, test_df: pd.DataFrame, output_path: str):
    """Run predictions on test DataFrame and save results."""
    logger.info("Running batch inference on {} transactions…", len(test_df))

    # Prefer vectorised batch prediction if the predictor supports it;
    # iterrows() is prohibitively slow for 100k+ row DataFrames.
    if hasattr(predictor, "predict_batch"):
        batch_results = predictor.predict_batch(test_df)
        out_df = pd.DataFrame(batch_results)
        out_df.insert(0, "TransactionID",
                      test_df.get("TransactionID", pd.RangeIndex(len(test_df))).values)
    else:
        # Fallback: row-by-row (only for small datasets or debugging)
        logger.warning(
            "predictor.predict_batch() not found — falling back to iterrows(). "
            "This will be slow for large datasets."
        )
        results = []
        for i, row in test_df.iterrows():
            result = predictor.predict_single(row.to_dict())
            results.append({
                "TransactionID": row.get("TransactionID", i),
                "fraud_prob":    result["fraud_prob"],
                "decision":      result["decision"],
                "action":        result["action"],
                "latency_ms":    result["latency_ms"],
                **{f"model_{k}": v for k, v in result["model_probs"].items()},
            })
            if (i + 1) % 1000 == 0:
                logger.info("  Processed {}/{}", i + 1, len(test_df))
        out_df = pd.DataFrame(results)

    out_df.to_csv(output_path, index=False)

    # Summary stats
    decisions = out_df["decision"].value_counts()
    avg_lat   = out_df["latency_ms"].mean() if "latency_ms" in out_df.columns else float("nan")
    logger.info(
        "Batch inference complete.\n"
        "  auto_approve: {:,}  ({:.1%})\n"
        "  flag_review:  {:,}  ({:.1%})\n"
        "  block:        {:,}  ({:.1%})\n"
        "  avg_latency:  {:.1f}ms",
        decisions.get("auto_approve", 0), decisions.get("auto_approve", 0) / len(out_df),
        decisions.get("flag_review",  0), decisions.get("flag_review",  0) / len(out_df),
        decisions.get("block",        0), decisions.get("block",        0) / len(out_df),
        avg_lat,
    )
    logger.info("Predictions saved to {}", output_path)


def main(args):
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    predictor = FraudDetectionPredictor(
        artifacts_dir=cfg["paths"]["checkpoints"],
    )
    predictor.load_all(
        redis_host=cfg["redis"]["host"],
        redis_port=cfg["redis"]["port"],
    )

    if args.single:
        transaction = json.loads(args.single)
        result      = predictor.predict_single(transaction)
        print("\n" + "="*50)
        print(f"Fraud Probability : {result['fraud_prob']:.4f}")
        print(f"Decision          : {result['decision']}")
        print(f"Action            : {result['action']}")
        print(f"Latency           : {result['latency_ms']}ms")
        print("\nModel probabilities:")
        for k, v in result["model_probs"].items():
            print(f"  {k:20s}: {v:.4f}")
        print("="*50)
    else:
        test_df = pd.read_csv(args.input)
        output  = args.output or "predictions.csv"
        batch_predict(predictor, test_df, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run fraud detection inference")
    parser.add_argument("--input",  type=str, help="Path to test CSV")
    parser.add_argument("--output", type=str, help="Output predictions CSV path", default="predictions.csv")
    parser.add_argument("--single", type=str, help="Single transaction as JSON string")
    parser.add_argument("--redis-host", type=str, default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    args = parser.parse_args()
    main(args)
