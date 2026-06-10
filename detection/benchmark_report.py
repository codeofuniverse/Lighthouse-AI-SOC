"""Standalone benchmark report script for Phase 3 Detection Engine.

Run with:
    python -m detection.benchmark_report [--csv PATH] [--model-dir PATH]

Actions:
1. Load both saved models from detection/models/
2. Accept a labeled CSV (--csv) or use synthetic data
3. Re-run ModelBenchmark.compare() on fresh data
4. Output a new model_benchmark.json
5. Print the per-attack-type ASCII comparison table
6. Warn if winning model changed vs the previous benchmark run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("benchmark_report")


def _load_models(model_dir: Path) -> tuple[Any, Any]:
    """Load XGBoost and LightGBM models from disk.

    Args:
        model_dir: Directory containing saved models.

    Returns:
        Tuple of (xgb_model, lgb_model). Raises RuntimeError if XGBoost missing.
    """
    import pickle

    import lightgbm as lgb
    import xgboost as xgb

    xgb_path = model_dir / "xgb_model.json"
    if not xgb_path.exists():
        raise RuntimeError(f"XGBoost model not found: {xgb_path}")

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(str(xgb_path))
    logger.info("Loaded XGBoost model from %s", xgb_path)

    lgb_model = None
    lgb_pkl = model_dir / "lgbm_model.pkl"
    lgb_txt = model_dir / "lgb_model.txt"
    if lgb_pkl.exists():
        with open(lgb_pkl, "rb") as f:
            lgb_model = pickle.load(f)
        logger.info("Loaded LightGBM model (pkl) from %s", lgb_pkl)
    elif lgb_txt.exists():
        lgb_model = lgb.Booster(model_file=str(lgb_txt))
        logger.info("Loaded LightGBM model (txt) from %s", lgb_txt)
    else:
        logger.warning("No LightGBM model found — benchmark will use XGBoost only")

    return xgb_model, lgb_model


def _load_data(csv_path: str | None, model_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load benchmark data from CSV or fall back to synthetic generation.

    The CSV must have the same engineered feature columns as the model was
    trained on, plus a ``label`` column (int: 0/1/2) and an ``attack_type``
    column (str from ATTACK_TYPES).

    Args:
        csv_path: Optional path to labeled CSV.
        model_dir: Directory with model artifacts (used to load scaler).

    Returns:
        Tuple of (X, y, attack_type_labels).
    """
    import pickle

    from detection.ml_classifier import ATTACK_TYPES, MLClassifier

    classifier = MLClassifier(str(model_dir))
    # Try to load scaler from saved state
    scaler_path = model_dir / "scaler.pkl"
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            classifier.scaler = pickle.load(f)
        logger.info("Loaded scaler from %s", scaler_path)

    if csv_path:
        logger.info("Loading labeled data from %s", csv_path)
        df = pd.read_csv(csv_path)
        if "label" not in df.columns:
            raise ValueError("CSV must contain a 'label' column (0=benign, 1=suspicious, 2=malicious)")
        y = df["label"].values.astype(int)
        if "attack_type" in df.columns:
            atl = df["attack_type"].tolist()
        else:
            # Derive attack_type from label
            atl = [
                "brute_force" if lbl == 2 else ("port_scan" if lbl == 1 else "benign")
                for lbl in y
            ]
        feature_cols = [c for c in df.columns if c not in ("label", "attack_type")]
        X = df[feature_cols].values.astype(float)
    else:
        logger.info("No CSV provided — generating synthetic benchmark data")
        n = 200
        rng = np.random.default_rng(42)
        rows = []
        labels = []
        atl = []
        for i in range(n):
            lbl = i % 3
            row = {
                "rule_level": 8 if lbl == 2 else (6 if lbl == 1 else 2),
                "src_port": 22,
                "session_event_count": 25 if lbl == 2 else (10 if lbl == 1 else 1),
                "session_duration_seconds": 30 if lbl == 2 else (60 if lbl == 1 else 300),
                "geoip": {"is_tor": lbl == 2, "is_vpn": False},
                "threat_intel": {
                    "abuse_score": 90 if lbl == 2 else (50 if lbl == 1 else 5),
                    "is_known_attacker": lbl == 2,
                },
                "mitre_techniques": [{"technique_id": "T1110"}] if lbl > 0 else [],
                "rule_groups": ["sshd"] if lbl > 0 else ["system"],
                "asset_criticality": "high" if lbl == 2 else "medium",
                "protocol": "ssh",
            }
            rows.append(row)
            labels.append(lbl)
            at_lbl = ["benign", "port_scan", "brute_force"][lbl]
            atl.append(at_lbl)

        df_synth = pd.DataFrame(rows)
        from detection.ml_classifier import MLClassifier as MC

        tmp = MC(str(model_dir))
        feat_df = tmp._engineer_features(df_synth)
        try:
            from sklearn.utils.validation import check_is_fitted

            check_is_fitted(classifier.scaler)
            X = classifier._preprocess_features(feat_df, fit=False)
        except Exception:
            X = tmp._preprocess_features(feat_df, fit=True)
        y = np.array(labels, dtype=int)

    return np.asarray(X, dtype=float), np.asarray(y, dtype=int), atl


def _detect_winner_changes(
    old_bench: dict[str, Any], new_bench: dict[str, Any]
) -> list[str]:
    """Compare two benchmarks and return warning messages for changed winners.

    Args:
        old_bench: Previous model_benchmark.json content.
        new_bench: Freshly computed benchmark.

    Returns:
        List of warning message strings (empty if no changes).
    """
    warnings: list[str] = []
    old_per = old_bench.get("per_attack_type", {})
    new_per = new_bench.get("per_attack_type", {})

    for at in set(list(old_per.keys()) + list(new_per.keys())):
        old_w = old_per.get(at, {}).get("winner", "xgb")
        new_w = new_per.get(at, {}).get("winner", "xgb")
        if old_w != new_w:
            warnings.append(
                f"WARNING: best model for '{at}' changed from '{old_w}' to '{new_w}'"
                " — consider retraining or manually reviewing"
            )

    return warnings


def run_benchmark(
    model_dir: Path,
    csv_path: str | None,
    output_path: Path,
) -> dict[str, Any]:
    """Run the full benchmark and produce output.

    Args:
        model_dir: Directory containing saved models.
        csv_path: Optional path to labeled CSV data.
        output_path: Where to write the new model_benchmark.json.

    Returns:
        New benchmark dict.
    """
    from detection.ml_classifier import ModelBenchmark

    # Load previous benchmark if it exists
    old_bench: dict[str, Any] = {}
    if output_path.exists():
        try:
            with open(output_path) as f:
                old_bench = json.load(f)
            logger.info("Loaded previous benchmark from %s (trained_at=%s)", output_path, old_bench.get("trained_at"))
        except Exception as exc:
            logger.warning("Could not load previous benchmark: %s", exc)

    # Load models
    xgb_model, lgb_model = _load_models(model_dir)

    # Load data
    X, y, atl = _load_data(csv_path, model_dir)
    logger.info("Benchmark data: %d samples, %d features", X.shape[0], X.shape[1])

    # Create a stub LightGBM wrapper if needed (Booster doesn't have predict_proba)
    import lightgbm as lgb_lib

    class _BoosterWrapper:
        """Thin wrapper giving Booster a predict_proba interface."""

        def __init__(self, booster: Any) -> None:
            self._b = booster

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            raw = np.asarray(self._b.predict(X), dtype=float)
            if raw.ndim == 1:
                return np.column_stack([1 - raw, raw])
            return raw

        def predict(self, X: np.ndarray) -> np.ndarray:
            raw = np.asarray(self._b.predict(X), dtype=float)
            if raw.ndim == 2:
                return np.argmax(raw, axis=1)
            return (raw > 0.5).astype(int)

    if isinstance(lgb_model, lgb_lib.Booster):
        lgb_model = _BoosterWrapper(lgb_model)

    if lgb_model is None:
        # Create a dummy model that always returns XGBoost results
        class _DummyLGB:
            def predict_proba(self, X: np.ndarray) -> np.ndarray:
                return xgb_model.predict_proba(X)

        lgb_model = _DummyLGB()
        logger.warning("Using XGBoost as LightGBM placeholder for benchmark")

    # Run comparison
    print(f"\n{'='*60}")
    print(f"  Model Benchmark Report — {date.today().isoformat()}")
    print(f"{'='*60}\n")

    new_bench = ModelBenchmark.compare(xgb_model, lgb_model, X, y, atl)

    # Print overall summary
    print(f"\nOverall XGBoost F1: {new_bench['xgb_overall_f1']:.4f}")
    print(f"Overall LightGBM F1: {new_bench['lgbm_overall_f1']:.4f}")
    print(f"Overall Winner: {new_bench['overall_winner'].upper()}\n")

    # Detect and print winner changes
    if old_bench:
        changes = _detect_winner_changes(old_bench, new_bench)
        if changes:
            print("\n" + "=" * 60)
            print("  WINNER CHANGE ALERTS")
            print("=" * 60)
            for msg in changes:
                print(msg)
            print()
        else:
            print("No winner changes detected vs previous benchmark.\n")

    # Save new benchmark
    with open(output_path, "w") as f:
        json.dump(new_bench, f, indent=2)
    logger.info("Saved new benchmark to %s", output_path)

    return new_bench


def main() -> None:
    """Entry point for standalone benchmark script."""
    parser = argparse.ArgumentParser(
        description="Regenerate per-attack-type model benchmark report."
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Path to labeled CSV file with 'label' and optionally 'attack_type' columns.",
    )
    parser.add_argument(
        "--model-dir",
        default="detection/models",
        help="Directory containing saved models (default: detection/models).",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        logger.error("Model directory not found: %s", model_dir)
        sys.exit(1)

    output_path = model_dir / "model_benchmark.json"
    run_benchmark(model_dir, args.csv, output_path)


if __name__ == "__main__":
    main()
