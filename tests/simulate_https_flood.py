"""Simulate an HTTPS flood attack and verify the CIC 2017 joblib model detects it."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
CIC_CSV = (
    REPO_ROOT
    / "data"
    / "raw"
    / "cic2017"
    / "MachineLearningCSV"
    / "MachineLearningCVE"
    / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
)
MODEL_PATH = Path(
    os.getenv(
        "CIC_MODEL_PATH",
        "data/models/cic2017_pipeline_smote.joblib",
    )
)

FEATURES = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Max",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Bwd IAT Mean",
    "FIN Flag Count",
    "SYN Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
]

N_SAMPLES = 1000
SEED = 42


def load_model(path: Path) -> dict:
    if not path.exists():
        print(f"[ERROR] Model not found at: {path}", file=sys.stderr)
        sys.exit(1)
    pipeline = joblib.load(path)
    if not isinstance(pipeline, dict):
        print(f"[ERROR] Expected dict pipeline, got {type(pipeline)}", file=sys.stderr)
        sys.exit(1)
    return pipeline


def load_ddos_rows(csv_path: Path, features: list[str]) -> np.ndarray:
    """Return a numpy array of clean DDoS rows for the given feature list."""
    if not csv_path.exists():
        print(f"[WARNING] CIC CSV not found at {csv_path}; cannot sample real rows.")
        return np.empty((0, len(features)))

    print(f"Loading CIC 2017 DDoS CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    label_col = next((c for c in df.columns if c.lower() == "label"), None)
    if label_col is None:
        print("[ERROR] Could not find 'Label' column in CSV.", file=sys.stderr)
        sys.exit(1)

    ddos_rows = df[df[label_col].str.strip() == "DDoS"]
    print(f"DDoS rows found: {len(ddos_rows):,} / {len(df):,} total")

    available = [f for f in features if f in ddos_rows.columns]
    missing = [f for f in features if f not in ddos_rows.columns]
    if missing:
        print(f"[WARNING] Missing columns (will use 0): {missing}")

    clean = ddos_rows[available].replace([np.inf, -np.inf], np.nan).dropna()
    return clean[available].values.astype(float)


def generate_https_flood(ddos_base: np.ndarray, model_features: list[str], n: int = N_SAMPLES) -> np.ndarray:
    """Generate synthetic HTTPS flood feature vectors.

    Bootstraps from real CIC 2017 DDoS rows to preserve all feature correlations
    and adds +/-2% jitter. If 'Destination Port' is in the feature list it is
    overridden to 443; if it was dropped during retraining the simulation runs
    port-agnostically (which is the intended behaviour after retraining).
    """
    rng = np.random.default_rng(SEED)

    if ddos_base.shape[0] == 0:
        raise RuntimeError("No DDoS rows available. Check CIC CSV path.")

    idx = rng.integers(0, ddos_base.shape[0], size=n)
    X = ddos_base[idx].copy()

    jitter = rng.uniform(0.98, 1.02, size=X.shape)
    X = X * jitter
    X = np.clip(X, 0, None)

    # If port is still a feature, override to 443
    if "Destination Port" in model_features:
        port_idx = model_features.index("Destination Port")
        X[:, port_idx] = 443

    return X


def run_pipeline(pipeline: dict, X: np.ndarray) -> list[str]:
    """Scale and run the two-stage ensemble; return a label per sample."""
    scaler = pipeline["scaler"]
    stage1 = pipeline["stage1_model"]
    stage2 = pipeline["stage2_model"]
    encoder = pipeline["fam_encoder"]

    X_scaled = scaler.transform(X)
    stage1_preds = stage1.predict(X_scaled)

    labels: list[str] = []
    for i, is_attack in enumerate(stage1_preds):
        if not is_attack:
            labels.append("BENIGN")
        else:
            family_idx = int(stage2.predict(X_scaled[i : i + 1])[0])
            family_label = str(encoder.inverse_transform([family_idx])[0])
            labels.append(family_label)

    return labels


def get_class_names(pipeline: dict) -> list[str]:
    return [str(l) for l in pipeline.get("final_labels", [])]


def main() -> None:
    print("=" * 60)
    print("  HTTPS Flood Attack Simulation")
    print("=" * 60)

    pipeline = load_model(MODEL_PATH)
    print(f"Model loaded from: {MODEL_PATH}")

    class_names = get_class_names(pipeline)
    print(f"Output classes ({len(class_names)}): {class_names}")

    model_features: list[str] = pipeline.get("features", FEATURES)
    ddos_base = load_ddos_rows(CIC_CSV, model_features)

    # --- Baseline: real DDoS rows at port 80 ---
    print("\n--- Baseline Verification (real DDoS rows, port 80) ---")
    rng = np.random.default_rng(SEED)
    base_n = min(500, ddos_base.shape[0])
    base_idx = rng.integers(0, ddos_base.shape[0], size=base_n)
    base_labels = run_pipeline(pipeline, ddos_base[base_idx])
    base_ddos = sum(1 for l in base_labels if l in ("DDoS", "DoS"))
    base_pct = 100 * base_ddos / base_n
    print(f"  {base_ddos}/{base_n} ({base_pct:.1f}%) detected as DDoS/DoS")
    print("  (Expected >95% -- confirms model is working correctly)")

    # --- HTTPS Flood Simulation ---
    print(f"\n--- HTTPS Flood Simulation (port 443, N={N_SAMPLES}) ---")
    print("  Strategy: bootstrap from real DDoS rows + Destination Port -> 443")
    print("  Note: CIC 2017 DDoS trained mainly on port 80; port shift may reduce certainty.")
    X = generate_https_flood(ddos_base, model_features, N_SAMPLES)

    print("\n  Sample vectors (first 5 features, 3 rows):")
    sample_df = pd.DataFrame(X[:3, :5], columns=model_features[:5])
    for line in sample_df.to_string(index=False).split("\n"):
        print("    " + line)

    print("\n  Running predictions...")
    pred_labels = run_pipeline(pipeline, X)

    pred_series = pd.Series(pred_labels)
    counts = pred_series.value_counts()

    print("\n  Prediction Distribution:")
    for label, count in counts.items():
        pct = 100 * count / N_SAMPLES
        bar = "#" * int(pct / 2)
        print(f"    {label:<20} {count:>5} ({pct:5.1f}%)  {bar}")

    ddos_count = sum(1 for p in pred_labels if p in ("DDoS", "DoS"))
    ddos_pct = 100 * ddos_count / N_SAMPLES
    threat_count = sum(1 for p in pred_labels if p != "BENIGN")
    threat_pct = 100 * threat_count / N_SAMPLES

    print(f"\n--- Summary ---")
    print(f"  Baseline DDoS detection  : {base_pct:.1f}%  (real port-80 DDoS rows)")
    print(f"  HTTPS flood DDoS+DoS     : {ddos_count}/{N_SAMPLES} ({ddos_pct:.1f}%)")
    print(f"  HTTPS flood any threat   : {threat_count}/{N_SAMPLES} ({threat_pct:.1f}%)")

    if ddos_pct >= 80:
        print("  DDoS result              : PASS -- model identifies HTTPS flood as DDoS")
    elif threat_pct >= 80:
        print("  DDoS result              : PARTIAL PASS")
        print("  Interpretation           : >80% flagged as non-BENIGN but misclassified attack type")
        print("  Likely cause             : Port 443 was rare in DDoS training data (CIC 2017 used port 80)")
        print("  Recommendation           : Retrain with HTTPS flood samples or drop port as a feature")
    else:
        print("  DDoS result              : NEEDS INVESTIGATION")
        print("  Likely cause             : Port 443 shifts feature vector outside DDoS decision boundary")

    print("=" * 60)


if __name__ == "__main__":
    main()
