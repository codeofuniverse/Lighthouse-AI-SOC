"""SHAP on the PCAP-rich Web-Attack model — confirm what the extra features add.

Closes the loop the user asked for: explainability drives the v3 feature decision.
Trains the v3-full Web-Attack model (flow + HTTP + IAT) on the PCAP-extracted rich
parquet and runs SHAP so the ranking shows whether the HTTP and inter-arrival-time
features are what lift recall/precision over the flow-only ceiling.

Output: reports/feature_discovery/cic/shap_v3_webattack.png
        reports/feature_discovery/cic_v3_ranking.json

Usage:
    python scripts/discover_v3_features.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_cic_v3_webattack import _derive, FLOW_18, HTTP_FEATS, IAT_FEATS, RICH  # noqa: E402

warnings.filterwarnings("ignore")

DARK_BG, PANEL_BG, GRID, ACCENT = "#0d1117", "#161b22", "#30363d", "#00ff88"
plt.rcParams.update({
    "figure.facecolor": DARK_BG, "axes.facecolor": PANEL_BG, "savefig.facecolor": DARK_BG,
    "text.color": "white", "axes.labelcolor": "#8b949e",
    "xtick.color": "#8b949e", "ytick.color": "#8b949e", "axes.edgecolor": GRID,
})

OUT_FIG = Path("reports/feature_discovery/cic/shap_v3_webattack.png")
OUT_JSON = Path("reports/feature_discovery/cic_v3_ranking.json")
# tag each feature with the sensor that can reproduce it live
_SENSOR = {**{f: "Suricata/Zeek" for f in FLOW_18},
           **{f: "Zeek http.log" for f in HTTP_FEATS},
           **{f: "Zeek packet-timing policy" for f in IAT_FEATS}}


def main() -> None:
    import pandas as pd
    print("=" * 70)
    print("  SHAP on PCAP-rich Web-Attack model (flow + HTTP + IAT)")
    print("=" * 70)
    df = _derive(pd.read_parquet(RICH))
    feats = FLOW_18 + HTTP_FEATS + IAT_FEATS
    feats = [f for f in feats if f in df.columns]
    X = df[feats].values.astype(float)
    y = (df["Label"] == "Web Attack").astype(int).values

    clf = lgb.LGBMClassifier(n_estimators=400, max_depth=8, num_leaves=63,
                             learning_rate=0.05, class_weight="balanced",
                             random_state=42, n_jobs=-1, verbose=-1)
    clf.fit(X, y, feature_name=feats)

    rng = np.random.default_rng(42)
    idx = rng.choice(len(X), min(5000, len(X)), replace=False)
    sv = shap.TreeExplainer(clf).shap_values(X[idx])
    sv_pos = sv[1] if isinstance(sv, list) else sv
    mean_abs = np.abs(sv_pos).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]

    plt.figure(figsize=(10, 8))
    colors = []
    for i in order:
        s = _SENSOR.get(feats[i], "")
        colors.append("#f093fb" if "timing" in s else ("#4ecdc4" if "http" in s.lower() else ACCENT))
    plt.barh([feats[i] for i in order][::-1], mean_abs[order][::-1], color=colors[::-1])
    plt.xlabel("mean |SHAP| toward Web Attack")
    plt.suptitle("Web Attack drivers (PCAP-rich): green=flow, teal=HTTP, pink=IAT timing",
                 color="white", fontsize=12, fontweight="bold")
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  saved -> {OUT_FIG}")

    ranking = [{"rank": r + 1, "feature": feats[i], "mean_abs_shap": float(mean_abs[i]),
                "reproducible_by": _SENSOR.get(feats[i], "?")} for r, i in enumerate(order)]
    OUT_JSON.write_text(json.dumps({"ranking": ranking}, indent=2))
    print("  Top 6:", [feats[i] for i in order[:6]])
    print(f"  Saved {OUT_JSON}")
    print("  Done.")


if __name__ == "__main__":
    main()
