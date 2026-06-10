"""Anomaly detection using Isolation Forest and SHAP explainability."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Detect anomalies using Isolation Forest with SHAP-based explainability."""

    def __init__(self, contamination: float = 0.05) -> None:
        """Initialize the anomaly detector.

        Args:
            contamination: Expected proportion of anomalies in the dataset.
        """
        self.model = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
        self.scaler = StandardScaler()
        self.baseline_features: list[str] = []
        self.explainer: shap.TreeExplainer | None = None
        self.is_trained = False

    def _engineer_features(self, alerts: list[dict[str, Any]]) -> tuple[pd.DataFrame, list[str]]:
        """Engineer features from raw alert data.

        Args:
            alerts: List of enriched alert dictionaries.

        Returns:
            Tuple of (features_df, feature_names).
        """
        features = {}

        # Numerical features
        features["rule_level"] = [alert.get("rule_level", 0) for alert in alerts]
        features["src_port"] = [
            int(alert.get("src_port", 0)) if isinstance(alert.get("src_port"), (int, str)) else 0 
            for alert in alerts
        ]
        features["session_event_count"] = [alert.get("session_event_count", 1) for alert in alerts]
        features["session_duration_seconds"] = [alert.get("session_duration_seconds", 0) for alert in alerts]

        def _geoip_view(alert: dict[str, Any]) -> dict[str, Any]:
            geo = alert.get("geoip", {})
            if isinstance(geo, dict):
                if "is_tor" in geo or "is_vpn" in geo:
                    return geo
                if "src" in geo and isinstance(geo["src"], dict):
                    return geo["src"]
            return {}

        def _threat_view(alert: dict[str, Any]) -> dict[str, Any]:
            ti = alert.get("threat_intel", {})
            if isinstance(ti, dict):
                if "abuse_score" in ti or "is_known_attacker" in ti:
                    return ti
                if "src" in ti and isinstance(ti["src"], dict):
                    return ti["src"]
            return {}

        # GeoIP features
        features["geoip_is_tor"] = [
            int(bool(_geoip_view(alert).get("is_tor"))) for alert in alerts
        ]
        features["geoip_is_vpn"] = [
            int(bool(_geoip_view(alert).get("is_vpn"))) for alert in alerts
        ]

        # Threat intelligence features
        features["threat_intel_abuse_score"] = [
            _threat_view(alert).get("abuse_score", 0) for alert in alerts
        ]
        features["threat_intel_is_known_attacker"] = [
            int(_threat_view(alert).get("is_known_attacker", False)) for alert in alerts
        ]

        # MITRE techniques count
        features["mitre_technique_count"] = [len(alert.get("mitre_techniques", [])) for alert in alerts]

        # Timestamp hour (temporal feature)
        features["hour_of_day"] = [
            int(alert.get("timestamp", "").split("T")[1].split(":")[0]) if "T" in alert.get("timestamp", "") else 0
            for alert in alerts
        ]

        features_df = pd.DataFrame(features)
        feature_names = list(features_df.columns)
        return features_df, feature_names

    def train_baseline(self, normal_alerts: list[dict[str, Any]]) -> dict[str, Any]:
        """Train on normal alerts (rule_level < 7) to establish baseline.

        Args:
            normal_alerts: List of normal alert dictionaries.

        Returns:
            Dictionary with training info.
        """
        # Filter to rule_level < 7 (normal behavior)
        baseline_alerts = [a for a in normal_alerts if a.get("rule_level", 0) < 7]
        logger.info("Training anomaly detector on %d normal alerts", len(baseline_alerts))

        if len(baseline_alerts) < 10:
            logger.warning("Insufficient baseline data: %d samples", len(baseline_alerts))
            return {"samples": len(baseline_alerts), "status": "insufficient_data"}

        # Engineer features
        features_df, self.baseline_features = self._engineer_features(baseline_alerts)

        # Normalize
        X = self.scaler.fit_transform(features_df)

        # Fit model
        self.model.fit(X)
        self.is_trained = True

        # Initialize SHAP explainer
        try:
            self.explainer = shap.TreeExplainer(self.model)
        except Exception as exc:
            logger.warning("Failed to initialize SHAP explainer: %s", exc)

        logger.info("Anomaly detector trained successfully")
        return {
            "samples": len(baseline_alerts),
            "features": len(self.baseline_features),
            "status": "trained",
        }

    @classmethod
    def train_from_db(cls, db_path: str, n: int = 500, contamination: float = 0.05) -> "AnomalyDetector":
        """Create and train a detector using recent low-severity alerts from SQLite.

        Returns an untrained detector if the DB has fewer than 50 usable rows.
        """
        import sqlite3, json

        detector = cls(contamination=contamination)
        try:
            conn = sqlite3.connect(db_path, timeout=5)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM alerts WHERE threat_level = 0 ORDER BY ingested_at DESC LIMIT ?", (n,)
            ).fetchall()
            conn.close()

            if len(rows) < 50:
                logger.warning("train_from_db: only %d low-severity rows — skipping training", len(rows))
                return detector

            alerts = [dict(r) for r in rows]
            for a in alerts:
                for col in ("mitre_techniques", "action_history"):
                    if isinstance(a.get(col), str):
                        try:
                            a[col] = json.loads(a[col])
                        except Exception:
                            a[col] = []

            detector.train_baseline(alerts)
            logger.info("train_from_db: trained anomaly detector on %d rows from %s", len(alerts), db_path)
        except Exception as exc:
            logger.error("train_from_db failed: %s", exc)
        return detector

    def score(self, alert: dict[str, Any]) -> dict[str, Any]:
        """Score a single alert for anomalies.

        Args:
            alert: Enriched alert dictionary.

        Returns:
            Dictionary with anomaly_score, is_anomaly, anomaly_features.
        """
        if not self.is_trained or not self.baseline_features:
            logger.warning("Model not trained, skipping anomaly detection")
            return {
                "anomaly_score": 0.0,
                "is_anomaly": False,
                "anomaly_features": [],
            }

        if not hasattr(self.model, "estimators_") or not hasattr(self.scaler, "scale_"):
            logger.warning("Anomaly detector not fitted, skipping anomaly detection")
            return {
                "anomaly_score": 0.0,
                "is_anomaly": False,
                "anomaly_features": [],
            }

        try:
            # Engineer features
            features_df, _ = self._engineer_features([alert])

            # Normalize
            X = self.scaler.transform(features_df)

            # score_samples returns negative values for anomalies — negate before
            # sigmoid so anomalies map to scores near 1, normal near 0.
            raw_score = self.model.score_samples(X)[0]
            anomaly_score = 1.0 / (1.0 + np.exp(raw_score))

            # Threshold (typically -0.5 for IsolationForest)
            prediction = self.model.predict(X)[0]
            is_anomaly = prediction == -1

            # SHAP explainability
            anomaly_features = []
            if self.explainer and is_anomaly:
                try:
                    shap_values = self.explainer.shap_values(X)
                    # Get absolute SHAP values for feature importance
                    abs_shap = np.abs(shap_values[0])
                    top_indices = np.argsort(abs_shap)[-3:][::-1]  # Top 3 features
                    anomaly_features = [
                        self.baseline_features[i] for i in top_indices if i < len(self.baseline_features)
                    ]
                except Exception as exc:
                    logger.warning("SHAP explanation failed: %s", exc)

            return {
                "anomaly_score": float(anomaly_score),
                "is_anomaly": bool(is_anomaly),
                "anomaly_features": anomaly_features,
            }
        except Exception as exc:
            logger.error("Anomaly scoring failed: %s", exc)
            return {
                "anomaly_score": 0.0,
                "is_anomaly": False,
                "anomaly_features": [],
            }
