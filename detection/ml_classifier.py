"""ML-based attack classification using XGBoost and LightGBM."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, cast

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import LabelEncoder, StandardScaler

from detection.cic_feature_bridge import CicFeatureBridge
from detection.gpu_utils import detect_gpu, xgb_gpu_params, lgb_gpu_params, xgb_fit_with_fallback, lgb_fit_with_fallback

logger = logging.getLogger(__name__)


class MLClassifier:
    """XGBoost-based classification with LightGBM fallback for attack detection."""

    def __init__(self, model_dir: str = "detection/models") -> None:
        """Initialize the ML classifier.

        Args:
            model_dir: Directory to save/load models.
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.xgb_model = None
        self.lgb_model = None
        self.scaler = StandardScaler()
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.feature_names: list[str] = []
        self.class_labels = np.array([0, 1, 2], dtype=int)
        # CIC 2017 joblib ensemble (XGBoost + LightGBM trained on network flow features)
        self.cic_model: dict | None = None
        self.cic_bridge = CicFeatureBridge()

    def _row_to_dict(self, row: Any) -> dict[str, Any]:
        if isinstance(row, dict):
            return row
        if hasattr(row, "to_dict"):
            converted = row.to_dict()
            if isinstance(converted, dict):
                return converted
        return dict(row) if hasattr(row, "items") else {}

    def _ensure_class_coverage(self, X: Any, y: Any) -> tuple[Any, Any]:
        """Ensure all three classes (0,1,2) are represented to avoid model errors."""
        if X.size == 0:
            return X, y
        present = set(np.unique(y).tolist())
        missing = [c for c in self.class_labels.tolist() if c not in present]
        if not missing:
            return X, y

        # Add synthetic rows using feature means for missing classes
        synth = np.tile(X.mean(axis=0), (len(missing), 1))
        X_aug = np.vstack([X, synth])
        y_aug = np.concatenate([y, np.array(missing, dtype=int)])
        return X_aug, y_aug

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features from raw alert data.

        Args:
            df: DataFrame with raw alert dictionaries.

        Returns:
            DataFrame with engineered features.
        """
        features = {}

        # Numerical features
        features["rule_level"] = df.get("rule_level", [0])
        features["src_port"] = pd.Series(
            [int(row.get("src_port", 0)) if isinstance(row.get("src_port"), (int, str)) else 0 for _, row in df.iterrows()]
        ).values
        features["session_event_count"] = pd.Series([row.get("session_event_count", 1) for _, row in df.iterrows()]).values
        features["session_duration_seconds"] = pd.Series([row.get("session_duration_seconds", 0) for _, row in df.iterrows()]).values

        rows: list[dict[str, Any]] = [self._row_to_dict(row) for _, row in df.iterrows()]

        def _geoip_view(row: dict[str, Any]) -> dict[str, Any]:
            geo = row.get("geoip", {})
            if isinstance(geo, dict):
                if "is_tor" in geo or "is_vpn" in geo:
                    return geo
                if "src" in geo and isinstance(geo["src"], dict):
                    return geo["src"]
            return {}

        def _threat_view(row: dict[str, Any]) -> dict[str, Any]:
            ti = row.get("threat_intel", {})
            if isinstance(ti, dict):
                if "abuse_score" in ti or "is_known_attacker" in ti:
                    return ti
                if "src" in ti and isinstance(ti["src"], dict):
                    return ti["src"]
            return {}

        # GeoIP features
        features["geoip_is_tor"] = pd.Series([bool(_geoip_view(row).get("is_tor")) for row in rows]).astype(int).values
        features["geoip_is_vpn"] = pd.Series([bool(_geoip_view(row).get("is_vpn")) for row in rows]).astype(int).values

        # Threat intelligence features
        features["threat_intel_abuse_score"] = pd.Series([_threat_view(row).get("abuse_score", 0) for row in rows]).values
        features["threat_intel_is_known_attacker"] = pd.Series([int(_threat_view(row).get("is_known_attacker", False)) for row in rows]).values

        # MITRE techniques count
        features["mitre_technique_count"] = pd.Series(
            [len(row.get("mitre_techniques", [])) for _, row in df.iterrows()]
        ).values

        # Rule groups (categorical)
        rule_groups = [",".join(row.get("rule_groups", [])) for _, row in df.iterrows()]
        features["rule_groups"] = pd.Categorical(rule_groups)

        # Asset criticality (categorical)
        features["asset_criticality"] = pd.Categorical(
            [row.get("asset_criticality", "medium") for _, row in df.iterrows()]
        )

        # Protocol (categorical)
        features["protocol"] = pd.Categorical(
            [row.get("protocol", "unknown") for _, row in df.iterrows()]
        )

        features_df = pd.DataFrame(features)
        return features_df

    def _preprocess_features(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """Preprocess features (encode categoricals, normalize numerics).

        Args:
            df: DataFrame with engineered features.
            fit: Whether to fit the encoders/scaler (True for training).

        Returns:
            Preprocessed feature array.
        """
        df_copy: Any = df.copy()

        # Encode categorical features
        categorical_cols = df_copy.select_dtypes(include="category").columns
        for col in categorical_cols:
            if fit:
                self.label_encoders[col] = LabelEncoder()
                encoded = self.label_encoders[col].fit_transform(df_copy[col].astype(str))
                df_copy.loc[:, col] = pd.Series(np.asarray(encoded, dtype=int), index=df_copy.index)
            else:
                if col in self.label_encoders:
                    encoder = self.label_encoders[col]
                    known = set(encoder.classes_.tolist())
                    values = df_copy[col].astype(str).tolist()
                    encoded = []
                    for value in values:
                        if value in known:
                            transformed = encoder.transform([value])
                            encoded.append(int(np.asarray(transformed, dtype=int)[0]))
                        else:
                            encoded.append(-1)
                    df_copy.loc[:, col] = pd.Series(np.asarray(encoded, dtype=int), index=df_copy.index)
                else:
                    df_copy.loc[:, col] = pd.Series([-1] * len(df_copy), index=df_copy.index)

        # Normalize numerical features
        numerical_cols = df_copy.select_dtypes(include=[np.number]).columns
        if fit:
            self.scaler.fit(df_copy[numerical_cols])
        df_copy[numerical_cols] = self.scaler.transform(df_copy[numerical_cols])

        self.feature_names = list(df_copy.columns)
        return df_copy.values

    def train(self, df: pd.DataFrame, y: np.ndarray | None = None) -> dict[str, Any]:
        """Train XGBoost and LightGBM models.

        Args:
            df: DataFrame with alert dictionaries.
            y: Optional target labels. If None, generates synthetic labels based on rule_level.

        Returns:
            Dictionary with training results and model comparison.
        """
        # Engineer and preprocess features
        features_df = self._engineer_features(df)
        X = self._preprocess_features(features_df, fit=True)

        # Generate or use provided labels
        if y is None:
            # Synthetic labels: rule_level > 5 = suspicious, > 6 = malicious, else benign
            rule_levels = np.asarray(features_df["rule_level"], dtype=int)
            y = np.zeros(len(rule_levels), dtype=int)
            y[rule_levels > 6] = 2  # malicious
            y[(rule_levels > 5) & (rule_levels <= 6)] = 1  # suspicious
            # 0 is benign (default)
        else:
            y = np.asarray(y, dtype=int)

        # Ensure all classes are present for multi-class models before training
        X, y = self._ensure_class_coverage(X, y)

        # Determine SMOTE eligibility on the raw data
        class_counts = np.bincount(y, minlength=3)
        min_class = int(class_counts[class_counts > 0].min()) if np.any(class_counts > 0) else 0
        smote_eligible = min_class >= 2 and len(y) >= 10

        # Final model training: apply SMOTE once to the full training set
        if smote_eligible:
            try:
                smote_k = min(5, min_class - 1)
                resampled = cast(Any, SMOTE(random_state=42, k_neighbors=smote_k).fit_resample(X, y))
                X_balanced = np.asarray(resampled[0])
                y_balanced = np.asarray(resampled[1], dtype=int)
                logger.info("Applied SMOTE: original samples %d, balanced %d", len(y), len(y_balanced))
            except Exception as exc:
                logger.warning("SMOTE failed: %s, using original data", exc)
                X_balanced, y_balanced = X, y
        else:
            X_balanced, y_balanced = X, y

        use_gpu, gpu_name = detect_gpu()
        if use_gpu:
            logger.info("Training on GPU: %s", gpu_name)
        else:
            logger.info("No GPU found — training on CPU")

        xgb_clf_params: dict[str, Any] = xgb_gpu_params(dict(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
        ), use_gpu)
        lgb_clf_params: dict[str, Any] = lgb_gpu_params(dict(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            objective="multiclass",
            num_class=3,
            min_data_in_leaf=1,
            min_data_in_bin=1,
        ), use_gpu)

        self.xgb_model = xgb_fit_with_fallback(
            xgb.XGBClassifier, xgb_clf_params, use_gpu, X=X_balanced, y=y_balanced
        )
        self.lgb_model = lgb_fit_with_fallback(
            lgb.LGBMClassifier, lgb_clf_params, use_gpu, X=X_balanced, y=y_balanced
        )

        # Cross-validation: SMOTE applied inside each fold to prevent leakage
        # (validation folds are always raw, unsynthesised samples)
        class_counts_raw = np.bincount(y, minlength=3)
        min_class_raw = int(class_counts_raw[class_counts_raw > 0].min()) if np.any(class_counts_raw > 0) else 0
        if min_class_raw >= 2 and len(y) >= 6:
            n_splits = min(5, min_class_raw)
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            cv_k = min(5, min_class_raw - 1)
            smote_step = SMOTE(random_state=42, k_neighbors=cv_k)
            xgb_pipe = ImbPipeline([("smote", smote_step), ("clf", xgb.XGBClassifier(**xgb_clf_params))])
            lgb_pipe = ImbPipeline([("smote", SMOTE(random_state=42, k_neighbors=cv_k)), ("clf", lgb.LGBMClassifier(**lgb_clf_params))])
            xgb_scores = cross_val_score(xgb_pipe, X, y, cv=cv, scoring="f1_weighted")
            lgb_scores = cross_val_score(cast(Any, lgb_pipe), X, y, cv=cv, scoring="f1_weighted")
        else:
            logger.warning("Insufficient data for CV; skipping cross-validation")
            xgb_scores = np.array([0.0])
            lgb_scores = np.array([0.0])

        logger.info("XGBoost CV F1: %.4f (±%.4f)", xgb_scores.mean(), xgb_scores.std())
        logger.info("LightGBM CV F1: %.4f (±%.4f)", lgb_scores.mean(), lgb_scores.std())

        # Save models
        # Save models (avoid sklearn metadata issues in some xgboost builds)
        try:
            self.xgb_model.get_booster().save_model(str(self.model_dir / "xgb_model.json"))
        except Exception as exc:
            logger.warning("Failed to save XGBoost model: %s", exc)
        try:
            self.lgb_model.booster_.save_model(str(self.model_dir / "lgb_model.txt"))
        except Exception as exc:
            logger.warning("Failed to save LightGBM model: %s", exc)
        logger.info("Saved models to %s", self.model_dir)

        return {
            "xgb_f1": float(xgb_scores.mean()),
            "xgb_f1_std": float(xgb_scores.std()),
            "lgb_f1": float(lgb_scores.mean()),
            "lgb_f1_std": float(lgb_scores.std()),
            "best_model": "xgb" if xgb_scores.mean() > lgb_scores.mean() else "lgb",
        }

    def predict(self, alert: dict[str, Any]) -> dict[str, Any]:
        """Predict attack type for a single alert.

        Args:
            alert: Enriched alert dictionary.

        Returns:
            Dictionary with attack_type, confidence, class_probabilities.
        """
        if not self.xgb_model and not self.lgb_model:
            logger.warning("Model not trained, returning default prediction")
            return {
                "attack_type": "benign",
                "confidence": 0.0,
                "class_probabilities": {"benign": 0.5, "suspicious": 0.3, "malicious": 0.2},
            }

        try:
            check_is_fitted(self.scaler)
        except NotFittedError:
            logger.warning("Preprocessor not fitted, returning default prediction")
            return {
                "attack_type": "benign",
                "confidence": 0.0,
                "class_probabilities": {"benign": 0.5, "suspicious": 0.3, "malicious": 0.2},
            }

        try:
            # Feature engineering for single alert
            features_df = self._engineer_features(pd.DataFrame([alert]))
            X = np.asarray(self._preprocess_features(features_df, fit=False), dtype=float)
            X = np.atleast_2d(X)

            # Predict with XGBoost (fallback to LightGBM if missing)
            if self.xgb_model:
                proba = np.asarray(self.xgb_model.predict_proba(X)[0], dtype=float)
                pred_class = int(np.asarray(self.xgb_model.predict(X), dtype=int)[0])
            else:
                model = cast(Any, self.lgb_model)
                if model is None:
                    raise RuntimeError("LightGBM model is unavailable")
                if hasattr(model, "predict_proba"):
                    proba = np.asarray(model.predict_proba(X)[0], dtype=float)
                else:
                    proba = np.asarray(model.predict(X), dtype=float).ravel()
                pred_class = int(np.argmax(proba))

            class_names = {0: "benign", 1: "suspicious", 2: "malicious"}
            # Ensure probabilities map to all 3 classes
            if proba.size < 3:
                padded = np.zeros(3, dtype=float)
                padded[: proba.size] = proba
                proba = padded

            attack_type = class_names.get(pred_class, "unknown")
            confidence = float(proba[pred_class]) if pred_class in class_names else float(np.max(proba))

            class_probabilities = {class_names[i]: float(proba[i]) for i in range(3)}

            result: dict[str, Any] = {
                "attack_type": attack_type,
                "confidence": confidence,
                "class_probabilities": class_probabilities,
            }

            # Supplement with CIC 2017 network-flow model when available
            if self.cic_model is not None:
                try:
                    cic_df = self.cic_bridge.transform(alert)
                    X_cic = self.cic_model["scaler"].transform(cic_df.values)
                    stage1_pred = int(self.cic_model["stage1_model"].predict(X_cic)[0])
                    if stage1_pred == 0:
                        cic_label = "BENIGN"
                    else:
                        family_idx = int(self.cic_model["stage2_model"].predict(X_cic)[0])
                        cic_label = str(
                            self.cic_model["fam_encoder"].inverse_transform([family_idx])[0]
                        )
                    result["cic_prediction"] = cic_label
                    result["cic_is_ddos"] = cic_label in ("DDoS", "DoS")
                except Exception as cic_exc:
                    logger.debug("CIC model prediction skipped: %s", cic_exc)

            return result
        except Exception as exc:
            logger.error("Prediction failed: %s", exc)
            return {
                "attack_type": "unknown",
                "confidence": 0.0,
                "class_probabilities": {},
            }

    def load(self) -> bool:
        """Load trained models from disk.

        Returns:
            True if models loaded successfully.
        """
        try:
            xgb_path = self.model_dir / "xgb_model.json"
            lgb_path = self.model_dir / "lgb_model.txt"

            if xgb_path.exists():
                self.xgb_model = xgb.XGBClassifier()
                self.xgb_model.load_model(str(xgb_path))
                logger.info("Loaded XGBoost model")

            if lgb_path.exists():
                self.lgb_model = lgb.Booster(model_file=str(lgb_path))
                logger.info("Loaded LightGBM model")

            # Load CIC 2017 joblib pipeline (optional — does not block startup)
            cic_path = Path(
                os.getenv(
                    "CIC_MODEL_PATH",
                    "data/models/cic2017_pipeline_smote.joblib",
                )
            )
            if cic_path.exists():
                loaded = joblib.load(cic_path)
                if isinstance(loaded, dict) and "scaler" in loaded and "stage1_model" in loaded:
                    self.cic_model = loaded
                    logger.info("Loaded CIC 2017 joblib model from %s", cic_path)
                else:
                    logger.warning("CIC model at %s has unexpected format; skipping", cic_path)
            else:
                logger.info("CIC model not found at %s; skipping", cic_path)

            return self.xgb_model is not None
        except Exception as exc:
            logger.error("Failed to load models: %s", exc)
            return False
