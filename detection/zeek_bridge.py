"""Zeek conn.log -> DetectionEvent, the second network sensor.

Parallel to detection/suricata_bridge.py::SuricataBridge. Zeek's conn.log carries
the flow primitives Suricata lacks, so this bridge can drive the high-accuracy
28-feature UNSW model live, and supplies real HTTP context (http.log, joined by
uid) for CIC Web Attack — all reusing the shared compute_* funcs so there is no
training-serving skew, and emitting the SAME DetectionEvent dataclass so the
backend's risk-scorer / enrichment / store path is unchanged.

Cross-sensor correlation: every event carries Zeek's community_id, which matches
the community-id Suricata also emits for the same flow — the backend can treat a
Suricata+Zeek agreement like the existing Wazuh host+network correlation bonus.

Enable JSON conn.log in Zeek with `redef LogAscii::use_json=T;` (infra/zeek/local.zeek).

Usage:
    bridge = ZeekBridge(conn_path="/opt/zeek/logs/current/conn.log")
    for event in bridge.stream():
        ...
"""

from __future__ import annotations

import json
import logging
import os
import time
import warnings
from pathlib import Path
from typing import Any, Generator, Iterator

import joblib
import numpy as np
import pandas as pd

from detection.flow_features import CIC_FLOW_FEATURES_V2
from detection.suricata_bridge import DetectionEvent, _http_features, _sig_to_attack_type
from detection.zeek_features import (
    CtWindow, UNSW28_FEATURES, unsw28_from_zeek, zeek_conn_to_vectors, zeek_http_features,
)

logger = logging.getLogger(__name__)

_WEB_PORTS = {80, 443, 8080, 8000, 8443}


def _tail_json(path: Path, poll_interval: float = 0.1) -> Iterator[dict[str, Any] | None]:
    """Tail a Zeek JSON log (one JSON object per line). Yields None on idle so the
    caller can flush a partial batch. Identical shape to suricata_bridge._tail_eve."""
    logger.info("Tailing Zeek log %s", path)
    while not path.exists():
        time.sleep(2)
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        fh.seek(0, 2)
        while True:
            line = fh.readline()
            if not line:
                yield None
                time.sleep(poll_interval)
                continue
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                pass


class ZeekBridge:
    """Reads Zeek conn.log (+ optional http.log), runs the CIC and UNSW models."""

    def __init__(
        self,
        cic_model_path: str | Path | None = None,
        unsw_model_path: str | Path | None = None,
        conn_path: str | Path | None = None,
        http_path: str | Path | None = None,
    ) -> None:
        cic_model_path = Path(cic_model_path or os.getenv(
            "CIC_MODEL_PATH", "data/models/cic2017_pipeline_smote.joblib"))
        self.conn_path = Path(conn_path or os.getenv("ZEEK_CONN_PATH",
                                                     "/opt/zeek/logs/current/conn.log"))
        self.http_path = Path(http_path or os.getenv("ZEEK_HTTP_PATH",
                                                     "/opt/zeek/logs/current/http.log"))
        self._http_enrich = os.getenv("LIGHTHOUSE_HTTP_ENRICH", "0") == "1"

        logger.info("ZeekBridge loading CIC model from %s", cic_model_path)
        pipe = joblib.load(cic_model_path)
        self._scaler = pipe["scaler"]
        self._stage1 = pipe["stage1_model"]
        self._stage2 = pipe["stage2_model"]
        self._encoder = pipe["fam_encoder"]
        self._cic_feats = pipe.get("features", CIC_FLOW_FEATURES_V2)

        # UNSW model removed (2026) — see suricata_bridge for the rationale and
        # docs/INFORMATION.md "Why UNSW was dropped". The 28-feature variant was a
        # net regression and its weak classes are dataset-limited. CIC is the sole
        # ML detector now. unsw_* event fields are kept (always empty) for schema
        # stability. (Opt back in by setting UNSW28_MODEL_PATH if ever desired.)
        unsw_path = os.getenv("UNSW28_MODEL_PATH", "")
        if unsw_path and Path(unsw_path).exists():
            self._unsw = joblib.load(unsw_path)
            self._unsw_feats = self._unsw.get("features", UNSW28_FEATURES)
            self._freq_maps = self._unsw.get("freq_maps", {})
            logger.info("ZeekBridge loaded UNSW-28 model from %s (opt-in)", unsw_path)
        else:
            self._unsw = None
            self._unsw_feats = UNSW28_FEATURES
            self._freq_maps = {}
        self._ct = CtWindow()

    # ── CIC prediction (identical model + slicing as SuricataBridge) ──
    def _predict_cic_batch(self, vecs: list[list[float]]) -> list[tuple[str, float]]:
        if not vecs:
            return []
        raw = np.asarray(vecs, dtype=float)
        n = len(self._cic_feats)
        if raw.shape[1] != n:
            raw = raw[:, :n]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_s = self._scaler.transform(raw)
        probs = self._stage1.predict_proba(X_s)[:, 1]
        is_attack = self._stage1.predict(X_s)
        out: list[tuple[str, float]] = [("BENIGN", float(p)) for p in probs]
        att_idx = np.where(is_attack == 1)[0]
        if len(att_idx):
            X_df = pd.DataFrame(X_s[att_idx], columns=self._cic_feats)
            fams = self._stage2.predict(X_df)
            for i, fam in zip(att_idx, fams):
                out[i] = (str(self._encoder.inverse_transform([int(fam)])[0]), float(probs[i]))
        return out

    def _predict_unsw28(self, vec28: list[float] | None) -> tuple[str, float] | None:
        if self._unsw is None or vec28 is None:
            return None
        try:
            raw = np.asarray([vec28], dtype=float)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                X_s = self._unsw["scaler"].transform(raw)
            prob = float(self._unsw["stage1_model"].predict_proba(X_s)[:, 1][0])
            if int(self._unsw["stage1_model"].predict(X_s)[0]) == 0:
                return ("Normal", prob)
            enc = self._unsw.get("cat_encoder") or self._unsw["fam_encoder"]
            X_df = pd.DataFrame(X_s, columns=self._unsw_feats)
            cat = int(self._unsw["stage2_model"].predict(X_df)[0])
            return (str(enc.inverse_transform([cat])[0]), prob)
        except Exception as exc:
            logger.debug("UNSW-28 prediction failed: %s", exc)
            return None

    def stream(self, sig_pending: dict[str, str] | None = None,
               ) -> Generator[DetectionEvent, None, None]:
        """Yield DetectionEvents from Zeek flows.

        sig_pending: optional shared {community_id: signature} dict populated by the
        Suricata signatures-only reader (detection.suricata_bridge.tail_signatures).
        When a flow's community-id is present, its Suricata signature (Layer 3) is
        attached — this is how the Zeek-primary topology keeps signature detection.
        """
        sig_pending = sig_pending if sig_pending is not None else {}
        _BATCH_SIZE = int(os.getenv("INFER_BATCH_SIZE", "32"))
        _BATCH_MAX_WAIT = float(os.getenv("INFER_BATCH_MAX_WAIT", "0.1"))
        _MAX_PENDING = 2000
        http_pending: dict[str, dict] = {}          # uid -> distilled http features
        batch: list[tuple[dict, list[float], list[float] | None]] = []
        last_flush = time.time()

        def _flush() -> Generator[DetectionEvent, None, None]:
            if not batch:
                return
            cic_results = self._predict_cic_batch([b[1] for b in batch])
            for (conn, cic_vec, vec28), (prediction, attack_prob) in zip(batch, cic_results):
                unsw_res = self._predict_unsw28(vec28)
                unsw_label = unsw_res[0] if unsw_res else None
                unsw_prob = unsw_res[1] if unsw_res else 0.0
                cic_attack = prediction != "BENIGN"
                unsw_attack = unsw_label not in (None, "BENIGN", "Normal")
                is_threat = cic_attack or (unsw_attack and unsw_prob >= 0.7 and attack_prob >= 0.15)

                uid = str(conn.get("uid", ""))
                dport = int(conn.get("id.resp_p", conn.get("id_resp_p", 0)) or 0)
                http_meta: dict[str, Any] = {}
                if self._http_enrich:
                    meta = http_pending.pop(uid, None)
                    if meta and (prediction == "Web Attack" or dport in _WEB_PORTS):
                        http_meta = meta

                # Layer 3: join the Suricata signature for this flow by community-id.
                cid = str(conn.get("community_id", "") or "")
                sig = sig_pending.pop(cid, None) if cid else None

                yield DetectionEvent(
                    timestamp=str(conn.get("ts", "")),
                    src_ip=str(conn.get("id.orig_h", conn.get("id_orig_h", ""))),
                    dst_ip=str(conn.get("id.resp_h", conn.get("id_resp_h", ""))),
                    dst_port=dport,
                    proto=str(conn.get("proto", "")),
                    app_proto=str(conn.get("service", "")),
                    flow_duration_us=float(conn.get("duration", 0) or 0) * 1_000_000,
                    prediction=prediction,
                    is_threat=is_threat or (sig is not None),
                    stage1_attack_prob=attack_prob,
                    suricata_alert=sig,
                    sig_attack_type=_sig_to_attack_type(sig),
                    cic_features=dict(zip(CIC_FLOW_FEATURES_V2, cic_vec)),
                    unsw_prediction=unsw_label,
                    unsw_attack_prob=unsw_prob,
                    http_meta={**http_meta, "sensor": "zeek", "community_id": cid},
                )
            batch.clear()

        for ev in _tail_json(self.conn_path):
            if ev is None:
                if batch and (time.time() - last_flush) >= _BATCH_MAX_WAIT:
                    yield from _flush()
                    last_flush = time.time()
                continue

            vectors = zeek_conn_to_vectors(ev)
            if vectors is None:
                continue
            cic_vec, _ = vectors
            vec28 = unsw28_from_zeek(ev, self._freq_maps, self._ct) if self._unsw else None
            batch.append((ev, cic_vec, vec28))

            now = time.time()
            if len(batch) >= _BATCH_SIZE or (now - last_flush) >= _BATCH_MAX_WAIT:
                yield from _flush()
                last_flush = now

    def read_http_log(self, http_pending: dict[str, dict]) -> None:
        """Optional: fold http.log records into http_pending keyed by uid. (Wired
        from a separate tail in the backend; kept here for symmetry/testing.)"""
        for ev in _tail_json(self.http_path):
            if ev is None:
                return
            uid = str(ev.get("uid", ""))
            if uid:
                if len(http_pending) >= 2000:
                    http_pending.pop(next(iter(http_pending)))
                http_pending[uid] = zeek_http_features(ev)
