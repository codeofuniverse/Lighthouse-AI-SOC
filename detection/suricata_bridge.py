"""Suricata eve.json -> shared flow features -> two-stage ML detection.

Reads Suricata's eve.json in real time (tail -F style), extracts flow records,
computes features via detection/flow_features.py (the SAME module the training
scripts use — zero training-serving skew), runs the CIC two-stage pipeline plus
the optional UNSW-NB15 second opinion, and yields DetectionEvent objects.
"""

from __future__ import annotations

import json
import logging
import os
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Iterator

import joblib
import numpy as np
import pandas as pd

from detection.flow_features import (
    CIC_FLOW_FEATURES,
    CIC_FLOW_FEATURES_V2,
    UNSW_FLOW_FEATURES,
    suricata_to_vectors,
)

logger = logging.getLogger(__name__)


@dataclass
class DetectionEvent:
    timestamp: str
    src_ip: str
    dst_ip: str
    dst_port: int
    proto: str
    app_proto: str
    flow_duration_us: float
    prediction: str
    is_threat: bool
    stage1_attack_prob: float
    suricata_alert: str | None = None
    sig_attack_type: str | None = None   # parsed from Suricata signature (Layer 3)
    cic_features: dict[str, float] = field(default_factory=dict)
    unsw_prediction: str | None = None
    unsw_attack_prob: float = 0.0
    # Serve-time HTTP enrichment (opt-in via LIGHTHOUSE_HTTP_ENRICH=1). This is
    # NOT a model feature — the CIC CSVs have no HTTP content so feeding it to the
    # model would be training-serving skew. It is side-band context for the risk
    # scorer, joined from Suricata `http` events by flow_id (same pattern as alerts).
    http_meta: dict[str, Any] = field(default_factory=dict)


# Web ports — used to decide when HTTP enrichment is worth attaching.
_WEB_PORTS = {80, 443, 8080, 8000, 8443}


def _http_features(http: dict) -> dict[str, Any]:
    """Distil a Suricata http event into a few risk-relevant fields (no skew:
    these never enter the model, only the DetectionEvent enrichment)."""
    url = str(http.get("url", "") or "")
    method = str(http.get("http_method", "") or "")
    status = http.get("status")
    susp = any(tok in url.lower() for tok in
               ("'", "\"", "<script", "select ", "union ", "../", "%27", "%3c"))
    return {
        "http_method": method,
        "url_len": len(url),
        "http_status": int(status) if isinstance(status, int) else None,
        "url_suspicious": bool(susp),
    }


# Map Suricata signature text -> normalized attack type (Layer 3 initiator).
# Keyed on substrings of the lighthouse.rules msg field. Order matters:
# more specific patterns first.
_SIG_ATTACK_MAP: tuple[tuple[str, str], ...] = (
    ("syn flood",   "DDoS"),
    ("udp flood",   "DDoS"),
    ("ddos",        "DDoS"),
    ("portscan",    "PortScan"),
    ("port scan",   "PortScan"),
    ("bruteforce",  "Brute Force"),
    ("brute force", "Brute Force"),
    ("slowloris",   "DoS"),
    ("dos",         "DoS"),
)


def _sig_to_attack_type(sig: str | None) -> str | None:
    """Map a Suricata signature message to a normalized attack type, or None."""
    if not sig:
        return None
    s = sig.lower()
    for needle, atype in _SIG_ATTACK_MAP:
        if needle in s:
            return atype
    return None


def tail_signatures(eve_path: str | Path, sig_pending: dict[str, str],
                    max_pending: int = 4000) -> None:
    """Signatures-only Suricata reader (Zeek-primary topology).

    Tails eve.json for `alert` events and records {community_id: signature} into
    the shared `sig_pending` dict, which the Zeek bridge joins onto its flows by
    community-id. No model is loaded and no flow scoring happens here — this is
    purely the Layer-3 signature/IOC feed. Runs forever (call in a thread/executor).

    Requires `community-id: true` in suricata.yaml so alerts carry the same
    community-id Zeek emits for the connection.
    """
    path = Path(eve_path)
    for ev in _tail_eve(path):
        if ev is None:
            continue
        if ev.get("event_type") != "alert":
            continue
        cid = str(ev.get("community_id", "") or "")
        if not cid:
            continue
        if len(sig_pending) >= max_pending:
            sig_pending.pop(next(iter(sig_pending)))
        sig_pending[cid] = ev.get("alert", {}).get("signature", "")


def _tail_eve(path: Path, poll_interval: float = 0.1) -> Iterator[dict[str, Any] | None]:
    """Tail eve.json. Yields parsed events, or None on idle so callers can flush
    a partial inference batch instead of waiting indefinitely for the next line."""
    logger.info("Tailing %s", path)
    while not path.exists():
        time.sleep(2)
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        fh.seek(0, 2)
        while True:
            line = fh.readline()
            if not line:
                yield None  # idle tick — lets the batch flush
                time.sleep(poll_interval)
                continue
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                pass


class SuricataBridge:
    """Reads Suricata eve.json, computes shared features, runs ML models."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        eve_path: str | Path = "/var/log/suricata/eve.json",
    ) -> None:
        model_path = Path(model_path or os.getenv(
            "CIC_MODEL_PATH", "data/models/cic2017_pipeline_smote.joblib"))
        self.eve_path = Path(eve_path)

        logger.info("Loading CIC model from %s", model_path)
        pipe = joblib.load(model_path)
        self._scaler  = pipe["scaler"]
        self._stage1  = pipe["stage1_model"]
        self._stage2  = pipe["stage2_model"]
        self._encoder = pipe["fam_encoder"]
        self._cic_feats = pipe.get("features", CIC_FLOW_FEATURES)

        # UNSW model removed (2026): its weak attack classes (DoS/Shellcode/Worms)
        # are limited by UNSW-NB15's intrinsic DoS↔Exploits label overlap and tiny
        # rare-class support — not fixable with features — and the 28-feature variant
        # was a net regression (accuracy 0.77→0.75, Exploits F1 0.71→0.69). It only
        # ever corroborated, never fired alone, so dropping it removes noise without
        # losing standalone detections. See docs/INFORMATION.md "Why UNSW was dropped".
        # The unsw_* fields on DetectionEvent are kept (always empty) for schema
        # stability across the DB / API / frontend.
        self._unsw = None
        self._unsw_feats = UNSW_FLOW_FEATURES

    def _predict_cic(self, cic_vec: list[float]) -> tuple[str, float]:
        return self._predict_cic_batch([cic_vec])[0]

    def _predict_cic_batch(self, vecs: list[list[float]]) -> list[tuple[str, float]]:
        """Vectorized CIC prediction over a batch of flow vectors.

        One scaler.transform + predict_proba call for the whole batch instead of
        N calls — cuts per-flow overhead under heavy load (Kajiura & Nakamura 2024).
        """
        if not vecs:
            return []
        raw = np.asarray(vecs, dtype=float)
        # suricata_to_vectors now emits the 18-feature V2 vector (…+dst_port). A
        # legacy 17-feature model takes the leading slots (identical order); a V2
        # model takes all of them. Slice to the loaded model's feature count.
        n_feats = len(self._cic_feats)
        if raw.shape[1] != n_feats:
            raw = raw[:, :n_feats]
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

    def _predict_unsw_batch(self, vecs: list[list[float]]) -> list[tuple[str, float] | None]:
        """Vectorized UNSW prediction over a batch. Returns None per-row on failure."""
        if self._unsw is None or not vecs:
            return [None] * len(vecs)
        try:
            raw = np.asarray(vecs, dtype=float)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                X_s = self._unsw["scaler"].transform(raw)
            probs = self._unsw["stage1_model"].predict_proba(X_s)[:, 1]
            is_attack = self._unsw["stage1_model"].predict(X_s)
            enc = self._unsw.get("cat_encoder") or self._unsw["fam_encoder"]
            out: list[tuple[str, float] | None] = [("Normal", float(p)) for p in probs]
            att_idx = np.where(is_attack == 1)[0]
            if len(att_idx):
                X_df = pd.DataFrame(X_s[att_idx], columns=self._unsw_feats)
                fams = self._unsw["stage2_model"].predict(X_df)
                for i, fam in zip(att_idx, fams):
                    out[i] = (str(enc.inverse_transform([int(fam)])[0]), float(probs[i]))
            return out
        except Exception as exc:
            logger.debug("UNSW batch prediction failed: %s", exc)
            return [None] * len(vecs)

    def stream(self) -> Generator[DetectionEvent, None, None]:
        """Yield DetectionEvents, batching flow inference for throughput.

        Flows are accumulated up to _BATCH_SIZE or _BATCH_MAX_WAIT seconds (so a
        trickle of flows isn't delayed), then classified in one vectorized call.
        """
        _MAX_PENDING = 2000
        _BATCH_SIZE = int(os.getenv("INFER_BATCH_SIZE", "32"))
        _BATCH_MAX_WAIT = float(os.getenv("INFER_BATCH_MAX_WAIT", "0.1"))
        _HTTP_ENRICH = os.getenv("LIGHTHOUSE_HTTP_ENRICH", "0") == "1"
        pending: dict[str, str] = {}
        http_pending: dict[str, dict] = {}   # flow_id -> distilled http features
        batch: list[tuple[dict, list[float], list[float]]] = []
        last_flush = time.time()

        def _flush() -> Generator[DetectionEvent, None, None]:
            if not batch:
                return
            cic_vecs  = [b[1] for b in batch]
            unsw_vecs = [b[2] for b in batch]
            try:
                cic_results = self._predict_cic_batch(cic_vecs)
            except Exception as exc:
                logger.warning("CIC batch prediction failed: %s", exc)
                batch.clear()
                return
            unsw_results = self._predict_unsw_batch(unsw_vecs)

            for (ev_, cic_vec_, _), (prediction, attack_prob), unsw_res in zip(
                batch, cic_results, unsw_results
            ):
                unsw_label = unsw_res[0] if unsw_res else None
                unsw_prob  = unsw_res[1] if unsw_res else 0.0
                cic_attack  = prediction != "BENIGN"
                unsw_attack = unsw_label not in (None, "BENIGN", "Normal")
                is_threat = cic_attack or (unsw_attack and unsw_prob >= 0.7 and attack_prob >= 0.15)

                fid = str(ev_.get("flow_id", ""))
                sig = pending.pop(fid, None)
                # Opt-in HTTP enrichment: attach distilled http features for web flows.
                http_meta: dict[str, Any] = {}
                if _HTTP_ENRICH:
                    meta = http_pending.pop(fid, None)
                    dport = int(ev_.get("dest_port", 0) or 0)
                    if meta and (prediction == "Web Attack" or dport in _WEB_PORTS):
                        http_meta = meta
                yield DetectionEvent(
                    timestamp        = ev_.get("timestamp", ""),
                    src_ip           = ev_.get("src_ip", ""),
                    dst_ip           = ev_.get("dest_ip", ""),
                    dst_port         = int(ev_.get("dest_port", 0) or 0),
                    proto            = ev_.get("proto", ""),
                    app_proto        = ev_.get("app_proto", ""),
                    flow_duration_us = cic_vec_[0] * 1_000_000,  # duration_s -> us
                    prediction       = prediction,
                    is_threat        = is_threat,
                    stage1_attack_prob = attack_prob,
                    suricata_alert   = sig,
                    sig_attack_type  = _sig_to_attack_type(sig),
                    cic_features     = dict(zip(CIC_FLOW_FEATURES_V2, cic_vec_)),
                    unsw_prediction  = unsw_label,
                    unsw_attack_prob = unsw_prob,
                    http_meta        = http_meta,
                )
            batch.clear()

        for ev in _tail_eve(self.eve_path):
            # Idle tick: flush any partial batch so low-rate flows aren't delayed.
            if ev is None:
                if batch and (time.time() - last_flush) >= _BATCH_MAX_WAIT:
                    yield from _flush()
                    last_flush = time.time()
                continue

            etype = ev.get("event_type", "")
            if etype == "alert":
                fid = str(ev.get("flow_id", ""))
                if fid:
                    if len(pending) >= _MAX_PENDING:
                        pending.pop(next(iter(pending)))
                    pending[fid] = ev.get("alert", {}).get("signature", "")
                continue
            if etype == "http" and _HTTP_ENRICH:
                fid = str(ev.get("flow_id", ""))
                if fid:
                    if len(http_pending) >= _MAX_PENDING:
                        http_pending.pop(next(iter(http_pending)))
                    http_pending[fid] = _http_features(ev.get("http", {}) or {})
                continue
            if etype != "flow":
                continue

            vectors = suricata_to_vectors(ev)
            if vectors is None:
                continue
            cic_vec, unsw_vec = vectors
            batch.append((ev, cic_vec, unsw_vec))

            now = time.time()
            if len(batch) >= _BATCH_SIZE or (now - last_flush) >= _BATCH_MAX_WAIT:
                yield from _flush()
                last_flush = now
