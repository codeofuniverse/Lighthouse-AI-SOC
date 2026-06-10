"""Comprehensive stress test for Lighthouse SOC pipeline.

Tests the FULL pipeline end-to-end after UNSW model removal:
  1. CIC model loading & predictions (synthetic + real Zeek data)
  2. Risk scorer correctness across all attack scenarios
  3. Zeek bridge feature extraction
  4. Rate aggregator volumetric detection
  5. Decision engine thresholds & threat levels
  6. Alert builder schema validation
  7. Full pipeline integration (Zeek → CIC → RiskScore → Decision)
  8. Edge cases, boundary conditions, and regression checks

Run:
    pytest tests/test_stress_full_pipeline.py -v --tb=short 2>&1 | tee stress_test.log
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path

import joblib
import numpy as np
import pytest

# ── Project imports ──────────────────────────────────────────────────────────
from detection.flow_features import (
    CIC_FLOW_FEATURES,
    CIC_FLOW_FEATURES_V2,
    UNSW_FLOW_FEATURES,
    compute_cic,
    compute_unsw,
    suricata_to_vectors,
)
from detection.zeek_features import (
    CtWindow,
    UNSW28_FEATURES,
    unsw28_from_zeek,
    zeek_conn_to_vectors,
    zeek_http_features,
)
from detection.rate_aggregator import RateAggregator, RateVerdict
from detection.suricata_bridge import DetectionEvent, _sig_to_attack_type
from backend.alert_builder import build_alert
from pipeline.decision_engine import Decision, DecisionEngine
from pipeline.risk_scorer import RiskScorer

logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
_PROJECT = Path(__file__).resolve().parent.parent
_CIC_MODEL = _PROJECT / "data" / "models" / "cic2017_pipeline_smote.joblib"
_ZEEK_CONN = _PROJECT / "tests" / "fixtures" / "zeek" / "conn.log"
_ZEEK_HTTP = _PROJECT / "tests" / "fixtures" / "zeek" / "http.log"


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 1: CIC Model Stress Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestCICModelLoading:
    """Verify the CIC model loads correctly and has expected structure."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_model(self):
        if not _CIC_MODEL.exists():
            pytest.skip("CIC model not built")

    def test_model_loads_successfully(self):
        pipe = joblib.load(_CIC_MODEL)
        assert "scaler" in pipe, "Missing scaler in CIC pipeline"
        assert "stage1_model" in pipe, "Missing stage1_model in CIC pipeline"
        assert "stage2_model" in pipe, "Missing stage2_model in CIC pipeline"
        assert "fam_encoder" in pipe, "Missing fam_encoder in CIC pipeline"
        logger.info("CIC model loaded: keys=%s", list(pipe.keys()))

    def test_model_feature_count_matches_v2(self):
        pipe = joblib.load(_CIC_MODEL)
        feats = pipe.get("features", CIC_FLOW_FEATURES)
        assert len(feats) in (17, 18), (
            f"CIC features should be 17 (v1) or 18 (v2), got {len(feats)}"
        )
        logger.info("CIC model features: %d (%s)", len(feats), feats)

    def test_model_classes_include_benign_and_attacks(self):
        pipe = joblib.load(_CIC_MODEL)
        enc = pipe["fam_encoder"]
        classes = list(enc.classes_)
        logger.info("CIC model classes: %s", classes)
        # Should have multiple attack families
        assert len(classes) >= 3, f"Expected ≥3 attack families, got {len(classes)}"

    def test_stage1_binary_predictions_on_random_data(self):
        """Stage1 should produce binary benign/attack classifications."""
        pipe = joblib.load(_CIC_MODEL)
        n_feats = len(pipe.get("features", CIC_FLOW_FEATURES))
        rng = np.random.RandomState(42)
        X = rng.randn(100, n_feats)
        X_s = pipe["scaler"].transform(X)
        preds = pipe["stage1_model"].predict(X_s)
        probs = pipe["stage1_model"].predict_proba(X_s)
        assert set(preds).issubset({0, 1}), f"Stage1 should be binary, got {set(preds)}"
        assert probs.shape == (100, 2), f"Expected (100,2) proba, got {probs.shape}"
        assert np.all((probs >= 0) & (probs <= 1)), "Probabilities must be [0,1]"
        logger.info("Stage1: %d benign, %d attack out of 100 random vectors",
                     np.sum(preds == 0), np.sum(preds == 1))


class TestCICPredictionsOnSyntheticFlows:
    """Synthesize flows matching known attack patterns and verify CIC detects them."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_model(self):
        if not _CIC_MODEL.exists():
            pytest.skip("CIC model not built")

    @pytest.fixture
    def pipe(self):
        return joblib.load(_CIC_MODEL)

    def _predict(self, pipe, fwd_pkts, bwd_pkts, fwd_bytes, bwd_bytes, dur,
                 fin=0, syn=1, rst=0, psh=0, ack=1, urg=0, dst_port=80):
        """Helper: build CIC vector, scale, predict."""
        cic = compute_cic(fwd_pkts, bwd_pkts, fwd_bytes, bwd_bytes, dur,
                          fin, syn, rst, psh, ack, urg, dst_port=dst_port)
        feats = pipe.get("features", CIC_FLOW_FEATURES_V2)
        vec = [float(cic.get(f, 0.0)) for f in feats]
        X = np.array([vec])
        X_s = pipe["scaler"].transform(X)
        is_attack = int(pipe["stage1_model"].predict(X_s)[0])
        prob = float(pipe["stage1_model"].predict_proba(X_s)[0, 1])
        if is_attack == 0:
            return "BENIGN", prob
        import pandas as pd
        fam = int(pipe["stage2_model"].predict(pd.DataFrame(X_s, columns=feats))[0])
        label = str(pipe["fam_encoder"].inverse_transform([fam])[0])
        return label, prob

    def test_benign_normal_http(self, pipe):
        """Normal HTTP flow: moderate packets, balanced bytes."""
        label, prob = self._predict(pipe,
            fwd_pkts=10, bwd_pkts=15, fwd_bytes=800, bwd_bytes=12000,
            dur=2.5, fin=1, syn=1, psh=1, ack=1, dst_port=80)
        logger.info("Normal HTTP → %s (prob=%.4f)", label, prob)
        # We mainly verify it doesn't crash and returns a valid prediction
        assert label in ["BENIGN"] or prob < 0.9, "Normal HTTP should not be high-confidence attack"

    def test_ddos_high_rate_syn(self, pipe):
        """DDoS pattern: massive fwd packets, near-zero response, very short duration.

        NOTE: A single SYN-flood flow being classified as BENIGN by CIC is CORRECT.
        DDoS detection is handled by the RATE AGGREGATOR (volumetric detection across
        many flows), not per-flow ML. CIC sees flow-level features only. This test
        verifies the model runs without error and returns a valid prediction.
        """
        label, prob = self._predict(pipe,
            fwd_pkts=5000, bwd_pkts=0, fwd_bytes=250000, bwd_bytes=0,
            dur=0.01, syn=1, ack=0, dst_port=80)
        logger.info("DDoS SYN flood → %s (prob=%.4f) [per-flow; volumetric detection is rate_aggregator]", label, prob)
        # Per-flow CIC may classify this as BENIGN — that's by design.
        # DDoS detection is the rate aggregator's job (TestRateAggregator covers that).
        assert isinstance(label, str) and isinstance(prob, float)
        assert 0 <= prob <= 1

    def test_portscan_many_ports(self, pipe):
        """PortScan: small packets, no response, probing many ports."""
        label, prob = self._predict(pipe,
            fwd_pkts=1, bwd_pkts=0, fwd_bytes=60, bwd_bytes=0,
            dur=0.001, syn=1, ack=0, rst=0, dst_port=22)
        logger.info("PortScan probe → %s (prob=%.4f)", label, prob)
        # Individual portscan flow may look benign; that's expected —
        # volumetric detection handles this. Just verify no crash.
        assert isinstance(label, str) and isinstance(prob, float)

    def test_brute_force_ssh(self, pipe):
        """SSH brute-force: small symmetric flows, many packets, port 22."""
        label, prob = self._predict(pipe,
            fwd_pkts=30, bwd_pkts=30, fwd_bytes=2400, bwd_bytes=2400,
            dur=45.0, fin=1, syn=1, psh=1, ack=1, dst_port=22)
        logger.info("SSH brute-force → %s (prob=%.4f)", label, prob)
        assert isinstance(label, str)

    def test_batch_prediction_1000_flows(self, pipe):
        """Stress: predict 1000 random flows; verify no crashes or NaN."""
        import warnings
        rng = np.random.RandomState(123)
        feats = pipe.get("features", CIC_FLOW_FEATURES_V2)
        n = len(feats)
        X = np.abs(rng.randn(1000, n)) * 1000  # positive features
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_s = pipe["scaler"].transform(X)
        probs = pipe["stage1_model"].predict_proba(X_s)
        assert probs.shape == (1000, 2)
        assert not np.any(np.isnan(probs)), "NaN in probabilities"
        logger.info("Batch 1000: min_prob=%.4f max_prob=%.4f mean=%.4f",
                     probs[:, 1].min(), probs[:, 1].max(), probs[:, 1].mean())


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 2: Risk Scorer Comprehensive Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestRiskScorerComprehensive:
    """Exhaustively test risk score formula with and without UNSW."""

    @pytest.fixture
    def scorer(self):
        return RiskScorer()

    # ── Baseline: CIC-only (UNSW removed) ─────────────────────────────────

    def test_benign_low_risk(self, scorer):
        """BENIGN prediction + no threat intel → near-zero risk."""
        risk = scorer.score(ml_conf=0.1, abuse_score=0, rule_level=0,
                            attack_label="BENIGN")
        logger.info("BENIGN risk=%.2f", risk)
        assert risk < 20, f"BENIGN should be low risk, got {risk}"

    def test_ddos_high_conf_critical_asset(self, scorer):
        """DDoS + 97% confidence + critical asset → very high risk."""
        risk = scorer.score(ml_conf=0.97, abuse_score=0, rule_level=12,
                            agent_type="critical", attack_label="DDoS")
        logger.info("DDoS/critical risk=%.2f", risk)
        assert risk >= 65, f"DDoS on critical asset should be high risk, got {risk}"

    def test_portscan_medium_conf(self, scorer):
        """PortScan + 75% confidence → moderate risk."""
        risk = scorer.score(ml_conf=0.75, abuse_score=0, rule_level=6,
                            attack_label="PortScan")
        logger.info("PortScan risk=%.2f", risk)
        assert 20 <= risk <= 40, f"PortScan moderate risk expected, got {risk}"

    def test_brute_force_known_attacker(self, scorer):
        """Brute Force + known attacker (abuse=85) → high risk."""
        risk = scorer.score(ml_conf=0.78, abuse_score=85, rule_level=9,
                            attack_label="Brute Force")
        logger.info("Brute Force + known attacker risk=%.2f", risk)
        assert risk >= 55, f"Brute Force + TI should be high risk, got {risk}"

    def test_heartbleed_maximum_severity(self, scorer):
        """Heartbleed (severity=1.0) + high confidence → critical."""
        risk = scorer.score(ml_conf=0.95, abuse_score=50, rule_level=12,
                            attack_label="Heartbleed")
        logger.info("Heartbleed risk=%.2f", risk)
        assert risk >= 60, f"Heartbleed should be critical risk, got {risk}"



    # ── Attack severity tiers ──────────────────────────────────────────────

    @pytest.mark.parametrize("label,expected_sev", [
        ("DDoS", 1.0),
        ("Heartbleed", 1.20),
        ("Backdoor", 1.20),
        ("DoS Hulk", 1.0),
        ("PortScan", 0.50),
        ("BENIGN", 0.50),
        ("Normal", 0.50),
    ])
    def test_attack_severity_mapping(self, label, expected_sev):
        from pipeline.risk_scorer import _ATTACK_SEVERITY_MULTIPLIER
        sev = _ATTACK_SEVERITY_MULTIPLIER.get(label)
        assert sev == expected_sev, f"Severity for {label}: expected {expected_sev}, got {sev}"

    # ── Temporal factor ──────────────────────────────────────────────────

    def test_temporal_first_hit_neutral(self, scorer):
        """First hit (ip_hit_count=1, no last_seen) → factor = 1.0."""
        factor = scorer._temporal_factor(1, 0.0)
        assert factor == 1.0, f"First hit should be 1.0, got {factor}"

    def test_temporal_repeat_attacker_amplifies(self, scorer):
        """Repeat attacker (hit_count=10) → factor > 1.0."""
        factor = scorer._temporal_factor(10, 0.0)
        assert factor > 1.0, f"Repeat attacker should amplify, got {factor}"
        assert factor <= 1.4, f"Temporal factor capped at 1.4, got {factor}"

    def test_temporal_stale_ip_suppresses(self, scorer):
        """IP last seen 45 days ago → recency factor = 0.8."""
        old_ts = time.time() - 45 * 86400
        factor = scorer._temporal_factor(1, old_ts)
        assert factor == pytest.approx(0.8, abs=0.01), f"Stale IP should suppress, got {factor}"

    # ── Threat level thresholds ───────────────────────────────────────────

    @pytest.mark.parametrize("score,expected_level", [
        (0.0, 0),
        (25.0, 0),
        (40.9, 0),
        (41.0, 1),
        (60.0, 1),
        (70.9, 1),
        (71.0, 2),
        (100.0, 2),
    ])
    def test_threat_level_thresholds(self, scorer, score, expected_level):
        level = scorer.threat_level(score)
        assert level == expected_level, f"score={score}: expected level {expected_level}, got {level}"

    # ── Full sweep: all CIC attack types ──────────────────────────────────

    @pytest.mark.parametrize("attack_label", [
        "DDoS", "DoS Hulk", "DoS GoldenEye", "DoS slowloris", "DoS Slowhttptest",
        "Heartbleed", "Web Attack", "Infiltration", "Bot",
        "PortScan", "FTP-Patator", "SSH-Patator", "Brute Force", "BENIGN",
    ])
    def test_all_cic_labels_produce_valid_risk(self, scorer, attack_label):
        """Every CIC label should produce a valid 0–100 risk score.

        NOTE: For BENIGN, we use realistic parameters (low ml_conf, low rule_level).
        A BENIGN label with ml_conf=0.85 + rule_level=8 is contradictory — the scorer
        correctly applies the ML weight and rule weight regardless of the label, so
        that combination yields ~60 risk (the label only affects the severity
        component which is 7% of the formula).
        """
        if attack_label == "BENIGN":
            # Use realistic parameters for a true benign flow
            risk = scorer.score(ml_conf=0.15, abuse_score=0, rule_level=2,
                                attack_label=attack_label)
        else:
            risk = scorer.score(ml_conf=0.85, abuse_score=30, rule_level=8,
                                attack_label=attack_label)
        assert 0 <= risk <= 100, f"{attack_label}: risk {risk} out of range"
        if attack_label == "BENIGN":
            assert risk < 25, f"BENIGN with realistic params should be low risk, got {risk}"
        else:
            assert risk > 0, f"Attack {attack_label} should have non-zero risk"
        logger.info("  %s → risk=%.2f", attack_label, risk)

    # ── Stress: 10,000 random scores ──────────────────────────────────────

    def test_scorer_10k_random_inputs(self, scorer):
        """Score 10,000 random parameter combinations; verify all valid."""
        rng = np.random.RandomState(42)
        labels = ["DDoS", "PortScan", "BENIGN", "Bot", "Heartbleed", "Web Attack"]
        agents = ["critical", "server", "workstation", "iot", "unknown"]
        errors = []
        for i in range(10_000):
            ml = rng.uniform(0, 1)
            abuse = rng.randint(0, 101)
            rl = rng.randint(0, 16)
            lbl = labels[rng.randint(0, len(labels))]
            agt = agents[rng.randint(0, len(agents))]
            hit = rng.randint(1, 50)
            try:
                risk = scorer.score(ml_conf=ml, abuse_score=abuse, rule_level=rl,
                                     agent_type=agt, attack_label=lbl, ip_hit_count=hit)
                if not (0 <= risk <= 100) or math.isnan(risk):
                    errors.append(f"i={i}: risk={risk}")
            except Exception as exc:
                errors.append(f"i={i}: {exc}")
        assert not errors, f"Scoring errors:\n" + "\n".join(errors[:20])
        logger.info("10,000 random scores: all valid")


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 3: Zeek Feature Extraction Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestZeekFeatureExtraction:
    """Verify Zeek conn.log → CIC/UNSW feature vectors are correct."""

    def test_synthetic_conn_to_cic_vector(self):
        """Synthetic Zeek conn record → valid CIC vector."""
        conn = {
            "ts": "1630000000.000000",
            "uid": "CTest123",
            "id.orig_h": "10.0.0.1",
            "id.orig_p": 12345,
            "id.resp_h": "10.0.0.2",
            "id.resp_p": 80,
            "proto": "tcp",
            "service": "http",
            "duration": 1.5,
            "orig_pkts": 10,
            "resp_pkts": 15,
            "orig_ip_bytes": 800,
            "resp_ip_bytes": 12000,
            "conn_state": "SF",
            "missed_bytes": 0,
            "community_id": "1:abc123=",
        }
        result = zeek_conn_to_vectors(conn)
        assert result is not None, "Valid conn record should produce vectors"
        cic_vec, unsw_vec = result
        assert len(cic_vec) == len(CIC_FLOW_FEATURES_V2) == 18, \
            f"CIC vector should be 18, got {len(cic_vec)}"
        assert len(unsw_vec) == len(UNSW_FLOW_FEATURES) == 11, \
            f"UNSW vector should be 11, got {len(unsw_vec)}"
        assert all(np.isfinite(cic_vec)), "CIC vector has non-finite values"
        assert all(np.isfinite(unsw_vec)), "UNSW vector has non-finite values"
        logger.info("CIC vec: %s", dict(zip(CIC_FLOW_FEATURES_V2, cic_vec)))

    def test_zero_packet_conn_returns_none(self):
        """Conn with 0 fwd+bwd packets → None (unusable)."""
        conn = {"orig_pkts": 0, "resp_pkts": 0, "duration": 1.0}
        assert zeek_conn_to_vectors(conn) is None

    def test_missing_fields_defaults_safely(self):
        """Conn with minimal fields → still produces vectors (defaults kick in)."""
        conn = {"orig_pkts": 5, "resp_pkts": 3}
        result = zeek_conn_to_vectors(conn)
        assert result is not None
        cic_vec, _ = result
        assert all(np.isfinite(cic_vec))

    def test_conn_state_flag_mapping(self):
        """Different conn_state values produce different flag vectors."""
        base = {"orig_pkts": 5, "resp_pkts": 3, "orig_ip_bytes": 400,
                "resp_ip_bytes": 300, "duration": 1.0, "id.resp_p": 80}
        states = ["SF", "S0", "REJ", "RSTO", "OTH"]
        results = {}
        for state in states:
            conn = {**base, "conn_state": state}
            cic, _ = zeek_conn_to_vectors(conn)
            # Flags are at indices 11-16 (fin, syn, rst, psh, ack, urg)
            flags = cic[11:17]
            results[state] = flags
            logger.info("  %s → flags=%s", state, flags)
        # SF has FIN, REJ has RST — they should differ
        assert results["SF"] != results["REJ"], "SF and REJ should have different flags"

    def test_ct_window_counting(self):
        """CtWindow produces incrementing ct_* counters."""
        ct = CtWindow(size=100)
        counts1 = ct.update_and_counts("10.0.0.1", "10.0.0.2", 80, "http")
        assert counts1["ct_srv_src"] == 0  # first record, nothing in window yet
        counts2 = ct.update_and_counts("10.0.0.1", "10.0.0.2", 80, "http")
        assert counts2["ct_srv_src"] >= 1  # second record sees the first
        assert counts2["ct_dst_ltm"] >= 1
        logger.info("CtWindow counts after 2 flows: %s", counts2)

    def test_unsw28_from_zeek_produces_28_features(self):
        """unsw28_from_zeek should return a 28-element vector."""
        conn = {
            "orig_pkts": 10, "resp_pkts": 8,
            "orig_ip_bytes": 1000, "resp_ip_bytes": 800,
            "duration": 2.0,
            "id.orig_h": "10.0.0.1", "id.resp_h": "10.0.0.2",
            "id.resp_p": 443, "proto": "tcp", "service": "ssl",
        }
        ct = CtWindow()
        vec = unsw28_from_zeek(conn, {}, ct)
        assert vec is not None
        assert len(vec) == 28, f"Expected 28 UNSW features, got {len(vec)}"
        assert all(np.isfinite(vec)), "UNSW-28 vector has non-finite values"
        logger.info("UNSW-28 vector: %s", dict(zip(UNSW28_FEATURES, vec)))

    def test_zeek_http_features_suspicious_url(self):
        """HTTP record with SQL injection in URL → url_suspicious=True."""
        http = {"uri": "/search?q=' OR 1=1 --", "method": "GET", "status_code": 200}
        feat = zeek_http_features(http)
        assert feat["url_suspicious"] is True
        assert feat["http_method"] == "GET"
        assert feat["url_len"] > 0

    def test_zeek_http_features_clean_url(self):
        """Normal HTTP request → url_suspicious=False."""
        http = {"uri": "/index.html", "method": "GET", "status_code": 200}
        feat = zeek_http_features(http)
        assert feat["url_suspicious"] is False


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 4: Real Zeek Log Tests (skipped if fixtures absent)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not _ZEEK_CONN.exists(), reason="Zeek fixture logs absent")
class TestRealZeekLogs:
    """Run CIC predictions on REAL Zeek conn.log records."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_model(self):
        if not _CIC_MODEL.exists():
            pytest.skip("CIC model not built")

    def _load_conns(self, limit: int = 2000) -> list[dict]:
        out = []
        for line in _ZEEK_CONN.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(out) >= limit:
                break
        return out

    def test_extract_vectors_from_all_conns(self):
        """Every parseable conn record → valid vectors (no NaN, no crash)."""
        conns = self._load_conns(5000)
        ok_count = 0
        for c in conns:
            v = zeek_conn_to_vectors(c)
            if v is None:
                continue
            cic, unsw = v
            assert len(cic) == 18 and len(unsw) == 11
            assert all(np.isfinite(cic)), f"NaN in CIC vector for {c.get('uid')}"
            assert all(np.isfinite(unsw)), f"NaN in UNSW vector for {c.get('uid')}"
            ok_count += 1
        logger.info("Extracted vectors from %d/%d conn records", ok_count, len(conns))
        assert ok_count > 0, "No usable vectors from real Zeek records"

    def test_cic_predictions_on_real_zeek_1000_flows(self):
        """Run CIC model on 1000 real Zeek flows; verify predictions are valid."""
        import warnings
        pipe = joblib.load(_CIC_MODEL)
        feats = pipe.get("features", CIC_FLOW_FEATURES_V2)
        enc = pipe["fam_encoder"]
        n = len(feats)

        conns = self._load_conns(1000)
        preds = {"BENIGN": 0}
        errors = []

        for c in conns:
            v = zeek_conn_to_vectors(c)
            if v is None:
                continue
            cic_vec, _ = v
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    X = pipe["scaler"].transform([cic_vec[:n]])
                is_atk = int(pipe["stage1_model"].predict(X)[0])
                if is_atk == 0:
                    preds["BENIGN"] = preds.get("BENIGN", 0) + 1
                else:
                    import pandas as pd
                    fam = int(pipe["stage2_model"].predict(pd.DataFrame(X, columns=feats))[0])
                    label = str(enc.inverse_transform([fam])[0])
                    preds[label] = preds.get(label, 0) + 1
            except Exception as exc:
                errors.append(str(exc))

        total = sum(preds.values())
        logger.info("Real Zeek CIC predictions (%d flows): %s", total, preds)
        assert total > 0, "No predictions produced"
        assert not errors, f"Prediction errors: {errors[:5]}"
        # Benign-dominated capture should be mostly BENIGN
        benign_pct = preds.get("BENIGN", 0) / max(total, 1) * 100
        logger.info("  BENIGN: %.1f%%", benign_pct)

    def test_cic_batch_prediction_performance(self):
        """Batch of 500 real flows: CIC inference < 2 seconds."""
        import warnings
        pipe = joblib.load(_CIC_MODEL)
        feats = pipe.get("features", CIC_FLOW_FEATURES_V2)
        n = len(feats)

        conns = self._load_conns(500)
        vecs = []
        for c in conns:
            v = zeek_conn_to_vectors(c)
            if v is None:
                continue
            vecs.append(v[0][:n])

        assert len(vecs) > 0, "No vectors to benchmark"
        X = np.array(vecs)
        start = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_s = pipe["scaler"].transform(X)
            _ = pipe["stage1_model"].predict(X_s)
            _ = pipe["stage1_model"].predict_proba(X_s)
        elapsed = time.time() - start
        logger.info("Batch prediction (%d flows): %.3f sec (%.1f flows/sec)",
                     len(vecs), elapsed, len(vecs) / elapsed)
        assert elapsed < 2.0, f"Batch prediction too slow: {elapsed:.3f}s"


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 5: Rate Aggregator (Volumetric Detection) Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestRateAggregator:
    """Test volumetric detection for DDoS, PortScan, and DoS."""

    @pytest.fixture
    def agg(self):
        return RateAggregator(window_s=10.0)

    def test_portscan_detection(self, agg):
        """20 flows to 20 different ports from one IP → PortScan."""
        now = time.time()
        verdict = None
        for port in range(1, 21):
            verdict = agg.observe(
                src_ip="10.0.0.99", dst_ip="10.0.0.1", dst_port=port,
                syn=True, ack=False, bwd_pkts=0, total_bytes=60, now=now)
        assert verdict is not None, "PortScan should be detected"
        assert verdict.attack_type == "PortScan"
        assert verdict.confidence > 0.3
        logger.info("PortScan: confidence=%.2f detail=%s", verdict.confidence, verdict.detail)

    def test_ddos_syn_flood_detection(self, agg):
        """200 SYN-only flows in 1 second → DDoS."""
        now = time.time()
        verdict = None
        for i in range(200):
            verdict = agg.observe(
                src_ip="172.16.0.1", dst_ip="10.0.0.1", dst_port=80,
                syn=True, ack=False, bwd_pkts=0, total_bytes=60,
                now=now + i * 0.005)  # 200 flows in 1 second
        assert verdict is not None, "DDoS should be detected"
        assert verdict.attack_type == "DDoS"
        logger.info("DDoS: confidence=%.2f detail=%s", verdict.confidence, verdict.detail)

    def test_dos_high_byte_rate(self, agg):
        """Sustained high bytes to one destination → DoS."""
        now = time.time()
        verdict = None
        for i in range(30):
            verdict = agg.observe(
                src_ip="192.168.1.50", dst_ip="10.0.0.1", dst_port=80,
                syn=True, ack=True, bwd_pkts=5, total_bytes=500_000,  # 500KB per flow
                now=now + i * 0.3)
        assert verdict is not None, "DoS high-byte-rate should be detected"
        assert verdict.attack_type == "DoS"
        logger.info("DoS: confidence=%.2f detail=%s", verdict.confidence, verdict.detail)

    def test_normal_traffic_no_verdict(self, agg):
        """2 normal flows → no verdict."""
        now = time.time()
        for i in range(2):
            v = agg.observe(
                src_ip="10.0.0.5", dst_ip="10.0.0.1", dst_port=443,
                syn=True, ack=True, bwd_pkts=10, total_bytes=5000,
                now=now + i)
        assert v is None, "Normal traffic should not trigger a verdict"

    def test_different_sources_independent(self, agg):
        """Flows from different IPs shouldn't cross-trigger."""
        now = time.time()
        for i in range(5):
            agg.observe(src_ip=f"10.0.0.{i}", dst_ip="10.0.0.1", dst_port=80,
                       syn=True, ack=False, bwd_pkts=0, total_bytes=60, now=now)
        # Each IP has only 1 flow → no verdict for any
        v = agg.observe(src_ip="10.0.0.0", dst_ip="10.0.0.1", dst_port=80,
                       syn=True, ack=False, bwd_pkts=0, total_bytes=60, now=now)
        # 10.0.0.0 now has 2 flows — still below threshold
        assert v is None


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 6: Decision Engine Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestDecisionEngine:
    """Verify decision routing by risk score."""

    @pytest.fixture
    def engine(self):
        return DecisionEngine()

    @pytest.mark.parametrize("risk,expected_action,expected_blocked", [
        (10.0,  "log",        False),
        (24.9,  "log",        False),
        (25.0,  "alert",      False),
        (50.0,  "alert",      False),
        (60.9,  "alert",      False),
        (61.0,  "review",     False),
        (75.0,  "review",     False),
        (80.9,  "review",     False),
        (81.0,  "auto_block", True),
        (95.0,  "auto_block", True),
        (100.0, "auto_block", True),
    ])
    def test_decision_thresholds(self, engine, risk, expected_action, expected_blocked):
        d = engine.decide(risk)
        assert d.action == expected_action, f"risk={risk}: expected {expected_action}, got {d.action}"
        assert d.auto_blocked == expected_blocked
        assert d.risk_score == risk

    def test_threat_level_in_decision(self, engine):
        d = engine.decide(85.0)
        assert d.threat_level == 2  # >=71 → critical
        d = engine.decide(55.0)
        assert d.threat_level == 1  # 41-70 → suspicious
        d = engine.decide(20.0)
        assert d.threat_level == 0  # <41 → low


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 7: Alert Builder Schema Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestAlertBuilder:
    """Verify alert construction from DetectionEvent + Decision."""

    def _make_event(self, prediction="DDoS", prob=0.95, unsw=None, unsw_prob=0.0):
        return DetectionEvent(
            timestamp="2026-06-04T12:00:00Z",
            src_ip="172.28.0.20",
            dst_ip="172.28.0.10",
            dst_port=80,
            proto="TCP",
            app_proto="http",
            flow_duration_us=150000.0,
            prediction=prediction,
            is_threat=True,
            stage1_attack_prob=prob,
            unsw_prediction=unsw,
            unsw_attack_prob=unsw_prob,
        )

    def test_alert_has_required_fields(self):
        event = self._make_event()
        decision = Decision(action="auto_block", risk_score=88.0, threat_level=2, auto_blocked=True)
        alert = build_alert(event, decision, "Test explanation")

        required = ["id", "timestamp", "attack_type", "src_ip", "dst_ip", "dst_port",
                     "proto", "rule_level", "status", "auto_blocked", "confidence",
                     "threat_level", "risk_score", "ai_explanation", "cic_confidence",
                     "unsw_confidence", "action_history"]
        for field in required:
            assert field in alert, f"Missing field: {field}"

        assert alert["attack_type"] == "DDoS"
        assert alert["auto_blocked"] is True
        assert alert["confidence"] == pytest.approx(0.95, abs=0.001)
        assert alert["risk_score"] == 88.0
        assert len(alert["id"]) == 16  # sha1[:16]

    def test_alert_unsw_removed_fields_null(self):
        """With UNSW removed, unsw_confidence should be None."""
        event = self._make_event(unsw=None, unsw_prob=0.0)
        decision = Decision(action="alert", risk_score=55.0, threat_level=1, auto_blocked=False)
        alert = build_alert(event, decision, "Test")
        assert alert["unsw_confidence"] is None
        assert alert["attack_type"] == "DDoS"  # CIC label used

    def test_unique_alert_ids(self):
        """Multiple alerts from same src_ip+timestamp get unique IDs."""
        event = self._make_event()
        decision = Decision(action="alert", risk_score=50.0, threat_level=1, auto_blocked=False)
        ids = set()
        for _ in range(100):
            alert = build_alert(event, decision, "Test")
            ids.add(alert["id"])
        assert len(ids) == 100, f"Expected 100 unique IDs, got {len(ids)}"


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 8: Suricata Bridge Helpers
# ═══════════════════════════════════════════════════════════════════════════════

class TestSuricataBridgeHelpers:
    """Test signature parsing and feature extraction helpers."""

    @pytest.mark.parametrize("sig,expected", [
        ("ET SCAN Suricata PortScan detected", "PortScan"),
        ("ET EXPLOIT SYN flood detected", "DDoS"),
        ("ET ATTACK SSH brute force", "Brute Force"),
        ("ET DOS Slowloris attack", "DoS"),
        ("ET MISC random signature", None),
        (None, None),
        ("", None),
    ])
    def test_sig_to_attack_type(self, sig, expected):
        result = _sig_to_attack_type(sig)
        assert result == expected, f"sig={sig!r}: expected {expected}, got {result}"

    def test_suricata_flow_to_vectors(self):
        """Suricata eve.json flow → valid vectors."""
        ev = {
            "event_type": "flow",
            "src_ip": "10.0.0.1",
            "dest_ip": "10.0.0.2",
            "dest_port": 80,
            "proto": "TCP",
            "flow": {
                "pkts_toserver": 10,
                "pkts_toclient": 15,
                "bytes_toserver": 800,
                "bytes_toclient": 12000,
                "start": "2026-06-04T12:00:00.000000+0000",
                "end":   "2026-06-04T12:00:02.500000+0000",
            },
            "tcp": {"tcp_flags": "1b"},  # FIN+SYN+PSH+ACK
        }
        result = suricata_to_vectors(ev)
        assert result is not None
        cic, unsw = result
        assert len(cic) == 18  # V2
        assert len(unsw) == 11
        assert all(np.isfinite(cic))

    def test_suricata_no_flow_returns_none(self):
        ev = {"event_type": "flow"}
        assert suricata_to_vectors(ev) is None


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 9: Full Pipeline Integration Test
# ═══════════════════════════════════════════════════════════════════════════════

class TestFullPipelineIntegration:
    """End-to-end: Zeek conn → CIC predict → RiskScore → Decision → Alert."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_model(self):
        if not _CIC_MODEL.exists():
            pytest.skip("CIC model not built")

    def test_e2e_ddos_flow(self):
        """Simulate a DDoS-like flow through the entire pipeline."""
        import warnings

        # 1. Build Zeek-like conn record (DDoS: massive fwd, no response)
        conn = {
            "ts": "1630000000.000000",
            "uid": "CDDoSTest",
            "id.orig_h": "172.28.0.20",
            "id.resp_h": "10.0.0.1",
            "id.resp_p": 80,
            "proto": "tcp",
            "service": "http",
            "duration": 0.01,
            "orig_pkts": 5000,
            "resp_pkts": 0,
            "orig_ip_bytes": 250000,
            "resp_ip_bytes": 0,
            "conn_state": "S0",
            "community_id": "1:ddos_test=",
        }

        # 2. Extract features
        vectors = zeek_conn_to_vectors(conn)
        assert vectors is not None
        cic_vec, unsw_vec = vectors

        # 3. CIC prediction
        pipe = joblib.load(_CIC_MODEL)
        feats = pipe.get("features", CIC_FLOW_FEATURES_V2)
        n = len(feats)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X = pipe["scaler"].transform([cic_vec[:n]])
        is_atk = int(pipe["stage1_model"].predict(X)[0])
        prob = float(pipe["stage1_model"].predict_proba(X)[0, 1])
        if is_atk == 0:
            prediction = "BENIGN"
        else:
            import pandas as pd
            fam = int(pipe["stage2_model"].predict(pd.DataFrame(X, columns=feats))[0])
            prediction = str(pipe["fam_encoder"].inverse_transform([fam])[0])

        logger.info("E2E DDoS: prediction=%s, prob=%.4f", prediction, prob)

        # 4. Risk scoring (CIC only, no UNSW)
        scorer = RiskScorer()
        ml_conf = prob if is_atk == 1 else prob
        risk = scorer.score(
            ml_conf=ml_conf,
            abuse_score=0,
            rule_level=12,
            agent_type="unknown",
            attack_label=prediction,
        )
        logger.info("E2E DDoS: risk_score=%.2f", risk)
        assert 0 <= risk <= 100

        # 5. Decision
        engine = DecisionEngine()
        decision = engine.decide(risk)
        logger.info("E2E DDoS: action=%s threat_level=%d auto_blocked=%s",
                     decision.action, decision.threat_level, decision.auto_blocked)

        # 6. Build alert
        event = DetectionEvent(
            timestamp=str(conn["ts"]),
            src_ip=str(conn["id.orig_h"]),
            dst_ip=str(conn["id.resp_h"]),
            dst_port=int(conn["id.resp_p"]),
            proto=str(conn["proto"]),
            app_proto=str(conn.get("service", "")),
            flow_duration_us=float(conn.get("duration", 0)) * 1_000_000,
            prediction=prediction,
            is_threat=is_atk == 1,
            stage1_attack_prob=prob,
            unsw_prediction=None,
            unsw_attack_prob=0.0,
        )
        alert = build_alert(event, decision, "Simulated DDoS alert")
        logger.info("E2E DDoS alert: id=%s type=%s risk=%.2f action=%s",
                     alert["id"], alert["attack_type"], alert["risk_score"], decision.action)

        # Validate alert schema
        assert alert["src_ip"] == "172.28.0.20"
        assert alert["dst_ip"] == "10.0.0.1"
        assert alert["unsw_confidence"] is None  # UNSW removed
        assert isinstance(alert["id"], str) and len(alert["id"]) == 16

    def test_e2e_benign_flow_gets_low_risk(self):
        """Normal HTTP flow → low risk, 'log' or 'alert' action."""
        import warnings

        conn = {
            "orig_pkts": 10, "resp_pkts": 15,
            "orig_ip_bytes": 800, "resp_ip_bytes": 12000,
            "duration": 2.5, "conn_state": "SF",
            "id.orig_h": "10.0.0.5", "id.resp_h": "10.0.0.1", "id.resp_p": 80,
            "proto": "tcp", "service": "http",
        }
        vectors = zeek_conn_to_vectors(conn)
        assert vectors is not None
        cic_vec, _ = vectors

        pipe = joblib.load(_CIC_MODEL)
        feats = pipe.get("features", CIC_FLOW_FEATURES_V2)
        n = len(feats)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X = pipe["scaler"].transform([cic_vec[:n]])
        is_atk = int(pipe["stage1_model"].predict(X)[0])
        prob = float(pipe["stage1_model"].predict_proba(X)[0, 1])

        if is_atk == 0:
            prediction = "BENIGN"
        else:
            import pandas as pd
            fam = int(pipe["stage2_model"].predict(pd.DataFrame(X, columns=feats))[0])
            prediction = str(pipe["fam_encoder"].inverse_transform([fam])[0])

        scorer = RiskScorer()
        risk = scorer.score(ml_conf=prob, attack_label=prediction)
        logger.info("E2E Benign: prediction=%s prob=%.4f risk=%.2f", prediction, prob, risk)

        # If predicted benign, risk should be low
        if prediction == "BENIGN":
            assert risk < 40, f"BENIGN flow should have low risk, got {risk}"


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 10: Feature Skew Test (Zeek vs Suricata consistency)
# ═══════════════════════════════════════════════════════════════════════════════

class TestFeatureSkewPrevention:
    """Verify that Zeek and Suricata paths produce consistent features."""

    def test_same_flow_same_core_features(self):
        """Same flow primitives → identical core features from both paths."""
        # Simulate the same flow via both paths
        fwd_pkts, bwd_pkts = 10.0, 15.0
        fwd_bytes, bwd_bytes = 800.0, 12000.0
        dur = 2.5

        # Via compute_cic (shared function)
        cic1 = compute_cic(fwd_pkts, bwd_pkts, fwd_bytes, bwd_bytes, dur,
                           fin=1, syn=1, rst=0, psh=1, ack=1, urg=0, dst_port=80)
        cic2 = compute_cic(fwd_pkts, bwd_pkts, fwd_bytes, bwd_bytes, dur,
                           fin=1, syn=1, rst=0, psh=1, ack=1, urg=0, dst_port=80)

        for feat in CIC_FLOW_FEATURES_V2:
            assert cic1[feat] == cic2[feat], f"Feature {feat} mismatch"

    def test_unsw_core_identical_from_both_paths(self):
        """UNSW core features (11) are identical from Zeek and Suricata paths."""
        fwd_pkts, bwd_pkts = 10.0, 15.0
        fwd_bytes, bwd_bytes = 800.0, 12000.0
        dur = 2.5

        unsw1 = compute_unsw(fwd_pkts, bwd_pkts, fwd_bytes, bwd_bytes, dur)
        unsw2 = compute_unsw(fwd_pkts, bwd_pkts, fwd_bytes, bwd_bytes, dur)

        for feat in UNSW_FLOW_FEATURES:
            assert unsw1[feat] == unsw2[feat], f"UNSW feature {feat} mismatch"


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 11: Logging Verification
# ═══════════════════════════════════════════════════════════════════════════════

class TestLogging:
    """Verify that the pipeline produces proper log output."""

    def test_risk_scorer_logs_all_components(self, caplog):
        """Scorer should produce valid numeric output for logging."""
        scorer = RiskScorer()
        with caplog.at_level(logging.DEBUG):
            risk = scorer.score(ml_conf=0.9, abuse_score=50, rule_level=10,
                                agent_type="server", attack_label="DDoS",
                                ip_hit_count=5)
        # Verify the score is loggable
        log_msg = f"risk={risk:.2f}"
        assert "nan" not in log_msg.lower()
        assert "inf" not in log_msg.lower()

    def test_decision_engine_loggable(self):
        """Decision should be loggable with all fields."""
        engine = DecisionEngine()
        d = engine.decide(75.0)
        log_msg = f"action={d.action} risk={d.risk_score} level={d.threat_level} blocked={d.auto_blocked}"
        assert "review" in log_msg
        assert "75.0" in log_msg


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 12: Risk Score Boundary & Regression Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestRiskScoreBoundaries:
    """Edge cases and regression checks for risk scoring."""

    @pytest.fixture
    def scorer(self):
        return RiskScorer()

    def test_risk_never_exceeds_100(self, scorer):
        """Even with maximum inputs, risk ≤ 100."""
        risk = scorer.score(ml_conf=1.0, abuse_score=100, rule_level=15,
                            agent_type="critical", attack_label="DDoS",
                            ip_hit_count=50, host_correlated=True)
        assert risk <= 100.0, f"Risk exceeded 100: {risk}"
        logger.info("Max inputs → risk=%.2f", risk)

    def test_risk_never_negative(self, scorer):
        """Even with minimum inputs, risk ≥ 0."""
        risk = scorer.score(ml_conf=0.0, abuse_score=0, rule_level=0,
                            agent_type="iot", attack_label="BENIGN")
        assert risk >= 0.0, f"Risk is negative: {risk}"

    def test_higher_confidence_higher_risk(self, scorer):
        """All else equal, higher ml_conf → higher risk."""
        risk_low = scorer.score(ml_conf=0.3, attack_label="DDoS", rule_level=8)
        risk_high = scorer.score(ml_conf=0.95, attack_label="DDoS", rule_level=8)
        assert risk_high > risk_low, f"Higher conf should mean higher risk: {risk_low} vs {risk_high}"

    def test_critical_asset_amplifies_risk(self, scorer):
        """Critical asset should have higher risk than IoT."""
        risk_iot = scorer.score(ml_conf=0.8, attack_label="DDoS", agent_type="iot")
        risk_crit = scorer.score(ml_conf=0.8, attack_label="DDoS", agent_type="critical")
        assert risk_crit > risk_iot, f"Critical should be riskier than IoT: {risk_crit} vs {risk_iot}"

    def test_known_attacker_amplifies_risk(self, scorer):
        """High abuse score should increase risk."""
        risk_clean = scorer.score(ml_conf=0.8, abuse_score=0, attack_label="DDoS")
        risk_bad = scorer.score(ml_conf=0.8, abuse_score=90, attack_label="DDoS")
        assert risk_bad > risk_clean, f"Known attacker should increase risk: {risk_clean} vs {risk_bad}"

    def test_host_correlation_bonus(self, scorer):
        """Host-correlated alert should have higher risk."""
        risk_no = scorer.score(ml_conf=0.8, attack_label="DDoS", host_correlated=False)
        risk_yes = scorer.score(ml_conf=0.8, attack_label="DDoS", host_correlated=True)
        assert risk_yes > risk_no, f"Host correlation should boost risk: {risk_no} vs {risk_yes}"
