"""Integration test: run the Zeek path against REAL Zeek output.

Unlike the synthetic skew test, this consumes actual conn.log/http.log lines that
Zeek produced from the CIC-2017 Thursday PCAP (generated with infra/zeek/local.zeek
via the zeek/zeek Docker image). It asserts the bridge can parse real records,
that the custom-policy fields (community_id, sttl/dttl, stcpb/dtcpb) are present,
and that both models produce predictions from Zeek-derived vectors.

The fixture logs live in tests/fixtures/zeek/. If they are absent the test skips
(so CI without Docker still passes); regenerate them with:

    docker run --rm -v "<pcap>:/pcap:ro" -v "<repo>/infra/zeek:/cfg:ro" \
        -v "<out>:/out" zeek/zeek:latest \
        sh -c "cd /out && zeek -C -r /pcap/Thursday-WorkingHours.pcap /cfg/local.zeek"
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from detection.zeek_features import (
    CtWindow, UNSW28_FEATURES, unsw28_from_zeek, zeek_conn_to_vectors,
)

FIX = Path(__file__).parent / "fixtures" / "zeek"
CONN = FIX / "conn.log"
HTTP = FIX / "http.log"

pytestmark = pytest.mark.skipif(
    not CONN.exists(), reason="Zeek fixture logs absent (need Docker to regenerate)")


def _load_conns(limit: int = 2000) -> list[dict]:
    out = []
    for line in CONN.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
        if len(out) >= limit:
            break
    return out


def test_real_zeek_conn_log_is_json_with_required_fields():
    conns = _load_conns(200)
    assert conns, "no conn records parsed"
    d = conns[0]
    for f in ("id.orig_h", "id.resp_p", "proto", "service", "conn_state",
              "orig_pkts", "resp_pkts", "duration", "community_id"):
        assert f in d, f"missing {f} in real Zeek conn.log"


def test_custom_policy_populates_ttl_and_tcp_seq():
    """infra/zeek/local.zeek must add sttl/dttl (all) and stcpb/dtcpb (TCP)."""
    conns = _load_conns(5000)
    assert any("sttl" in c for c in conns), "custom policy did not emit sttl"
    tcp = [c for c in conns if c.get("proto") == "tcp"]
    assert tcp, "no TCP flows in fixture"
    assert all("stcpb" in c for c in tcp), "stcpb missing on some TCP flows"


def test_vectors_from_real_records():
    conns = _load_conns(500)
    n_ok = 0
    for c in conns:
        v = zeek_conn_to_vectors(c)
        if v is None:
            continue
        cic, unsw = v
        assert len(cic) == 18 and len(unsw) == 11
        assert all(np.isfinite(cic)) and all(np.isfinite(unsw))
        n_ok += 1
    assert n_ok > 0, "no usable vectors from real Zeek records"


def test_cic_predicts_on_real_zeek_records():
    """CIC is the sole ML detector (UNSW dropped 2026). The Zeek path must produce
    valid CIC predictions on real Zeek conn.log records."""
    model_path = Path("data/models/cic2017_pipeline_smote.joblib")
    if not model_path.exists():
        pytest.skip("CIC model not built")
    pipe = joblib.load(model_path)
    feats = pipe["features"]
    enc = pipe["fam_encoder"]

    preds = []
    for c in _load_conns(1000):
        v = zeek_conn_to_vectors(c)
        if v is None:
            continue
        cic_vec, _ = v
        X = pipe["scaler"].transform([cic_vec[:len(feats)]])
        if int(pipe["stage1_model"].predict(X)[0]) == 0:
            preds.append("BENIGN")
        else:
            import pandas as pd
            fam = int(pipe["stage2_model"].predict(pd.DataFrame(X, columns=feats))[0])
            preds.append(str(enc.inverse_transform([fam])[0]))
    assert preds, "no CIC predictions produced from real Zeek records"
    # benign-dominated capture: the pipeline must run and mostly score BENIGN
    assert any(p == "BENIGN" for p in preds)


def test_signature_join_by_community_id():
    """Zeek-primary topology: a Suricata signature (keyed by community-id) must
    attach to the Zeek flow with the same community-id (Layer 3), and not to a
    different flow. Does not require the fixture — pure join logic."""
    from detection.suricata_bridge import _sig_to_attack_type

    # Suricata signatures-only reader output: {community_id: signature}
    sig_pending = {
        "1:abcDEF123=": "ET SCAN Suricata PortScan detected",
        "1:zzzOTHER99=": "ET WEB_SERVER SQL Injection",
    }
    # A Zeek conn record carrying a matching community-id.
    conn = {"community_id": "1:abcDEF123=", "id.resp_p": 22, "proto": "tcp"}

    cid = conn["community_id"]
    sig = sig_pending.pop(cid, None)
    assert sig == "ET SCAN Suricata PortScan detected"
    assert _sig_to_attack_type(sig) == "PortScan"      # "portscan" -> PortScan
    # consumed: a second lookup misses (no double-attach)
    assert sig_pending.pop(cid, None) is None
    # an unrelated flow does not pick up the other signature
    other = {"community_id": "1:NOPE000="}
    assert sig_pending.pop(other["community_id"], None) is None


def test_tail_signatures_parses_alert_community_id(tmp_path):
    """tail_signatures() must record {community_id: signature} from alert events
    and ignore non-alert / no-community-id lines."""
    import threading
    import time as _time
    from detection.suricata_bridge import tail_signatures

    eve = tmp_path / "eve.json"
    eve.write_text(
        json.dumps({"event_type": "alert", "community_id": "1:CID1=",
                    "alert": {"signature": "ET EXPLOIT test"}}) + "\n"
        + json.dumps({"event_type": "flow", "community_id": "1:CID2="}) + "\n"
        + json.dumps({"event_type": "alert",
                      "alert": {"signature": "no community id"}}) + "\n",
        encoding="utf-8")

    sig_pending: dict[str, str] = {}
    t = threading.Thread(target=tail_signatures, args=(eve, sig_pending), daemon=True)
    t.start()
    # tail_signatures seeks to EOF then reads new lines; append after start.
    _time.sleep(0.3)
    with eve.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"event_type": "alert", "community_id": "1:CID3=",
                             "alert": {"signature": "ET SCAN nmap"}}) + "\n")
        fh.flush()
    _time.sleep(0.5)
    assert sig_pending.get("1:CID3=") == "ET SCAN nmap"
    assert "1:CID2=" not in sig_pending     # flow event ignored
