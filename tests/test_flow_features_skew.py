"""Training-serving skew guard for the V2 CIC feature set.

The single-source-of-truth promise of detection/flow_features.py is that a flow
produces the *same* feature vector whether it arrives as a CIC CSV row (training)
or a Suricata eve.json flow record (serving). This test constructs one flow in
both representations and asserts the resulting CIC_FLOW_FEATURES_V2 vectors are
identical — so adding dst_port did not open a skew gap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from detection.flow_features import (
    CIC_FLOW_FEATURES,
    CIC_FLOW_FEATURES_V2,
    cic_features_from_df,
    suricata_to_vectors,
)


def test_v2_appends_dst_port_only():
    assert CIC_FLOW_FEATURES_V2[:len(CIC_FLOW_FEATURES)] == CIC_FLOW_FEATURES
    assert CIC_FLOW_FEATURES_V2[-1] == "dst_port"
    assert len(CIC_FLOW_FEATURES_V2) == len(CIC_FLOW_FEATURES) + 1


def test_csv_and_suricata_produce_identical_v2_vectors():
    # One flow: 10 fwd pkts / 1200 B, 8 bwd pkts / 9000 B, 3 s, SYN+ACK+PSH, port 80.
    fwd_pkts, bwd_pkts = 10, 8
    fwd_bytes, bwd_bytes = 1200, 9000
    dur_s, dst_port = 3.0, 80

    # CSV/training representation (CICFlowMeter duration is microseconds).
    df = pd.DataFrame([{
        "Destination Port": dst_port,
        "Flow Duration": dur_s * 1e6,
        "Total Fwd Packets": fwd_pkts,
        "Total Backward Packets": bwd_pkts,
        "Total Length of Fwd Packets": fwd_bytes,
        "Total Length of Bwd Packets": bwd_bytes,
        "FIN Flag Count": 0, "SYN Flag Count": 1, "RST Flag Count": 0,
        "PSH Flag Count": 1, "ACK Flag Count": 1, "URG Flag Count": 0,
    }])
    csv_vec = cic_features_from_df(df, version="v2").values[0]

    # Serving representation (eve.json flow record).
    ev = {
        "dest_port": dst_port,
        "flow": {
            "pkts_toserver": fwd_pkts, "pkts_toclient": bwd_pkts,
            "bytes_toserver": fwd_bytes, "bytes_toclient": bwd_bytes,
            "age": dur_s,
        },
        "tcp": {"syn": True, "ack": True, "psh": True},
    }
    cic_vec, _ = suricata_to_vectors(ev)

    assert len(cic_vec) == len(CIC_FLOW_FEATURES_V2)
    np.testing.assert_allclose(np.asarray(cic_vec), csv_vec, rtol=1e-9, atol=1e-9)
    # dst_port carried through both paths
    assert cic_vec[-1] == float(dst_port)


def test_zeek_and_suricata_produce_identical_core_vectors():
    """The second sensor (Zeek) must agree with Suricata on the shared core/CIC
    vector for an equivalent flow — no cross-sensor skew."""
    from detection.zeek_features import zeek_conn_to_vectors

    ev = {
        "dest_port": 80,
        "flow": {"pkts_toserver": 10, "pkts_toclient": 8,
                 "bytes_toserver": 1200, "bytes_toclient": 9000, "age": 3.0},
        "tcp": {"syn": True, "ack": True, "psh": True, "fin": True},
    }
    # Zeek conn_state "SF" == a normally established+terminated TCP conn
    # (SYN+ACK+PSH+FIN), the same flag picture as the Suricata event above.
    conn = {
        "orig_pkts": 10, "resp_pkts": 8,
        "orig_ip_bytes": 1200, "resp_ip_bytes": 9000,
        "duration": 3.0, "id.resp_p": 80, "conn_state": "SF",
    }
    s_cic, s_unsw = suricata_to_vectors(ev)
    z_cic, z_unsw = zeek_conn_to_vectors(conn)
    np.testing.assert_allclose(z_cic, s_cic, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(z_unsw, s_unsw, rtol=1e-9, atol=1e-9)
