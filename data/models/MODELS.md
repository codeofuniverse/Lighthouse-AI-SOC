# Lighthouse model manifest

Final models after the feature-discovery + sensor + recalibration work.

| File | Dataset | Features | Sensor | Role |
|---|---|---|---|---|
| `cic2017_pipeline_smote.joblib` | CIC-2017 | 18 (flow + dst_port) | Suricata or Zeek | **Canonical CIC** (production). 7-class two-stage. Web-context aware; ~70% lower benign FP than the legacy 17-feature model. |
| `cic2017_webattack_v3_http.joblib` | CIC-2017 PCAP | 22 (flow + HTTP) | Zeek `http.log` | Focused BENIGN-vs-WebAttack binary. ~97% recall. The IAT-tier (offline) reaches 98%/0.05% FP but needs per-packet timing. |
| `unsw_nb15_pipeline.joblib` | UNSW-NB15 | 11 (core flow) | Suricata | UNSW fallback for Suricata-only deployments. |
| `unsw_kfold_pipeline.joblib` | UNSW-NB15 | 28 (SHAP-discovered) | Zeek (+ custom policy) | **High-accuracy UNSW**, **recalibrated on real benign traffic**. Shellcode ~95%, Worms ~89% recall; 98.5% attack recall; real-benign Normal rate 4.5%→100% after recalibration. |

`*.bak` files are the pre-promotion / pre-recalibration archives.

## Serving

- Suricata-only: CIC canonical (18f) + UNSW fallback (11f). Default `docker compose up`.
- + Zeek (`LIGHTHOUSE_ZEEK=1`): adds the 28-feature UNSW model and real Web-Attack
  HTTP context. Zeek emits the missing features (`proto/service/sttl/ct_*` + HTTP);
  the custom policy `infra/zeek/local.zeek` adds TTL/TCP-seq for the full 28.

## Recalibration note

The UNSW-28 model's `Normal` class was learned from UNSW-NB15's synthetic 2015 lab
benign and over-flagged real traffic. `scripts/recalibrate_unsw.py` blends real
benign flows (Zeek over the CIC PCAP, attack 5-tuples excluded) into the Normal
class and retrains. Before deploying on a NEW network, re-run recalibration on that
network's own benign capture for best results.
