"""Risk scoring engine — NIST SP 800-30 & Hybrid NIDS formula for Lighthouse SOC platform.

Design basis:
  - NIST SP 800-30 Rev. 1: Risk = Likelihood × Impact
  - Hybrid NIDS: Weighted confidence voting (ML, Rules, Threat Intel)
  - Impact axis: Asset Criticality × Threat Severity
  - IoC shelf-life research (arXiv:2307.16852): Exponential temporal decay

References:
  NIST SP 800-30 Rev. 1    https://csrc.nist.gov/pubs/sp/800/30/r1/final
  CVSS v4                  https://nvd.nist.gov/vuln-metrics/cvss/v4-calculator
  Splunk RBA               https://lantern.splunk.com/Security/Implementing_risk-based_alerting
  IoC shelf-life           https://arxiv.org/pdf/2307.16852
"""

from __future__ import annotations

import math
import time

# ── Asset criticality — multiplicative (NIST 800-30 impact axis) ─────────────
# Range [0.5, 1.5]: critical server boosts risk 50%, IoT sensor reduces it 50%.
_ASSET_MULTIPLIERS: dict[str, float] = {
    "domain_controller": 1.5,
    "critical":          1.5,
    "server":            1.4,
    "database":          1.4,
    "high":              1.3,
    "workstation":       1.0,
    "medium":            1.0,
    "unknown":           1.0,   # neutral — an unknown asset under attack is NOT down-weighted
    "low":               0.85,
    "iot":               0.7,
}

# ── Attack-type severity tier (CVSS impact axis) ──────────────────────────────
# Maps prediction labels to a multiplier (0.5x to 1.2x).
# Low severity (recon) = 0.5x. High severity (DDoS/Bot) = 1.0x. Critical (Shellcode/Heartbleed) = 1.2x.
_ATTACK_SEVERITY_MULTIPLIER: dict[str, float] = {
    # Critical exploits / compromises (1.2x)
    "Heartbleed":       1.20,
    "Backdoor":         1.20,
    "Shellcode":        1.20,
    "Exploits":         1.20,
    "Worms":            1.20,
    
    # High severity / destructive (1.0x)
    "DDoS":             1.00,
    "DoS":              1.00,
    "DoS Hulk":         1.00,
    "DoS GoldenEye":    1.00,
    "Infiltration":     1.00,
    "Bot":              1.00,
    "Web Attack":       1.00,

    # Moderate severity (0.8x)
    "DoS slowloris":    0.80,
    "DoS Slowhttptest": 0.80,
    "Brute Force":      0.80,
    "FTP-Patator":      0.80,
    "SSH-Patator":      0.80,
    "Generic":          0.80,

    # Low severity / reconnaissance (0.5x)
    "PortScan":         0.50,
    "Reconnaissance":   0.50,
    "Analysis":         0.50,
    "Fuzzers":          0.50,
    
    # Benign (0.5x multiplier ensures low scores don't drop exactly to 0 but stay minimal)
    "BENIGN":           0.50,
    "Normal":           0.50,
}

_BENIGN_LABELS = {"BENIGN", "Normal", ""}


class RiskScorer:
    """
    NIST SP 800-30 aligned risk scorer returning a 0–100 score.

    Formula:
        Likelihood = (ML_conf * W_ML) + (Rule_conf * W_RULE) + (Intel_conf * W_INTEL)
        Impact = Asset_Multiplier * Severity_Multiplier
        Risk = clamp(Likelihood * Impact * Temporal_Factor, 0, 100)

    Weights (sum to 1.0):
        W_ML       = 0.50  Primary ML anomaly/zero-day signal
        W_RULE     = 0.30  Deterministic signatures (Wazuh/Suricata)
        W_INTEL    = 0.20  Threat intelligence (AbuseIPDB reputation)
    """

    W_ML       = 0.50
    W_RULE     = 0.30
    W_INTEL    = 0.20

    def score(
        self,
        ml_conf: float,
        abuse_score: int = 0,
        rule_level: int = 0,
        agent_type: str = "unknown",
        attack_label: str = "",
        ip_hit_count: int = 1,
        last_seen_ts: float = 0.0,
        host_correlated: bool = False,
    ) -> float:

        # 1. Likelihood (0–100 base)
        # ML signal
        ml_component = ml_conf * 100.0
        
        # Behavioral / rule signal (normalized 0-15 scale to 0-100)
        rule_component = (min(rule_level, 15) / 15.0) * 100.0

        # Threat intelligence signal
        intel_component = min(abuse_score, 100.0)

        base_likelihood = (
            ml_component     * self.W_ML
            + rule_component   * self.W_RULE
            + intel_component  * self.W_INTEL
        )

        # Host+network correlation: when a Wazuh host alert and a network alert
        # share the same source IP, that agreement is a strong signal. Add fixed bonus.
        if host_correlated:
            base_likelihood += 10.0

        # 2. Impact Factor
        # Asset multiplier — NIST 800-30 impact axis
        asset_mult = _ASSET_MULTIPLIERS.get(agent_type.lower(), 1.0)
        
        # Threat severity multiplier
        severity_mult = _ATTACK_SEVERITY_MULTIPLIER.get(attack_label, 0.5 if attack_label in _BENIGN_LABELS else 0.8)

        impact_factor = asset_mult * severity_mult

        # 3. Temporal amplification — IoC shelf-life decay + repeat-attacker boost
        temporal = self._temporal_factor(ip_hit_count, last_seen_ts)

        # 4. Final Risk Score
        risk = base_likelihood * impact_factor * temporal
        
        return round(min(max(risk, 0.0), 100.0), 2)

    def _temporal_factor(self, hit_count: int, last_seen_ts: float) -> float:
        """Multiplier in [0.8, 1.4].

        Recency decay (IoC shelf-life paper arXiv:2307.16852):
          - IP seen > 30 days ago: 0.8x (likely reassigned, suppress)
          - IP seen 7–30 days ago: 0.9x (stale)
          - Recent / first seen:   1.0x

        Repeat-attacker amplification (Splunk RBA cumulative risk):
          - hit_count=1  → 1.0x
          - hit_count=5  → ~1.24x
          - hit_count=20 → ~1.4x (capped)
        """
        if last_seen_ts > 0:
            age_days = (time.time() - last_seen_ts) / 86400
            recency = 0.8 if age_days > 30 else (0.9 if age_days > 7 else 1.0)
        else:
            recency = 1.0

        repeat_amp = min(1.0 + math.log(max(hit_count, 1)) * 0.15, 1.4)
        return recency * repeat_amp

    def threat_level(self, score: float) -> int:
        if score >= 71:
            return 2  # critical
        if score >= 41:
            return 1  # suspicious
        return 0
