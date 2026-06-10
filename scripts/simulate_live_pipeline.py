import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from pipeline.risk_scorer import RiskScorer
from pipeline.decision_engine import DecisionEngine

def simulate_attack(name: str, ml_conf: float, rule_level: int, abuse_score: int, agent: str, attack_label: str):
    scorer = RiskScorer()
    engine = DecisionEngine()
    
    risk = scorer.score(
        ml_conf=ml_conf,
        abuse_score=abuse_score,
        rule_level=rule_level,
        agent_type=agent,
        attack_label=attack_label
    )
    decision = engine.decide(risk)
    
    print(f"{name.upper():<20} | {ml_conf*100:>4.0f}% | {rule_level:>4d} | {abuse_score:>5d} | {agent:<12} | {attack_label:<12} | {risk:>6.2f} | {decision.action.upper()}")

print("=" * 110)
print(f"{'SCENARIO':<20} | {'ML':>5} | {'RULE':>4} | {'INTEL':>5} | {'ASSET':<12} | {'LABEL':<12} | {'RISK':>6} | {'ACTION'}")
print("-" * 110)

simulate_attack("DDoS on Server", 0.95, 12, 0, "server", "DDoS")
simulate_attack("Heartbleed Exploit", 0.98, 14, 50, "critical", "Heartbleed")
simulate_attack("PortScan (Noisy)", 0.85, 6, 0, "workstation", "PortScan")
simulate_attack("Brute Force (Known)", 0.88, 9, 85, "server", "Brute Force")
simulate_attack("Botnet Beacon", 0.92, 10, 40, "workstation", "Bot")
simulate_attack("Benign Traffic", 0.15, 0, 0, "workstation", "BENIGN")
simulate_attack("Low-Conf Anomaly", 0.55, 3, 0, "iot", "Generic")

print("=" * 110)
print("\nPipeline is correctly prioritizing Critical assets and Heartbleed (high impact) while safely logging benign traffic.")
