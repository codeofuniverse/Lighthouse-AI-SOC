"""Decision engine — routes alerts by risk score to the appropriate action."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pipeline.risk_scorer import RiskScorer

_scorer = RiskScorer()


@dataclass(slots=True)
class Decision:
    action: Literal["log", "alert", "review", "auto_block"]
    risk_score: float
    threat_level: int
    auto_blocked: bool


class DecisionEngine:
    def decide(self, risk_score: float) -> Decision:
        threat_level = _scorer.threat_level(risk_score)
        if risk_score >= 81:
            return Decision(action="auto_block", risk_score=risk_score, threat_level=threat_level, auto_blocked=True)
        if risk_score >= 61:
            return Decision(action="review",     risk_score=risk_score, threat_level=threat_level, auto_blocked=False)
        if risk_score >= 25:
            return Decision(action="alert",      risk_score=risk_score, threat_level=threat_level, auto_blocked=False)
        return     Decision(action="log",        risk_score=risk_score, threat_level=threat_level, auto_blocked=False)
