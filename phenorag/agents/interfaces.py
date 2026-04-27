"""
phenorag/agents/interfaces.py

Shared dataclasses: Agent1 -> Agent2 -> Agent3.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ── Agent 1 output ──────────────────────────────────────────────

@dataclass
class StayFacts:
    stay_id: str
    patient_id: str
    visit_date: str
    visit_number: int
    disease_mentions: List[Dict[str, Any]]
    labs: List[Dict[str, Any]]
    drugs: List[Dict[str, Any]]
    _rag_meta: Dict[str, Any] = field(default_factory=dict)
    _timing: Dict[str, Any] = field(default_factory=dict)
    _extraction_mode: str = "llm"


# ── Agent 2: dimension-level result ─────────────────────────────

@dataclass
class DimensionResult:
    """Result for one ACR/EULAR dimension (joint, serology, acute, duration)."""
    points: Optional[int]          # None = missing evidence
    status: str                    # "positive" | "negative" | "missing"
    reason: str


# ── Agent 2 output ──────────────────────────────────────────────

@dataclass
class StayAssessment:
    stay_id: str
    patient_id: str

    # RA-relatedness
    is_ra_related: bool
    ra_related_confidence: float
    ra_related_reasoning: str

    # ACR/EULAR scoring
    acr_eular_deterministic: Optional[Dict[str, Any]]
    acr_eular_llm: Optional[Dict[str, Any]]
    acr_eular_final_score: int

    # Evidence completeness (NEW)
    completeness: float = 0.0      # 0.0 - 1.0 (known_dims / 4)
    missing_dimensions: List[str] = field(default_factory=list)
    strong_ra_signals: List[str] = field(default_factory=list)
    strong_negative_signals: List[str] = field(default_factory=list)

    # Decision
    final_stay_label: str = "RA-"             # "RA+" | "RA-" | "UNCERTAIN"
    internal_label: str = "RA-"               # before forcing binary
    confidence: float = 0.5
    reasoning_trace: List[str] = field(default_factory=list)
    extracted_facts: Dict[str, Any] = field(default_factory=dict)

    # Probabilistic output (from EncounterClassifier)
    class_probabilities: Dict[str, float] = field(default_factory=dict)
    predicted_class: str = ""                 # PR_ABSENT, PR_LATENT, etc.
    pr_positive_probability: float = 0.0      # P(any PR state)


# ── Agent 3 output ──────────────────────────────────────────────

@dataclass
class PatientDecision:
    patient_id: str
    n_stays: int
    stay_labels: List[str]
    stay_confidences: List[float]
    n_ra_related_stays: int
    n_ra_positive_stays: int
    n_ra_negative_stays: int

    aggregation_strategy: str
    aggregation_label: str
    aggregation_confidence: float
    aggregation_reasoning: str

    final_label: str
    final_confidence: float
    decision_source: str

    guardrails_passed: bool
    guardrails_warnings: List[str]
    reasoning_trace: List[str]
    stay_details: List[Dict[str, Any]]

    llm_critic_used: bool = False
    llm_critic_recommendation: Optional[str] = None
    llm_critic_confidence: Optional[float] = None
    llm_critic_reasoning: Optional[str] = None
    llm_override_applied: bool = False

    # Probabilistic patient-level output
    patient_class_probabilities: Dict[str, float] = field(default_factory=dict)
    patient_predicted_class: str = ""
    patient_pr_positive_probability: float = 0.0
    patient_temporal_trend: str = "stable"
