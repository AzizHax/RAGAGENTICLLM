"""
phenorag/agents/probabilistic.py

Bayesian Probabilistic Phenotyping for Rheumatoid Arthritis

Two layers:
  Layer 1 (encounter-level):
    P(class | joints, serology, acute_phase, duration, signals)
    Using naive Bayes with literature-based priors

  Layer 2 (patient-level):
    P(patient_phenotype | stay_1, stay_2, ..., stay_T)
    Bayesian aggregation across stays

5 classes:
  0 = PR_ABSENT      : No rheumatoid arthritis
  1 = PR_LATENT      : Suspected/early, insufficient criteria
  2 = PR_REMISSION   : Known PR under treatment, low activity
  3 = PR_MODERATE    : Active PR, moderate disease activity
  4 = PR_SEVERE      : Active PR, high disease activity

References:
  - Aletaha D, et al. 2010 ACR/EULAR classification criteria (score 0-10)
  - Vidmar et al. 2024 — Probabilistic phenotyping from EHR
  - Prevalence/sensitivity data from rheumatology literature
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ════════════════════════════════════════════════════════════════
# CLASSES
# ════════════════════════════════════════════════════════════════

CLASSES = ["PR_ABSENT", "PR_LATENT", "PR_REMISSION", "PR_MODERATE", "PR_SEVERE"]
N_CLASSES = len(CLASSES)


@dataclass
class ProbabilisticOutput:
    """Output of the probabilistic model for one stay."""
    class_probabilities: Dict[str, float]       # {class_name: probability}
    predicted_class: str                         # argmax class
    pr_positive_probability: float               # P(any PR) = 1 - P(absent)
    confidence: float                            # max probability
    acr_score: int                               # raw ACR score used
    completeness: float                          # data completeness


@dataclass
class PatientProbabilisticOutput:
    """Aggregated patient-level probabilistic output."""
    class_probabilities: Dict[str, float]
    predicted_class: str
    pr_positive_probability: float
    confidence: float
    stay_trajectories: List[Dict[str, float]]    # per-stay class probs
    temporal_trend: str                           # "stable", "improving", "worsening"


# ════════════════════════════════════════════════════════════════
# LITERATURE-BASED PRIORS
# ════════════════════════════════════════════════════════════════
#
# These are P(class) — prior probability of each state
# in a rheumatology consultation population.
#
# Source rationale:
#   - ~40% of referrals to rheumatology do NOT have RA
#   - ~15% are early/suspected (not yet classifiable)
#   - ~15% are in remission under treatment
#   - ~20% have moderate active disease
#   - ~10% have severe/refractory disease

PRIOR = np.array([0.40, 0.15, 0.15, 0.20, 0.10])


# ════════════════════════════════════════════════════════════════
# LIKELIHOOD TABLES
# ════════════════════════════════════════════════════════════════
#
# P(observation | class) for each ACR dimension and clinical signal.
#
# Format: likelihood[feature_name][feature_value] = [P(val|absent), P(val|latent),
#                                                     P(val|remission), P(val|moderate),
#                                                     P(val|severe)]
#
# Sources: ACR/EULAR 2010, clinical rheumatology textbooks,
#          sensitivity/specificity of RF, anti-CCP, CRP from meta-analyses.

LIKELIHOOD = {
    # ── JOINTS (A) ──────────────────────────────────────────
    # Points:  0     1      2      3      5
    "joint": {
        "missing":  [0.50, 0.40, 0.30, 0.15, 0.05],  # no joint data
        0:          [0.80, 0.30, 0.50, 0.05, 0.02],  # no joint involvement
        1:          [0.10, 0.20, 0.20, 0.15, 0.05],  # 2-10 large joints
        2:          [0.05, 0.25, 0.15, 0.25, 0.10],  # 1-3 small joints
        3:          [0.03, 0.15, 0.10, 0.35, 0.30],  # 4-10 small joints
        5:          [0.02, 0.10, 0.05, 0.20, 0.53],  # >10 joints
    },

    # ── SEROLOGY (B) ────────────────────────────────────────
    # Points:  0     2      3
    "serology": {
        "missing":  [0.40, 0.35, 0.30, 0.20, 0.15],  # no serology data
        0:          [0.90, 0.40, 0.30, 0.20, 0.15],  # RF- and CCP-
        2:          [0.05, 0.30, 0.30, 0.35, 0.25],  # low positive
        3:          [0.02, 0.20, 0.25, 0.40, 0.55],  # high positive (>3x ULN)
    },

    # ── ACUTE PHASE (C) ────────────────────────────────────
    "acute_phase": {
        "missing":  [0.35, 0.30, 0.40, 0.20, 0.10],  # no CRP/ESR data
        0:          [0.85, 0.50, 0.70, 0.20, 0.05],  # normal
        1:          [0.15, 0.50, 0.30, 0.80, 0.95],  # elevated
    },

    # ── DURATION (D) ───────────────────────────────────────
    "duration": {
        "missing":  [0.40, 0.40, 0.20, 0.20, 0.15],  # unknown
        0:          [0.70, 0.50, 0.10, 0.10, 0.05],  # <6 weeks
        1:          [0.15, 0.40, 0.85, 0.85, 0.90],  # >=6 weeks
    },

    # ── DMARD PRESENT ──────────────────────────────────────
    "has_dmard": {
        False:      [0.95, 0.70, 0.10, 0.15, 0.10],
        True:       [0.02, 0.15, 0.85, 0.80, 0.85],
    },

    # ── BIOLOGIC PRESENT ───────────────────────────────────
    "has_biologic": {
        False:      [0.98, 0.90, 0.50, 0.60, 0.30],
        True:       [0.01, 0.05, 0.40, 0.35, 0.65],
    },

    # ── PR MENTION ─────────────────────────────────────────
    "pr_mentioned": {
        "none":     [0.85, 0.30, 0.05, 0.02, 0.01],
        "mentioned":[0.10, 0.40, 0.20, 0.15, 0.10],
        "confirmed":[0.02, 0.15, 0.60, 0.70, 0.75],
        "negated":  [0.90, 0.10, 0.02, 0.01, 0.01],
    },

    # ── PR EXPLICITLY NEGATED ──────────────────────────────
    "pr_negated": {
        False:      [0.50, 0.80, 0.95, 0.98, 0.99],
        True:       [0.95, 0.10, 0.02, 0.01, 0.01],
    },
}


# ════════════════════════════════════════════════════════════════
# LAYER 1: ENCOUNTER-LEVEL BAYESIAN CLASSIFIER
# ════════════════════════════════════════════════════════════════

class EncounterClassifier:
    """
    Naive Bayes classifier for stay-level PR phenotyping.

    P(class | features) ∝ P(class) × ∏ P(feature_i | class)
    """

    def __init__(self, prior: Optional[np.ndarray] = None):
        self.prior = prior if prior is not None else PRIOR
        self.likelihood = LIKELIHOOD

    def classify(self, features: Dict[str, Any]) -> ProbabilisticOutput:
        """
        Classify a single stay.

        Args:
            features: extracted from Agent2 dimensions. Keys:
                - joint: int (0,1,2,3,5) or "missing"
                - serology: int (0,2,3) or "missing"
                - acute_phase: int (0,1) or "missing"
                - duration: int (0,1) or "missing"
                - has_dmard: bool
                - has_biologic: bool
                - pr_mentioned: "none" | "mentioned" | "confirmed" | "negated"
                - pr_negated: bool
                - acr_score: int
                - completeness: float
        """
        # Start with log-prior
        log_posterior = np.log(self.prior + 1e-10)

        # Multiply likelihoods for each observed feature
        for feat_name, feat_value in features.items():
            if feat_name in ("acr_score", "completeness"):
                continue  # metadata, not a likelihood feature

            table = self.likelihood.get(feat_name)
            if table is None:
                continue

            likelihoods = table.get(feat_value)
            if likelihoods is None:
                # Unknown value — skip (uniform likelihood)
                continue

            log_posterior += np.log(np.array(likelihoods) + 1e-10)

        # Normalize (log-sum-exp trick for numerical stability)
        log_posterior -= np.max(log_posterior)
        posterior = np.exp(log_posterior)
        posterior /= posterior.sum()

        # Build output
        class_probs = {CLASSES[i]: float(posterior[i]) for i in range(N_CLASSES)}
        predicted = CLASSES[int(np.argmax(posterior))]
        pr_positive = 1.0 - float(posterior[0])  # 1 - P(absent)

        return ProbabilisticOutput(
            class_probabilities=class_probs,
            predicted_class=predicted,
            pr_positive_probability=round(pr_positive, 4),
            confidence=round(float(np.max(posterior)), 4),
            acr_score=features.get("acr_score", 0),
            completeness=features.get("completeness", 0.0),
        )

    def extract_features(self, stay_assessment: Dict) -> Dict[str, Any]:
        """
        Extract features from a StayAssessment dict (Agent2 output).
        Maps Agent2's dimension results to the feature format expected
        by the classifier.
        """
        det = stay_assessment.get("acr_eular_deterministic") or {}
        dim_status = det.get("dimension_status", {})
        facts = stay_assessment.get("extracted_facts", {})

        # Joint points
        joint_pts = det.get("joint_involvement", 0)
        joint_status = dim_status.get("joint", "missing")
        joint = "missing" if joint_status == "missing" else joint_pts

        # Serology points
        sero_pts = det.get("serology", 0)
        sero_status = dim_status.get("serology", "missing")
        serology = "missing" if sero_status == "missing" else sero_pts

        # Acute phase
        acute_pts = det.get("acute_phase", 0)
        acute_status = dim_status.get("acute_phase", "missing")
        acute_phase = "missing" if acute_status == "missing" else acute_pts

        # Duration
        dur_pts = det.get("duration", 0)
        dur_status = dim_status.get("duration", "missing")
        duration = "missing" if dur_status == "missing" else dur_pts

        # DMARD / Biologic
        drugs = facts.get("drugs") or []
        has_dmard = any(
            isinstance(d, dict) and (d.get("category") or "") in ("csDMARD", "bDMARD", "tsDMARD")
            for d in drugs)
        has_biologic = any(
            isinstance(d, dict) and (d.get("category") or "") in ("bDMARD", "tsDMARD")
            for d in drugs)

        # PR mention status
        dm = facts.get("disease_mentions") or []
        pr_status = "none"
        pr_negated = False
        for d in dm:
            if not isinstance(d, dict):
                continue
            entity = (d.get("entity") or "").lower()
            status = (d.get("status") or "").lower()
            if "rhumato" in entity or "rheumatoid" in entity:
                if status == "negated":
                    pr_negated = True
                    pr_status = "negated"
                elif status in ("confirmed", "suspected"):
                    pr_status = "confirmed"
                elif status == "mentioned":
                    if pr_status == "none":
                        pr_status = "mentioned"

        return {
            "joint": joint,
            "serology": serology,
            "acute_phase": acute_phase,
            "duration": duration,
            "has_dmard": has_dmard,
            "has_biologic": has_biologic,
            "pr_mentioned": pr_status,
            "pr_negated": pr_negated,
            "acr_score": det.get("total", 0),
            "completeness": stay_assessment.get("completeness", 0.0),
        }


# ════════════════════════════════════════════════════════════════
# LAYER 2: PATIENT-LEVEL BAYESIAN AGGREGATION
# ════════════════════════════════════════════════════════════════

class PatientAggregator:
    """
    Aggregate stay-level probabilities into patient-level phenotype.

    Strategy: Bayesian update across stays.
    P(patient_class) ∝ P(prior) × ∏_t P(class | stay_t)

    With temporal weighting: recent stays count more.

    Also detects temporal trends:
      - Improving: P(remission) increasing over time
      - Worsening: P(severe) increasing over time
      - Stable: no significant trend
    """

    def __init__(self, prior: Optional[np.ndarray] = None,
                 temporal_decay: float = 0.9):
        """
        Args:
            prior: patient-level prior (default: same as encounter)
            temporal_decay: weight multiplier per stay going back in time.
                            1.0 = all stays equal, 0.8 = older stays down-weighted.
        """
        self.prior = prior if prior is not None else PRIOR
        self.temporal_decay = temporal_decay

    def aggregate(self, stay_outputs: List[ProbabilisticOutput]) -> PatientProbabilisticOutput:
        """
        Aggregate multiple stay-level outputs into patient-level decision.
        Stays should be in chronological order.
        """
        if not stay_outputs:
            # No stays — return prior
            probs = {CLASSES[i]: float(self.prior[i]) for i in range(N_CLASSES)}
            return PatientProbabilisticOutput(
                class_probabilities=probs,
                predicted_class="PR_ABSENT",
                pr_positive_probability=1.0 - float(self.prior[0]),
                confidence=float(np.max(self.prior)),
                stay_trajectories=[],
                temporal_trend="stable",
            )

        n_stays = len(stay_outputs)

        # Bayesian aggregation with temporal weighting
        log_posterior = np.log(self.prior + 1e-10)

        stay_trajectories = []
        for t, stay in enumerate(stay_outputs):
            # Weight: more recent stays (later index) get higher weight
            weight = self.temporal_decay ** (n_stays - 1 - t)

            # Convert stay probs to array
            stay_probs = np.array([
                stay.class_probabilities.get(c, 0.0) for c in CLASSES
            ])

            # Bayesian update (weighted log-likelihood)
            # P(class | all_stays) ∝ prior × ∏ P(stay_t | class)^weight
            # We use stay posteriors as approximate likelihoods
            # (dividing out the prior to avoid double-counting)
            likelihood_ratio = (stay_probs + 1e-10) / (self.prior + 1e-10)
            log_posterior += weight * np.log(likelihood_ratio + 1e-10)

            stay_trajectories.append(stay.class_probabilities)

        # Normalize
        log_posterior -= np.max(log_posterior)
        posterior = np.exp(log_posterior)
        posterior /= posterior.sum()

        class_probs = {CLASSES[i]: round(float(posterior[i]), 4) for i in range(N_CLASSES)}
        predicted = CLASSES[int(np.argmax(posterior))]
        pr_positive = round(1.0 - float(posterior[0]), 4)

        # Detect temporal trend
        trend = self._detect_trend(stay_trajectories)

        return PatientProbabilisticOutput(
            class_probabilities=class_probs,
            predicted_class=predicted,
            pr_positive_probability=pr_positive,
            confidence=round(float(np.max(posterior)), 4),
            stay_trajectories=stay_trajectories,
            temporal_trend=trend,
        )

    @staticmethod
    def _detect_trend(trajectories: List[Dict[str, float]]) -> str:
        """Detect if patient is improving, worsening, or stable."""
        if len(trajectories) < 2:
            return "stable"

        # Track P(severe) + P(moderate) over time = "activity"
        activity = []
        for t in trajectories:
            act = t.get("PR_MODERATE", 0) + t.get("PR_SEVERE", 0)
            activity.append(act)

        # Simple trend: compare first half vs second half
        mid = len(activity) // 2
        if mid == 0:
            return "stable"

        first_half = sum(activity[:mid]) / mid
        second_half = sum(activity[mid:]) / (len(activity) - mid)

        diff = second_half - first_half
        if diff > 0.15:
            return "worsening"
        elif diff < -0.15:
            return "improving"
        return "stable"


# ════════════════════════════════════════════════════════════════
# CONVENIENCE: map probabilistic output to binary label
# ════════════════════════════════════════════════════════════════

def prob_to_label(prob_output: ProbabilisticOutput,
                  threshold: float = 0.50) -> Tuple[str, float]:
    """
    Convert probabilistic output to binary RA+/RA- label.

    Args:
        prob_output: from EncounterClassifier
        threshold: P(PR+) >= threshold -> RA+

    Returns:
        (label, confidence)
    """
    if prob_output.pr_positive_probability >= threshold:
        return "RA+", prob_output.pr_positive_probability
    else:
        return "RA-", 1.0 - prob_output.pr_positive_probability


def patient_prob_to_label(patient_output: PatientProbabilisticOutput,
                          threshold: float = 0.50) -> Tuple[str, float]:
    """Convert patient-level output to binary label."""
    if patient_output.pr_positive_probability >= threshold:
        return "RA+", patient_output.pr_positive_probability
    else:
        return "RA-", 1.0 - patient_output.pr_positive_probability
