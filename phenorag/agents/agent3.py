"""
phenorag/agents/agent3.py

Agent 3: Patient-Level Aggregation + LLM Critic + Guardrails

Refactored:
  - No globals
  - Uses LLMClient + PromptLoader + Agent3Config
  - Output via interfaces.PatientDecision
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

from phenorag.agents.configs import Agent3Config
from phenorag.agents.interfaces import PatientDecision
from phenorag.agents.probabilistic import (
    EncounterClassifier, PatientAggregator, ProbabilisticOutput,
    patient_prob_to_label, CLASSES,
)
from phenorag.utils.llm_client import LLMClient
from phenorag.utils.prompt_loader import PromptLoader


# ═══════════════════════════════════════════════════════════════
# GUARDRAILS
# ═══════════════════════════════════════════════════════════════

class Guardrails:

    @staticmethod
    def check_temporal_contradictions(assessments: List[Dict]) -> List[str]:
        labs_by_test: Dict[str, List[str]] = defaultdict(list)
        for a in assessments:
            for lab in (a.get("extracted_facts", {}).get("labs") or []):
                if not isinstance(lab, dict):
                    continue
                test = (lab.get("test") or "").lower().strip()
                pol = (lab.get("polarity") or "").lower().strip()
                if test and pol:
                    labs_by_test[test].append(pol)
        warnings = []
        for test, pols in labs_by_test.items():
            if "positive" in pols and "negative" in pols:
                warnings.append(f"Temporal contradiction: {test} both + and - across stays")
        return warnings

    @staticmethod
    def check_evidence_quality(assessments: List[Dict]) -> List[str]:
        n = len(assessments)
        with_ev = sum(
            1 for a in assessments
            if any(a.get("extracted_facts", {}).get(k) for k in ("disease_mentions", "labs", "drugs"))
        )
        if with_ev == 0:
            return ["No evidence in ANY stay"]
        if with_ev < n * 0.5:
            return [f"Evidence sparse: {with_ev}/{n} stays"]
        return []

    @classmethod
    def validate(cls, assessments: List[Dict]) -> Tuple[bool, List[str]]:
        w = cls.check_temporal_contradictions(assessments)
        w += cls.check_evidence_quality(assessments)
        return len(w) == 0, w


# ═══════════════════════════════════════════════════════════════
# AGGREGATION
# ═══════════════════════════════════════════════════════════════

class Aggregation:

    @staticmethod
    def any_positive(assessments: List[Dict]) -> Tuple[str, float, str]:
        n = len(assessments)
        pos = [a for a in assessments if a["final_stay_label"] == "RA+"]
        if pos:
            mx = max(a["confidence"] for a in pos)
            return "RA+", mx, f"ANY_POSITIVE: {len(pos)}/{n} RA+"
        avg = sum(a["confidence"] for a in assessments) / n
        return "RA-", avg, f"ANY_POSITIVE: 0/{n} RA+"

    @staticmethod
    def majority(assessments: List[Dict]) -> Tuple[str, float, str]:
        n = len(assessments)
        pos = sum(1 for a in assessments if a["final_stay_label"] == "RA+")
        if pos > n / 2:
            return "RA+", 0.85, f"MAJORITY: {pos}/{n} RA+"
        return "RA-", 0.75, f"MAJORITY: {pos}/{n} RA+"

    @staticmethod
    def confirmed(assessments: List[Dict]) -> Tuple[str, float, str]:
        high = [a for a in assessments if a.get("acr_eular_final_score", 0) >= 8]
        if high:
            return "RA+", 0.95, f"CONFIRMED: {len(high)} stays ACR>=8"
        pos = sum(1 for a in assessments if a["final_stay_label"] == "RA+")
        if pos:
            return "RA+", 0.75, f"CONFIRMED: {pos} RA+ but no high-conf"
        return "RA-", 0.85, "CONFIRMED: no RA+"

    @classmethod
    def run(cls, assessments: List[Dict], strategy: str) -> Tuple[str, float, str]:
        fn = {"any_positive": cls.any_positive,
              "majority": cls.majority,
              "confirmed": cls.confirmed}
        if strategy not in fn:
            raise ValueError(f"Unknown strategy: {strategy}")
        return fn[strategy](assessments)


# ═══════════════════════════════════════════════════════════════
# LLM CRITIC
# ═══════════════════════════════════════════════════════════════

class LLMCritic:

    def __init__(self, *, llm: LLMClient, prompts: PromptLoader, cfg: Agent3Config):
        self.llm = llm
        self.prompts = prompts
        self.cfg = cfg

    def review(self, assessments: List[Dict],
               agg_result: Tuple[str, float, str]) -> Optional[Dict]:
        label, conf, reason = agg_result
        summary = self._build_summary(assessments)

        prompt = self.prompts.render(
            self.cfg.prompt_critic,
            patient_summary=summary,
            agg_label=label,
            agg_conf=f"{conf:.2f}",
            agg_reason=reason,
        )

        parsed, _ = self.llm.generate(
            prompt=prompt, model=self.cfg.model,
            temperature=self.cfg.temperature,
            timeout_s=self.cfg.timeout_s,
        )
        if parsed and "recommendation" in parsed and "confidence" in parsed:
            return parsed
        return None

    @staticmethod
    def _build_summary(assessments: List[Dict]) -> str:
        n = len(assessments)
        n_pos = sum(1 for a in assessments if a["final_stay_label"] == "RA+")
        n_rel = sum(1 for a in assessments if a.get("is_ra_related", False))

        lines = [f"Total stays: {n}", f"RA-related: {n_rel}",
                 f"RA+: {n_pos}", f"RA-: {n - n_pos}", "", "STAYS:"]
        for i, a in enumerate(assessments, 1):
            lines.append(
                f"  {i}. {a.get('stay_id','?')}: {a['final_stay_label']} "
                f"(ACR: {a.get('acr_eular_final_score','?')}/10, "
                f"conf: {a.get('confidence',0):.2f})")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# PIPELINE
# ═══════════════════════════════════════════════════════════════

class Agent3Pipeline:

    def __init__(self, *, llm: LLMClient, prompts: PromptLoader, cfg: Agent3Config):
        self.cfg = cfg
        self.critic = LLMCritic(llm=llm, prompts=prompts, cfg=cfg) if cfg.use_critic else None
        self.patient_agg = PatientAggregator(temporal_decay=0.9)

    def process_patient(self, assessments: List[Dict]) -> PatientDecision:
        pid = assessments[0]["patient_id"]
        n = len(assessments)
        trace: List[str] = []

        labels = [a["final_stay_label"] for a in assessments]
        confs = [a["confidence"] for a in assessments]
        n_rel = sum(1 for a in assessments if a.get("is_ra_related", False))
        n_pos = labels.count("RA+")

        trace.append(f"[Stays] {n}: {n_pos} RA+, {n - n_pos} RA-")

        # 1) Rule-based aggregation
        agg_label, agg_conf, agg_reason = Aggregation.run(
            assessments, self.cfg.aggregation_strategy)
        trace.append(f"[Agg] {agg_reason}")

        # 2) Guardrails
        grd_ok, grd_warn = Guardrails.validate(assessments)
        if not grd_ok:
            for w in grd_warn:
                trace.append(f"[Guardrail] {w}")

        # 3) Critic
        final_label, final_conf, source = agg_label, agg_conf, "aggregation"
        critic_used = False
        critic_result = None
        override = False

        if self.critic:
            needs = (not self.cfg.critic_only_on_uncertain
                     or agg_conf < self.cfg.critic_confidence_threshold
                     or not grd_ok)
            if needs:
                critic_used = True
                critic_result = self.critic.review(assessments, (agg_label, agg_conf, agg_reason))
                if critic_result:
                    rec = critic_result.get("recommendation", "abstain")
                    c_conf = critic_result.get("confidence", 0.5)
                    trace.append(f"[Critic] {rec} (conf: {c_conf:.2f})")
                    if (self.cfg.allow_override and rec != agg_label
                            and rec != "abstain" and c_conf >= agg_conf):
                        final_label, final_conf, source = rec, c_conf, "llm_override"
                        override = True
                        trace.append(f"[Override] {agg_label} -> {rec}")

        # 4) BAYESIAN PATIENT AGGREGATION
        stay_prob_outputs = []
        for a in assessments:
            cp = a.get("class_probabilities", {})
            if cp:
                stay_prob_outputs.append(ProbabilisticOutput(
                    class_probabilities=cp,
                    predicted_class=a.get("predicted_class", ""),
                    pr_positive_probability=a.get("pr_positive_probability", 0.0),
                    confidence=a.get("confidence", 0.5),
                    acr_score=a.get("acr_eular_final_score", 0),
                    completeness=a.get("completeness", 0.0),
                ))

        patient_probs = {}
        patient_class = ""
        patient_pr_prob = 0.0
        patient_trend = "stable"

        if stay_prob_outputs:
            patient_output = self.patient_agg.aggregate(stay_prob_outputs)
            patient_probs = patient_output.class_probabilities
            patient_class = patient_output.predicted_class
            patient_pr_prob = patient_output.pr_positive_probability
            patient_trend = patient_output.temporal_trend

            # Override confidence with Bayesian probability
            bayes_label, bayes_conf = patient_prob_to_label(patient_output, threshold=0.50)

            trace.append(f"[Bayesian] P(PR+)={patient_pr_prob:.3f}, "
                         f"class={patient_class}, trend={patient_trend}")
            trace.append(f"[Bayesian] {', '.join(f'{k}:{v:.2f}' for k, v in patient_probs.items())}")

            # Bayesian confidence overrides rule-based confidence
            final_conf = bayes_conf

        trace.append(f"[Final] {final_label} (conf: {final_conf:.2f}, src: {source})")

        return PatientDecision(
            patient_id=pid, n_stays=n,
            stay_labels=labels, stay_confidences=confs,
            n_ra_related_stays=n_rel,
            n_ra_positive_stays=n_pos,
            n_ra_negative_stays=n - n_pos,
            aggregation_strategy=self.cfg.aggregation_strategy,
            aggregation_label=agg_label,
            aggregation_confidence=agg_conf,
            aggregation_reasoning=agg_reason,
            final_label=final_label,
            final_confidence=final_conf,
            decision_source=source,
            guardrails_passed=grd_ok,
            guardrails_warnings=grd_warn,
            reasoning_trace=trace,
            stay_details=[
                {"stay_id": a["stay_id"], "label": a["final_stay_label"],
                 "confidence": a["confidence"],
                 "acr_score": a.get("acr_eular_final_score"),
                 "is_ra_related": a.get("is_ra_related", False),
                 "class_probabilities": a.get("class_probabilities", {}),
                 "predicted_class": a.get("predicted_class", ""),
                 "pr_positive_probability": a.get("pr_positive_probability", 0.0)}
                for a in assessments
            ],
            llm_critic_used=critic_used,
            llm_critic_recommendation=critic_result.get("recommendation") if critic_result else None,
            llm_critic_confidence=critic_result.get("confidence") if critic_result else None,
            llm_critic_reasoning=critic_result.get("reasoning") if critic_result else None,
            llm_override_applied=override,
            patient_class_probabilities=patient_probs,
            patient_predicted_class=patient_class,
            patient_pr_positive_probability=patient_pr_prob,
            patient_temporal_trend=patient_trend,
        )
