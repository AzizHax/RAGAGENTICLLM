"""
phenorag/agents/agent2.py

Agent 2: Evidence-Aware Stay-Level Reasoning + Feedback Loop

Key architecture:
  Agent2 evaluates → finds missing dims → asks Agent1 to re-search
  → Agent1 returns new facts or confirms absent → Agent2 re-evaluates
  Max 2 feedback cycles.

                    ┌──────────┐
                    │  Agent 2  │
                    │ evaluate  │
                    └─────┬─────┘
                          │ missing dims?
                          ▼
                    ┌──────────┐
                    │  Agent 1  │
                    │ targeted  │
                    │  search   │
                    └─────┬─────┘
                          │ new facts / confirmed absent
                          ▼
                    ┌──────────┐
                    │  Agent 2  │
                    │ re-eval   │
                    └──────────┘
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from rank_bm25 import BM25Okapi

from phenorag.agents.configs import Agent2Config
from phenorag.agents.interfaces import DimensionResult, StayAssessment
from phenorag.agents.probabilistic import EncounterClassifier, prob_to_label
from phenorag.utils.llm_client import LLMClient
from phenorag.utils.prompt_loader import PromptLoader

# Type for the feedback callback
from typing import Callable
TargetedSearchFn = Callable[[Dict, List[str]], Dict[str, Any]]


# ═══════════════════════════════════════════════════════════════
# RAG GUIDELINES
# ═══════════════════════════════════════════════════════════════

class RAGGuidelinesRetriever:

    def __init__(self):
        self.chunks = [
            {"section": "overview", "title": "ACR/EULAR 2010",
             "text": "ACR EULAR 2010 classification criteria RA threshold 6 points 10"},
            {"section": "joint", "title": "Joint Involvement (0-5 pts)",
             "text": "joint involvement 1 large 0 pts 2-10 large 1 pt "
                     "1-3 small 2 pts 4-10 small 3 pts >10 joints 5 pts "
                     "MCP PIP MTP wrist small shoulder elbow hip knee ankle large"},
            {"section": "serology", "title": "Serology RF anti-CCP (0-3 pts)",
             "text": "serology RF anti-CCP ACPA negative 0 pts "
                     "low positive <=3x ULN 2 pts high positive >3x ULN 3 pts"},
            {"section": "acute_phase", "title": "Acute Phase Reactants (0-1 pt)",
             "text": "CRP ESR VHS acute phase both normal 0 at least one elevated 1 pt"},
            {"section": "duration", "title": "Symptom Duration (0-1 pt)",
             "text": "duration symptoms <6 weeks 0 >=6 weeks 1 pt chronicity DMARD"},
            {"section": "treatment", "title": "Treatment Context",
             "text": "DMARD methotrexate MTX biologic anti-TNF rituximab JAK inhibitor"},
        ]
        self.bm25 = BM25Okapi([c["text"].lower().split() for c in self.chunks])

    def retrieve(self, query: str, top_k: int = 4) -> List[Dict]:
        sc = self.bm25.get_scores(query.lower().split())
        top = sorted(range(len(sc)), key=lambda i: -sc[i])[:top_k]
        return [{**self.chunks[i], "score": float(sc[i])} for i in top if sc[i] > 0]

    def build_query(self, facts: Dict) -> str:
        parts = []
        labs = facts.get("labs") or []
        if any(isinstance(l, dict) and "rf" in (l.get("test") or "").lower() for l in labs):
            parts.append("serology RF anti-CCP")
        if any(isinstance(l, dict) and "crp" in (l.get("test") or "").lower() for l in labs):
            parts.append("CRP acute phase")
        if any(isinstance(d, dict) for d in (facts.get("drugs") or [])):
            parts.append("DMARD treatment")
        if any(isinstance(d, dict) for d in (facts.get("disease_mentions") or [])):
            parts.append("rheumatoid arthritis criteria")
        return " ".join(parts) or "ACR EULAR criteria"


# ═══════════════════════════════════════════════════════════════
# STAY-LEVEL REASONER
# ═══════════════════════════════════════════════════════════════

MAX_FEEDBACK_CYCLES = 2


class StayLevelReasoner:

    def __init__(self, *, llm: LLMClient, prompts: PromptLoader, cfg: Agent2Config):
        self.llm = llm
        self.prompts = prompts
        self.cfg = cfg
        self.rag = RAGGuidelinesRetriever()
        self.prob_clf = EncounterClassifier()  # Bayesian probabilistic layer

    # ── Helpers ──────────────────────────────────────────────

    @staticmethod
    def _num(s: Any) -> Optional[float]:
        if s is None: return None
        if isinstance(s, (int, float)): return float(s)
        m = re.search(r"(-?\d+(?:\.\d+)?)", str(s).replace(",", "."))
        return float(m.group(1)) if m else None

    @staticmethod
    def _format_facts(facts: Dict) -> str:
        lines = []
        for dm in (facts.get("disease_mentions") or []):
            if isinstance(dm, dict): lines.append(f"- {dm.get('entity','?')} ({dm.get('status','?')})")
        for lab in (facts.get("labs") or []):
            if isinstance(lab, dict): lines.append(f"- {lab.get('test','?')}: {lab.get('value','')} ({lab.get('polarity','?')})")
        for drug in (facts.get("drugs") or []):
            if isinstance(drug, dict): lines.append(f"- {drug.get('name','?')} ({drug.get('category','?')})")
        return "\n".join(lines) or "No facts"

    def _all_snippets(self, facts: Dict) -> str:
        snips = []
        for key in ("disease_mentions", "labs", "drugs"):
            for ent in (facts.get(key) or []):
                if not isinstance(ent, dict): continue
                ev = ent.get("evidence", {})
                if isinstance(ev, str):
                    # LLM sometimes returns evidence as a plain string
                    snips.append(ev.lower())
                    continue
                if isinstance(ev, list):
                    ev = ev[0] if ev else {}
                if isinstance(ev, dict):
                    s = (ev.get("snippet") or "").lower()
                    if s: snips.append(s)
        return " ".join(snips)

    # ── RA-relatedness ───────────────────────────────────────

    def assess_ra_relatedness(self, facts: Dict) -> Tuple[bool, float, str]:
        ctx = []
        for dm in (facts.get("disease_mentions") or [])[:3]:
            if isinstance(dm, dict): ctx.append(f"  - {dm.get('entity','')} ({dm.get('status','')})")
        for lab in (facts.get("labs") or [])[:5]:
            if isinstance(lab, dict): ctx.append(f"  - {lab.get('test','')}: {lab.get('value','')} ({lab.get('polarity','')})")
        for drug in (facts.get("drugs") or [])[:3]:
            if isinstance(drug, dict): ctx.append(f"  - {drug.get('name','')} ({drug.get('category','')})")

        prompt = self.prompts.render(self.cfg.prompt_ra_relatedness,
                                     stay_id=facts.get("stay_id", "?"),
                                     context_text="\n".join(ctx) or "No entities.")

        parsed, _ = self.llm.generate(prompt=prompt, model=self.cfg.model,
                                       temperature=self.cfg.temperature,
                                       timeout_s=min(self.cfg.timeout_s, 60))
        if parsed and "is_ra_related" in parsed:
            try:
                conf = float(parsed.get("confidence", 0.5))
            except (ValueError, TypeError):
                conf = 0.5
            return bool(parsed["is_ra_related"]), conf, str(parsed.get("reasoning", "LLM"))

        # Heuristic
        has_ra = any(isinstance(d, dict) and "rhumato" in (d.get("entity") or "").lower()
                     for d in (facts.get("disease_mentions") or []))
        has_dmard = any(isinstance(d, dict) and (d.get("category") or "") in ("csDMARD","bDMARD","tsDMARD")
                        for d in (facts.get("drugs") or []))
        has_sero = any(isinstance(l, dict) and (l.get("test") or "").lower() in ("rf","anti-ccp","acpa")
                       for l in (facts.get("labs") or []))
        if has_ra or has_dmard or has_sero:
            return True, 0.7, "Heuristic: RA/DMARD/serology"
        return False, 0.6, "Heuristic: no RA indicators"

    # ══════════════════════════════════════════════════════════
    # DIMENSION EVALUATION (evidence-aware)
    # ══════════════════════════════════════════════════════════

    def _eval_joints(self, facts: Dict) -> DimensionResult:
        snips = self._all_snippets(facts)
        if re.search(r"\bsynovite", snips):
            return DimensionResult(3, "positive", "Synovitis detected")
        if re.search(r"\bmcp\b|\bpip\b|\bpoignet|\bmtp\b", snips):
            return DimensionResult(2, "positive", "Small joints mentioned")
        if re.search(r"\barticul|\bjoint|\bgonfle|\btumef", snips):
            return DimensionResult(1, "positive", "Joint involvement mentioned")
        if re.search(r"pas d.atteinte articulaire|examen articulaire normal", snips):
            return DimensionResult(0, "negative", "Joints explicitly normal")
        return DimensionResult(None, "missing", "No joint evidence")

    def _eval_serology(self, facts: Dict) -> DimensionResult:
        labs = facts.get("labs") or []
        rf_best, ccp_best, rf_pos, ccp_pos, rf_neg, ccp_neg, has = None, None, False, False, False, False, False

        for lab in labs:
            if not isinstance(lab, dict): continue
            test = (lab.get("test") or "").lower()
            pol = (lab.get("polarity") or "").lower()
            num = self._num(lab.get("value"))
            if test in ("rf", "facteur rhumatoide", "rheumatoid factor"):
                has = True
                if pol in ("positive", "positif"): rf_pos = True
                if pol in ("negative", "negatif"): rf_neg = True
                if num is not None: rf_best = max(rf_best or 0, num)
            if "ccp" in test or "acpa" in test:
                has = True
                if pol in ("positive", "positif"): ccp_pos = True
                if pol in ("negative", "negatif"): ccp_neg = True
                if num is not None: ccp_best = max(ccp_best or 0, num)

        if not has:
            return DimensionResult(None, "missing", "No serology data")

        urf, uccp = self.cfg.rf_uln, self.cfg.ccp_uln
        rf_pts = (3 if rf_best and rf_best > 3*urf else 2 if rf_best and rf_best > urf else 2 if rf_pos else 0)
        ccp_pts = (3 if ccp_best and ccp_best > 3*uccp else 2 if ccp_best and ccp_best > uccp else 2 if ccp_pos else 0)
        pts = max(rf_pts, ccp_pts)

        if pts > 0:
            return DimensionResult(pts, "positive", f"RF:{rf_best or 'pos'} CCP:{ccp_best or 'pos'}")
        if rf_neg or ccp_neg:
            return DimensionResult(0, "negative", "Serology negative")
        return DimensionResult(0, "negative", "Serology present but negative")

    def _eval_acute_phase(self, facts: Dict) -> DimensionResult:
        has = False
        for lab in (facts.get("labs") or []):
            if not isinstance(lab, dict): continue
            test = (lab.get("test") or "").lower()
            pol = (lab.get("polarity") or "").lower()
            val = self._num(lab.get("value"))
            if "crp" in test or "esr" in test or "vhs" in test:
                has = True
                if pol in ("positive", "positif"):
                    return DimensionResult(1, "positive", f"{test.upper()} elevated")
                if "crp" in test and val is not None and val >= self.cfg.crp_threshold:
                    return DimensionResult(1, "positive", f"CRP {val:.1f} >= {self.cfg.crp_threshold}")
                if ("esr" in test or "vhs" in test) and val is not None and val >= self.cfg.esr_threshold:
                    return DimensionResult(1, "positive", f"ESR {val:.0f} >= {self.cfg.esr_threshold}")
        if has:
            return DimensionResult(0, "negative", "Inflammatory markers normal")
        return DimensionResult(None, "missing", "No CRP/ESR data")

    def _eval_duration(self, facts: Dict) -> DimensionResult:
        snips = self._all_snippets(facts)
        if re.search(r"depuis\s+\d+\s*(semaines|mois|ans)|>\s*6\s*semaines|chronique|suivi\s+depuis", snips):
            return DimensionResult(1, "positive", "Duration >= 6 weeks")
        if any(isinstance(d, dict) and (d.get("category") or "") in ("csDMARD","bDMARD","tsDMARD")
               for d in (facts.get("drugs") or [])):
            return DimensionResult(1, "positive", "DMARD implies >= 6 weeks")
        if re.search(r"debut\s+r[eé]cent|<\s*6\s*semaines|depuis\s+\d\s*jours", snips):
            return DimensionResult(0, "negative", "Recent onset < 6 weeks")
        return DimensionResult(None, "missing", "Duration unknown")

    # ══════════════════════════════════════════════════════════
    # STRONG SIGNALS
    # ══════════════════════════════════════════════════════════

    def _strong_ra(self, facts: Dict) -> List[str]:
        """
        Strong RA signals. STRICT: 'mentioned' alone is NOT strong.
        Only confirmed/suspected PR, or objective evidence (DMARD, sero+).
        """
        signals = []
        for d in (facts.get("disease_mentions") or []):
            if not isinstance(d, dict): continue
            entity = (d.get("entity") or "").lower()
            status = (d.get("status") or "").lower()
            # Only confirmed/suspected count — 'mentioned' is too weak
            if "rhumato" in entity and status in ("confirmed", "suspected"):
                signals.append(f"PR {status}")
        for d in (facts.get("drugs") or []):
            if isinstance(d, dict) and (d.get("category") or "") in ("csDMARD", "bDMARD", "tsDMARD"):
                signals.append(f"DMARD: {d.get('name', '?')}")
        for l in (facts.get("labs") or []):
            if not isinstance(l, dict): continue
            pol = (l.get("polarity") or "").lower()
            if pol in ("positive", "positif"):
                t = (l.get("test") or "").lower()
                if "rf" in t: signals.append("RF+")
                if "ccp" in t or "acpa" in t: signals.append("anti-CCP+")
        return signals

    def _strong_neg(self, facts: Dict) -> List[str]:
        """
        Strong negative signals. Includes:
        - Explicit negative labs (RF-, CCP-, CRP normal)
        - PR negated
        - ABSENCE of any RA indicator (no PR mention, no DMARD, no sero)
        """
        signals = []
        labs = facts.get("labs") or []
        drugs = facts.get("drugs") or []
        dm = facts.get("disease_mentions") or []

        # Explicit negative serology
        if any(isinstance(l, dict) and "rf" in (l.get("test") or "").lower()
               and (l.get("polarity") or "").lower() in ("negative", "negatif") for l in labs):
            signals.append("RF-")
        if any(isinstance(l, dict) and ("ccp" in (l.get("test") or "").lower() or "acpa" in (l.get("test") or "").lower())
               and (l.get("polarity") or "").lower() in ("negative", "negatif") for l in labs):
            signals.append("anti-CCP-")

        # Normal inflammation
        if any(isinstance(l, dict) and "crp" in (l.get("test") or "").lower()
               and (l.get("polarity") or "").lower() in ("negative", "negatif") for l in labs):
            signals.append("CRP normal")

        # PR explicitly negated
        for d in dm:
            if isinstance(d, dict) and (d.get("status") or "").lower() == "negated":
                signals.append("PR negated")

        # ABSENCE signals: no RA-related content at all
        has_any_ra_mention = any(
            isinstance(d, dict) and "rhumato" in (d.get("entity") or "").lower() for d in dm)
        has_any_dmard = any(
            isinstance(d, dict) and (d.get("category") or "") in ("csDMARD", "bDMARD", "tsDMARD") for d in drugs)
        has_any_sero = any(
            isinstance(l, dict) and (l.get("test") or "").lower() in ("rf", "anti-ccp", "acpa") for l in labs)

        if not has_any_ra_mention and not has_any_dmard and not has_any_sero:
            signals.append("No RA indicators at all")

        return signals

    # ══════════════════════════════════════════════════════════
    # DECISION
    # ══════════════════════════════════════════════════════════

    def _decide(self, score: int, completeness: float, missing: List[str],
                strong_ra: List[str], strong_neg: List[str],
                confirmed_absent: List[str], trace: List[str]) -> Tuple[str, str, float]:
        """
        Decision logic. Order matters:
        1. Score >= threshold → RA+
        2. Strong negatives (explicit or absence) → RA-
        3. Score near threshold + strong RA → rescue RA+
        4. Low completeness → UNCERTAIN
        5. Default → RA-
        """
        thr = self.cfg.acr_threshold
        truly_missing = [d for d in missing if d not in confirmed_absent]

        # ── Case 1: ACR score meets threshold → RA+ ──
        if score >= thr:
            trace.append(f"[Decision] Score {score} >= {thr} -> RA+")
            return "RA+", "RA+", 0.90

        # ── Case 2: Strong negatives → RA- (CHECK THIS EARLY) ──
        # If we have explicit negative evidence OR no RA indicators at all
        if strong_neg and not strong_ra:
            trace.append(f"[Decision] Score {score}, strong neg {strong_neg}, no strong RA -> RA-")
            return "RA-", "RA-", 0.85

        # If negatives outnumber positives
        if len(strong_neg) > len(strong_ra):
            trace.append(f"[Decision] Score {score}, negatives ({len(strong_neg)}) > positives ({len(strong_ra)}) -> RA-")
            return "RA-", "RA-", 0.80

        # ── Case 3: Score close to threshold + strong objective RA signals ──
        # Only rescue if score is borderline (>= threshold - 2) AND strong signals
        if strong_ra and score >= thr - 2:
            # Require at least 2 OBJECTIVE signals (DMARD, RF+, CCP+)
            # PR mention alone is not enough
            objective_ra = [s for s in strong_ra if not s.startswith("PR")]
            if len(objective_ra) >= 2 and truly_missing:
                trace.append(f"[Decision] Score {score} (borderline), "
                             f"objective RA {objective_ra}, missing {truly_missing} -> UNCERTAIN->RA+")
                return "UNCERTAIN", "RA+", 0.60
            if len(objective_ra) >= 2 and not truly_missing:
                trace.append(f"[Decision] Score {score} (borderline), "
                             f"objective RA {objective_ra}, all checked -> rescue RA+")
                return "RA+", "RA+", 0.65

        # ── Case 4: Low completeness, mixed signals ──
        if completeness < 0.5 and strong_ra and not strong_neg:
            trace.append(f"[Decision] Score {score}, low completeness {completeness:.2f}, "
                         f"some RA signals -> UNCERTAIN")
            return "UNCERTAIN", "RA+", 0.50

        # ── Case 5: Low completeness, no signals either way ──
        if completeness < 0.5 and not strong_ra and not strong_neg:
            trace.append(f"[Decision] Score {score}, low completeness {completeness:.2f}, "
                         f"no signals -> UNCERTAIN->RA-")
            return "UNCERTAIN", "RA-", 0.45

        # ── Case 6: Default — score below threshold ──
        trace.append(f"[Decision] Score {score} < {thr}, completeness {completeness:.2f} -> RA-")
        return "RA-", "RA-", 0.75

    # ══════════════════════════════════════════════════════════
    # LLM SCORING
    # ══════════════════════════════════════════════════════════

    def _llm_scoring(self, facts: Dict, det: Dict) -> Optional[Dict]:
        query = self.rag.build_query(facts)
        chunks = self.rag.retrieve(query, 4)
        guidelines = "\n\n".join(f"## {c['title']}\n{c['text']}" for c in chunks)
        bd = det["breakdown"]
        prompt = self.prompts.render(
            self.cfg.prompt_acr_scoring,
            guidelines_text=guidelines, facts_text=self._format_facts(facts),
            joint_pts=det["joint_involvement"], joint_detail=bd["joint"],
            sero_pts=det["serology"], sero_detail=bd["serology"],
            acute_pts=det["acute_phase"], acute_detail=bd["acute_phase"],
            dur_pts=det["duration"], dur_detail=bd["duration"],
            det_total=det["total"])
        parsed, _ = self.llm.generate(prompt=prompt, model=self.cfg.model,
                                       temperature=self.cfg.temperature,
                                       timeout_s=self.cfg.timeout_s)
        return parsed if parsed and "total" in parsed else None

    # ══════════════════════════════════════════════════════════
    # CORE: evaluate dimensions from facts
    # ══════════════════════════════════════════════════════════

    def _evaluate_dimensions(self, facts: Dict) -> Dict[str, DimensionResult]:
        return {
            "joint": self._eval_joints(facts),
            "serology": self._eval_serology(facts),
            "acute_phase": self._eval_acute_phase(facts),
            "duration": self._eval_duration(facts),
        }

    @staticmethod
    def _merge_facts(base: Dict, new_facts: Dict) -> Dict:
        """Merge new facts into existing facts (no duplicates by test name)."""
        merged = {k: list(base.get(k, [])) for k in ("disease_mentions", "labs", "drugs")}
        existing_tests = {(l.get("test") or "").lower() for l in merged["labs"] if isinstance(l, dict)}
        for lab in new_facts.get("labs", []):
            if isinstance(lab, dict) and (lab.get("test") or "").lower() not in existing_tests:
                merged["labs"].append(lab)
                existing_tests.add((lab.get("test") or "").lower())
        for key in ("disease_mentions", "drugs"):
            merged[key].extend(new_facts.get(key, []))
        # Preserve non-fact fields
        for k, v in base.items():
            if k not in merged:
                merged[k] = v
        return merged

    # ══════════════════════════════════════════════════════════
    # MAIN ENTRY: reason_stay with feedback loop
    # ══════════════════════════════════════════════════════════

    def reason_stay(self, stay_facts: Dict,
                    raw_stay: Optional[Dict] = None,
                    targeted_search_fn: Optional[TargetedSearchFn] = None) -> StayAssessment:
        """
        Args:
            stay_facts: extracted facts from Agent 1
            raw_stay: original stay dict (ALL records) for feedback loop
            targeted_search_fn: Agent1.targeted_search bound to this stay
        """
        sid = stay_facts.get("stay_id", "?")
        pid = stay_facts.get("patient_id", "?")
        trace: List[str] = []

        # 1) RA-relatedness
        is_ra, ra_conf, ra_reason = self.assess_ra_relatedness(stay_facts)
        trace.append(f"[RA-related] {ra_reason} (conf: {ra_conf:.2f})")

        if not is_ra:
            # Still compute probabilistic output: strong PR_ABSENT
            absent_probs = {"PR_ABSENT": 0.95, "PR_LATENT": 0.03,
                            "PR_REMISSION": 0.01, "PR_MODERATE": 0.005, "PR_SEVERE": 0.005}
            return StayAssessment(
                stay_id=sid, patient_id=pid,
                is_ra_related=False, ra_related_confidence=ra_conf,
                ra_related_reasoning=ra_reason,
                acr_eular_deterministic=None, acr_eular_llm=None,
                acr_eular_final_score=0,
                internal_label="RA-", final_stay_label="RA-",
                confidence=0.95, reasoning_trace=trace,
                extracted_facts=stay_facts,
                class_probabilities=absent_probs,
                predicted_class="PR_ABSENT",
                pr_positive_probability=0.05)

        # 2) Evaluate + feedback loop
        current_facts = stay_facts
        confirmed_absent: List[str] = []

        for cycle in range(MAX_FEEDBACK_CYCLES + 1):  # cycle 0 = initial, 1-2 = feedback
            dims = self._evaluate_dimensions(current_facts)
            missing = [name for name, d in dims.items() if d.status == "missing"]
            # Remove already confirmed absent
            actionable_missing = [d for d in missing if d not in confirmed_absent]

            for name, d in dims.items():
                icon = {"positive": "+", "negative": "-", "missing": "?"}[d.status]
                pts = str(d.points) if d.points is not None else "?"
                trace.append(f"  [{name}] {icon} pts={pts} | {d.reason}")

            completeness = (4 - len(missing)) / 4
            trace.append(f"[Cycle {cycle}] Completeness={completeness:.2f}, "
                         f"missing={missing}, confirmed_absent={confirmed_absent}")

            # If no actionable missing or no search function -> stop
            if not actionable_missing or not targeted_search_fn or not raw_stay:
                trace.append(f"[Feedback] No more search needed (cycle {cycle})")
                break

            if cycle >= MAX_FEEDBACK_CYCLES:
                trace.append(f"[Feedback] Max cycles reached ({MAX_FEEDBACK_CYCLES})")
                break

            # FEEDBACK: ask Agent 1 to re-search
            trace.append(f"[Feedback] Requesting Agent1 search for: {actionable_missing}")
            search_result = targeted_search_fn(raw_stay, actionable_missing)

            found = search_result.get("found_dims", [])
            newly_absent = search_result.get("confirmed_absent", [])
            new_facts = search_result.get("new_facts", {})

            confirmed_absent.extend(newly_absent)
            trace.append(f"[Feedback] Found: {found}, Confirmed absent: {newly_absent}, "
                         f"Method: {search_result.get('search_method', '?')}")

            if not found:
                trace.append("[Feedback] Nothing new found, stopping")
                break

            # Merge new facts and re-evaluate
            current_facts = self._merge_facts(current_facts, new_facts)
            trace.append(f"[Feedback] Facts enriched, re-evaluating...")

        # 3) Final scoring
        dims = self._evaluate_dimensions(current_facts)
        missing = [n for n, d in dims.items() if d.status == "missing"]
        completeness = (4 - len(missing)) / 4
        known_score = sum(d.points for d in dims.values() if d.points is not None)

        det = {
            "joint_involvement": dims["joint"].points or 0,
            "serology": dims["serology"].points or 0,
            "acute_phase": dims["acute_phase"].points or 0,
            "duration": dims["duration"].points or 0,
            "total": known_score,
            "threshold_met": known_score >= self.cfg.acr_threshold,
            "breakdown": {k: d.reason for k, d in dims.items()},
            "dimension_status": {k: d.status for k, d in dims.items()},
        }
        trace.append(f"[Score] Known={known_score}/10, completeness={completeness:.2f}")

        # 4) LLM scoring
        llm_result = self._llm_scoring(current_facts, det)
        final_score = llm_result.get("total", known_score) if llm_result else known_score
        if llm_result:
            trace.append(f"[LLM] {final_score}/10")

        # 5) Strong signals
        strong_ra = self._strong_ra(current_facts)
        strong_neg = self._strong_neg(current_facts)
        if strong_ra: trace.append(f"[Strong RA] {strong_ra}")
        if strong_neg: trace.append(f"[Strong Neg] {strong_neg}")

        # 6) Decision (rule-based, kept as baseline)
        internal, final, conf = self._decide(
            final_score, completeness, missing, strong_ra, strong_neg,
            confirmed_absent, trace)

        # 7) PROBABILISTIC LAYER — Bayesian classification
        assessment_dict = {
            "acr_eular_deterministic": det,
            "completeness": completeness,
            "extracted_facts": current_facts,
        }
        prob_features = self.prob_clf.extract_features(assessment_dict)
        prob_output = self.prob_clf.classify(prob_features)

        prob_label, prob_conf = prob_to_label(prob_output, threshold=0.50)

        trace.append(f"[Probabilistic] P(PR+)={prob_output.pr_positive_probability:.3f}, "
                     f"class={prob_output.predicted_class} "
                     f"({', '.join(f'{k}:{v:.2f}' for k, v in prob_output.class_probabilities.items())})")

        # ── Reconciliation rule-based vs probabilistic ──
        # The rule-based label is the baseline (clinical guarantee via ACR/EULAR).
        # But the probabilistic model can CORRECT specific cases:
        #
        # Case: Rule says RA- but Bayes says PR_REMISSION with high P(PR+)
        #   → The patient HAS RA, they're just in remission (low ACR score)
        #   → Correct to RA+
        #
        # Case: Rule says RA+ but Bayes says PR_ABSENT with high P(absent)
        #   → Unlikely, but possible if rule-based rescue was too aggressive
        #   → Keep RA+ (conservative: don't miss a true positive)

        reconciled_label = final
        reconciled_conf = prob_conf

        if final == "RA-" and prob_output.pr_positive_probability >= 0.80:
            # Bayes strongly disagrees — likely PR in remission
            pred_class = prob_output.predicted_class
            if pred_class in ("PR_REMISSION", "PR_MODERATE", "PR_SEVERE"):
                reconciled_label = "RA+"
                reconciled_conf = prob_output.pr_positive_probability
                trace.append(f"[Reconciliation] Rule=RA- but Bayes={pred_class} "
                             f"P(PR+)={prob_output.pr_positive_probability:.3f} -> override to RA+")

        trace.append(f"[Final] Rule={final}, Bayes={prob_label}, "
                     f"Reconciled={reconciled_label}(conf={reconciled_conf:.2f})")

        return StayAssessment(
            stay_id=sid, patient_id=pid,
            is_ra_related=True, ra_related_confidence=ra_conf,
            ra_related_reasoning=ra_reason,
            acr_eular_deterministic=det, acr_eular_llm=llm_result,
            acr_eular_final_score=final_score,
            completeness=completeness, missing_dimensions=missing,
            strong_ra_signals=strong_ra, strong_negative_signals=strong_neg,
            internal_label=internal, final_stay_label=reconciled_label,
            confidence=reconciled_conf, reasoning_trace=trace,
            extracted_facts=current_facts,
            class_probabilities=prob_output.class_probabilities,
            predicted_class=prob_output.predicted_class,
            pr_positive_probability=prob_output.pr_positive_probability)
