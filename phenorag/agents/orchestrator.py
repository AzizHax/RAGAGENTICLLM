"""
phenorag/agents/orchestrator.py

LangGraph Multi-Architecture Orchestrator: B1 / B2 / B3 / B4

Each architecture is a LangGraph StateGraph:

  B1 Sequential:
    extract → reason → aggregate

  B2 Hierarchical:
    reason → [should_feedback?] → feedback → reason → critic → aggregate

  B3 Adaptive:
    router → [fast|std|full] → aggregate

  B4 Consensus:
    vote (3 models) → tally → aggregate
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from phenorag.agents.agent1 import Agent1Pipeline
from phenorag.agents.agent2 import StayLevelReasoner
from phenorag.agents.agent3 import Agent3Pipeline, LLMCritic
from phenorag.agents.configs import (
    Agent1Config, Agent2Config, Agent3Config, PipelineConfig,
)
from phenorag.agents.interfaces import StayAssessment
from phenorag.utils.llm_client import LLMClient
from phenorag.utils.prompt_loader import PromptLoader


# ════════════════════════════════════════════════════════════════
# STATE
# ════════════════════════════════════════════════════════════════

class StayState(TypedDict):
    stay_facts: Dict[str, Any]
    raw_stay: Optional[Dict[str, Any]]
    patient_id: str
    stay_id: str
    assessment: Optional[Dict[str, Any]]
    feedback_cycle: int
    confirmed_absent: List[str]
    route: str
    voter_assessments: List[Dict[str, Any]]
    final_assessment: Optional[Dict[str, Any]]
    trace: List[str]


# ════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ════════════════════════════════════════════════════════════════

class Orchestrator:

    def __init__(self, *,
                 llm: LLMClient,
                 prompts: PromptLoader,
                 pipeline_cfg: PipelineConfig,
                 a1_cfg: Agent1Config,
                 a2_cfg: Agent2Config,
                 a3_cfg: Agent3Config,
                 # PATCH: kb_path optionnel (--no-kb)
                 kb_path: Optional[str] = None,
                 # PATCH: nouveaux params de run.py
                 n_patients: Optional[int] = None,
                 checkpoint_interval: float = 5.0,
                 gt_stay_csv: Optional[str] = None):

        self.llm = llm
        self.prompts = prompts
        self.pcfg = pipeline_cfg
        self.a2_cfg = a2_cfg
        self.n_patients = n_patients
        self.checkpoint_interval = checkpoint_interval
        self.gt_stay_csv = gt_stay_csv

        # PATCH: propagation use_inter_stay_rag depuis PipelineConfig → Agent1Config
        if not pipeline_cfg.use_inter_stay_rag:
            a1_cfg.use_inter_stay_rag = False

        # PATCH: propagation gt_patient_agg depuis PipelineConfig → Agent3Config
        if pipeline_cfg.gt_patient_agg != "any_positive":
            a3_cfg.aggregation_strategy = pipeline_cfg.gt_patient_agg

        self.agent1 = Agent1Pipeline(llm=llm, prompts=prompts, cfg=a1_cfg, kb_path=kb_path)
        self.agent2 = StayLevelReasoner(llm=llm, prompts=prompts, cfg=a2_cfg)
        self.agent3 = Agent3Pipeline(llm=llm, prompts=prompts, cfg=a3_cfg)
        self.critic = LLMCritic(llm=llm, prompts=prompts, cfg=a3_cfg) if a3_cfg.use_critic else None

        # PATCH: stocker les noms de modèles B4 séparément pour _node_tally
        self.b4_voter_models: List[str] = []
        self.b4_voters: List[StayLevelReasoner] = []
        if pipeline_cfg.architecture.upper() == "B4":
            models = pipeline_cfg.b4_models or [
                "qwen7b:latest", "mistral_7b:latest", "llama3_1_8b_gguf:latest"]
            self.b4_voter_models = models
            for m in models:
                cfg = Agent2Config(
                    model=m, temperature=a2_cfg.temperature,
                    timeout_s=a2_cfg.timeout_s, acr_threshold=a2_cfg.acr_threshold,
                    rf_uln=a2_cfg.rf_uln, ccp_uln=a2_cfg.ccp_uln,
                    crp_threshold=a2_cfg.crp_threshold, esr_threshold=a2_cfg.esr_threshold,
                    prompt_ra_relatedness=a2_cfg.prompt_ra_relatedness,
                    prompt_acr_scoring=a2_cfg.prompt_acr_scoring)
                self.b4_voters.append(StayLevelReasoner(llm=llm, prompts=prompts, cfg=cfg))
            print(f"  B4 Consensus: {len(models)} voters -> {models}")

        self.graph = self._build_graph()

    # ══════════════════════════════════════════════════════════
    # GRAPH BUILDERS
    # ══════════════════════════════════════════════════════════

    def _build_graph(self):
        arch = self.pcfg.architecture.upper()
        builder = {"B1": self._build_b1, "B2": self._build_b2,
                   "B3": self._build_b3, "B4": self._build_b4}
        if arch not in builder:
            raise ValueError(f"Unknown architecture: {arch}")
        g = builder[arch]()
        return g.compile(checkpointer=MemorySaver())

    def _build_b1(self) -> StateGraph:
        g = StateGraph(StayState)
        g.add_node("reason", self._node_reason_simple)
        g.set_entry_point("reason")
        g.add_edge("reason", END)
        return g

    def _build_b2(self) -> StateGraph:
        g = StateGraph(StayState)
        g.add_node("reason", self._node_reason_feedback)
        g.add_node("feedback", self._node_feedback)
        g.add_node("critic", self._node_critic)
        g.set_entry_point("reason")
        g.add_conditional_edges("reason", self._should_feedback,
                                {"feedback": "feedback", "critic": "critic"})
        g.add_edge("feedback", "reason")
        g.add_edge("critic", END)
        return g

    def _build_b3(self) -> StateGraph:
        g = StateGraph(StayState)
        g.add_node("router", self._node_router)
        g.add_node("fast", self._node_fast)
        g.add_node("std", self._node_std)
        g.add_node("full", self._node_full)
        g.add_node("critic", self._node_critic)
        g.set_entry_point("router")
        g.add_conditional_edges("router", self._route_decision,
                                {"fast": "fast", "std": "std", "full": "full"})
        g.add_edge("fast", END)
        g.add_edge("std", END)
        g.add_conditional_edges("full", self._should_critic,
                                {"critic": "critic", "done": END})
        g.add_edge("critic", END)
        return g

    def _build_b4(self) -> StateGraph:
        g = StateGraph(StayState)
        g.add_node("vote", self._node_vote)
        g.add_node("tally", self._node_tally)
        g.set_entry_point("vote")
        g.add_edge("vote", "tally")
        g.add_edge("tally", END)
        return g

    # ══════════════════════════════════════════════════════════
    # B1 NODES
    # ══════════════════════════════════════════════════════════

    def _node_reason_simple(self, state: StayState) -> dict:
        a = self.agent2.reason_stay(state["stay_facts"], raw_stay=None,
                                     targeted_search_fn=None)
        return {"final_assessment": asdict(a),
                "trace": state.get("trace", []) + [f"[B1] {a.final_stay_label} conf={a.confidence:.2f}"]}

    # ══════════════════════════════════════════════════════════
    # B2 NODES
    # ══════════════════════════════════════════════════════════

    def _node_reason_feedback(self, state: StayState) -> dict:
        cycle = state.get("feedback_cycle", 0)
        a = self.agent2.reason_stay(
            state["stay_facts"],
            raw_stay=state.get("raw_stay") if cycle == 0 else None,
            targeted_search_fn=self.agent1.targeted_search if cycle == 0 else None)
        return {"assessment": asdict(a), "final_assessment": asdict(a),
                "feedback_cycle": cycle + 1,
                "trace": state.get("trace", []) + [
                    f"[B2-C{cycle}] {a.final_stay_label} comp={a.completeness:.2f} miss={a.missing_dimensions}"]}

    def _should_feedback(self, state: StayState) -> str:
        a = state.get("assessment", {})
        cycle = state.get("feedback_cycle", 0)
        missing = a.get("missing_dimensions", [])
        confirmed = state.get("confirmed_absent", [])
        actionable = [d for d in missing if d not in confirmed]
        if cycle <= 2 and actionable and state.get("raw_stay"):
            return "feedback"
        return "critic"

    def _node_feedback(self, state: StayState) -> dict:
        a = state.get("assessment", {})
        missing = a.get("missing_dimensions", [])
        confirmed = list(state.get("confirmed_absent", []))
        actionable = [d for d in missing if d not in confirmed]
        raw = state.get("raw_stay")
        if not actionable or not raw:
            return {"trace": state.get("trace", []) + ["[Feedback] Nothing to search"]}

        result = self.agent1.targeted_search(raw, actionable)
        found = result.get("found_dims", [])
        confirmed.extend(result.get("confirmed_absent", []))

        facts = dict(state["stay_facts"])
        for key in ("disease_mentions", "labs", "drugs"):
            existing = list(facts.get(key, []))
            for item in result.get("new_facts", {}).get(key, []):
                if isinstance(item, dict):
                    existing.append(item)
            facts[key] = existing

        return {"stay_facts": facts, "confirmed_absent": confirmed,
                "trace": state.get("trace", []) + [
                    f"[Feedback] Found={found} Absent={result.get('confirmed_absent', [])}"]}

    def _node_critic(self, state: StayState) -> dict:
        a = state.get("final_assessment") or state.get("assessment", {})
        if not a or not self.critic:
            return {}
        if a.get("internal_label") != "UNCERTAIN":
            return {"trace": state.get("trace", []) + ["[Critic] Not needed"]}

        cr = self.critic.review(
            [a], (a.get("final_stay_label", "RA-"), a.get("confidence", 0.5),
                  f"Score {a.get('acr_eular_final_score', 0)}"))
        if cr:
            rec, cc = cr.get("recommendation", "abstain"), cr.get("confidence", 0.5)
            if rec in ("RA+", "RA-") and cc > a.get("confidence", 0):
                updated = dict(a)
                updated["final_stay_label"] = rec
                updated["confidence"] = cc
                return {"final_assessment": updated,
                        "trace": state.get("trace", []) + [f"[Critic] Override -> {rec} conf={cc:.2f}"]}
        return {"trace": state.get("trace", []) + ["[Critic] No override"]}

    def _should_critic(self, state: StayState) -> str:
        a = state.get("final_assessment") or state.get("assessment", {})
        if a.get("internal_label") == "UNCERTAIN" and self.critic:
            return "critic"
        return "done"

    # ══════════════════════════════════════════════════════════
    # B3 NODES
    # ══════════════════════════════════════════════════════════

    def _node_router(self, state: StayState) -> dict:
        facts = state["stay_facts"]
        raw = state.get("raw_stay")
        n_ent = (len(facts.get("disease_mentions") or [])
                 + len(facts.get("labs") or []) + len(facts.get("drugs") or []))
        n_raw = len(raw.get("records", [])) if raw else 0

        if n_ent <= 1 and n_raw <= self.pcfg.b3_fast_max_records:
            route = "fast"
        else:
            has_sero = any(isinstance(l, dict) and (l.get("test") or "").lower() in ("rf", "anti-ccp")
                           for l in (facts.get("labs") or []))
            has_crp = any(isinstance(l, dict) and "crp" in (l.get("test") or "").lower()
                          for l in (facts.get("labs") or []))
            has_dmard = any(isinstance(d, dict) and (d.get("category") or "") in ("csDMARD", "bDMARD", "tsDMARD")
                            for d in (facts.get("drugs") or []))
            route = "full" if sum([has_sero, has_crp, has_dmard]) == 0 and n_ent > 0 else "std"

        return {"route": route,
                "trace": state.get("trace", []) + [f"[B3-Router] -> {route.upper()}"]}

    def _route_decision(self, state: StayState) -> str:
        return state.get("route", "std")

    def _node_fast(self, state: StayState) -> dict:
        facts = state["stay_facts"]
        dm = facts.get("disease_mentions") or []
        drugs = facts.get("drugs") or []
        has_ra = any(isinstance(d, dict) and "rhumato" in (d.get("entity") or "").lower() for d in dm)
        has_dmard = any(isinstance(d, dict) and (d.get("category") or "") in ("csDMARD", "bDMARD", "tsDMARD") for d in drugs)

        if not has_ra and not has_dmard:
            a = StayAssessment(
                stay_id=state["stay_id"], patient_id=state["patient_id"],
                is_ra_related=False, ra_related_confidence=0.7,
                ra_related_reasoning="B3-FAST: no RA",
                acr_eular_deterministic=None, acr_eular_llm=None,
                acr_eular_final_score=0,
                internal_label="RA-", final_stay_label="RA-", confidence=0.7,
                reasoning_trace=["[B3-FAST] No RA"], extracted_facts=facts,
                class_probabilities={"PR_ABSENT": 0.95, "PR_LATENT": 0.03,
                                     "PR_REMISSION": 0.01, "PR_MODERATE": 0.005, "PR_SEVERE": 0.005},
                predicted_class="PR_ABSENT", pr_positive_probability=0.05)
        else:
            a = self.agent2.reason_stay(facts, raw_stay=None, targeted_search_fn=None)

        return {"final_assessment": asdict(a),
                "trace": state.get("trace", []) + [f"[B3-FAST] {a.final_stay_label}"]}

    def _node_std(self, state: StayState) -> dict:
        a = self.agent2.reason_stay(state["stay_facts"], raw_stay=None, targeted_search_fn=None)
        return {"final_assessment": asdict(a),
                "trace": state.get("trace", []) + [f"[B3-STD] {a.final_stay_label}"]}

    def _node_full(self, state: StayState) -> dict:
        a = self.agent2.reason_stay(
            state["stay_facts"], raw_stay=state.get("raw_stay"),
            targeted_search_fn=self.agent1.targeted_search)
        return {"assessment": asdict(a), "final_assessment": asdict(a),
                "trace": state.get("trace", []) + [f"[B3-FULL] {a.final_stay_label}"]}

    # ══════════════════════════════════════════════════════════
    # B4 NODES
    # ══════════════════════════════════════════════════════════

    def _node_vote(self, state: StayState) -> dict:
        facts = state["stay_facts"]
        results = []
        for voter in self.b4_voters:
            a = voter.reason_stay(facts, raw_stay=None, targeted_search_fn=None)
            results.append(asdict(a))
        return {"voter_assessments": results,
                "trace": state.get("trace", []) + [f"[B4-Vote] {len(results)} voters"]}

    def _node_tally(self, state: StayState) -> dict:
        votes = state.get("voter_assessments", [])
        if not votes:
            return {}

        strategy = self.pcfg.b4_vote_strategy
        # PATCH: utiliser self.b4_voter_models (liste de str) au lieu de v.cfg.model
        model_names = self.b4_voter_models
        labels = [v.get("final_stay_label", "RA-") for v in votes]
        scores = [v.get("acr_eular_final_score", 0) for v in votes]
        n_pos, n_neg = labels.count("RA+"), labels.count("RA-")
        n_v = len(votes)

        if strategy == "unanimous":
            final = "RA+" if n_pos == n_v else "RA-" if n_neg == n_v else ("RA+" if n_pos > n_neg else "RA-")
        elif strategy == "any_positive":
            final = "RA+" if n_pos >= 1 else "RA-"
        else:
            final = "RA+" if n_pos > n_neg else "RA-"

        agreeing = [v for v in votes if v.get("final_stay_label") == final]
        conf = sum(v.get("confidence", 0.5) for v in agreeing) / len(agreeing) if agreeing else 0.5

        merged = dict(votes[0])
        merged["final_stay_label"] = final
        merged["confidence"] = conf
        merged["internal_label"] = f"VOTE:{n_pos}v{n_neg}"
        merged["acr_eular_final_score"] = round(sum(scores) / len(scores))
        if merged.get("acr_eular_deterministic") is None:
            merged["acr_eular_deterministic"] = {}
        merged["acr_eular_deterministic"]["b4_voter_scores"] = scores
        merged["acr_eular_deterministic"]["b4_voter_labels"] = labels
        merged["acr_eular_deterministic"]["b4_voter_models"] = model_names

        trace = [f"[B4-Tally] {n_pos}v{n_neg} -> {final} conf={conf:.2f}"]
        for i, v in enumerate(votes):
            name = model_names[i] if i < len(model_names) else f"v{i}"
            trace.append(f"  {name}: {v.get('final_stay_label')} score={v.get('acr_eular_final_score')}")

        return {"final_assessment": merged,
                "trace": state.get("trace", []) + trace}

    # ══════════════════════════════════════════════════════════
    # MAIN ENTRY
    # ══════════════════════════════════════════════════════════

    def run(self, corpus_path: str, output_dir: str, skip_extraction: bool = False):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        facts_path = out / "facts.jsonl"
        assess_path = out / "assessments.jsonl"
        # PATCH: decisions.json (pas .jsonl) pour correspondre à evaluate() dans run.py
        decisions_path = out / "decisions.json"
        arch = self.pcfg.architecture.upper()

        print(f"\n{'='*60}")
        print(f"LangGraph Orchestrator — {arch}")
        print(f"{'='*60}")

        if skip_extraction and facts_path.exists():
            print(f"[SKIP] Using existing {facts_path}")
        else:
            self.agent1.process_corpus(corpus_path, str(facts_path))

        with open(facts_path, "r", encoding="utf-8") as f:
            all_facts = [json.loads(line) for line in f]
        raw_corpus = json.load(open(corpus_path, "r", encoding="utf-8"))

        # PATCH: support --n-patients pour limiter le corpus
        if self.n_patients is not None:
            raw_corpus = raw_corpus[:self.n_patients]
            patient_ids = {p.get("patient_id") for p in raw_corpus}
            all_facts = [f for f in all_facts if f.get("patient_id") in patient_ids]
            print(f"  [n_patients] Limité à {self.n_patients} patients ({len(all_facts)} séjours)")

        raw_stays = {s["stay_id"]: s for p in raw_corpus for s in p.get("stays", [])}

        print(f"\nStage 2: LangGraph {arch} ({len(all_facts)} stays)")
        print("-" * 60)
        t0 = time.time()
        last_checkpoint = t0

        assessments = []
        for i, facts in enumerate(all_facts, 1):
            sid = facts.get("stay_id", "?")
            pid = facts.get("patient_id", "?")

            state: StayState = {
                "stay_facts": facts, "raw_stay": raw_stays.get(sid),
                "patient_id": pid, "stay_id": sid,
                "assessment": None, "feedback_cycle": 0,
                "confirmed_absent": [], "route": "",
                "voter_assessments": [], "final_assessment": None, "trace": [],
            }

            config = {"configurable": {"thread_id": f"{pid}_{sid}"}}
            result = self.graph.invoke(state, config)

            final = result.get("final_assessment", {})
            if final:
                assessments.append(final)
                print(f"  [{i}/{len(all_facts)}] {pid}/{sid} -> "
                      f"{final.get('final_stay_label', '?')} "
                      f"(conf={final.get('confidence', 0):.2f}, "
                      f"comp={final.get('completeness', 0):.2f}) [{arch}]")

            # PATCH: checkpoint périodique (toutes les N minutes)
            now = time.time()
            if (now - last_checkpoint) >= self.checkpoint_interval * 60:
                ckpt_path = out / "assessments_checkpoint.jsonl"
                with open(ckpt_path, "w", encoding="utf-8") as f:
                    for a in assessments:
                        f.write(json.dumps(a, ensure_ascii=False) + "\n")
                print(f"  [Checkpoint] {len(assessments)} assessments sauvegardés -> {ckpt_path}")
                last_checkpoint = now

        print(f"\n  {len(assessments)} assessments in {time.time()-t0:.1f}s")

        with open(assess_path, "w", encoding="utf-8") as f:
            for a in assessments:
                f.write(json.dumps(a, ensure_ascii=False) + "\n")

        # Stage 3
        print(f"\nStage 3: Patient Aggregation")
        print("-" * 60)
        by_patient = defaultdict(list)
        for a in assessments:
            by_patient[a["patient_id"]].append(a)

        decisions = []
        for pid, pa in by_patient.items():
            dec = self.agent3.process_patient(pa)
            decisions.append(asdict(dec))
            print(f"  {pid} ({dec.n_stays} stays) -> {dec.final_label} "
                  f"(conf={dec.final_confidence:.2f})")

        # PATCH: écriture en JSON (liste) pour compatibilité avec evaluate()
        with open(decisions_path, "w", encoding="utf-8") as f:
            json.dump(decisions, f, indent=2, ensure_ascii=False)

        print(f"\n-> {output_dir}/")
        return decisions

    # ══════════════════════════════════════════════════════════
    # GRAPH VISUALIZATION
    # ══════════════════════════════════════════════════════════

    def draw(self, output_path: str = "graph"):
        """Export graph as Mermaid diagram."""
        try:
            mermaid = self.graph.get_graph().draw_mermaid()
            md_path = f"{output_path}.md"
            with open(md_path, "w") as f:
                f.write(f"```mermaid\n{mermaid}\n```")
            print(f"Graph -> {md_path}")
            print(mermaid)
        except Exception as e:
            print(f"Draw failed: {e}")
