"""
phenorag/agents/agent1.py

Agent 1: IOA Extraction with Inter-Stay RAG + Targeted Search

Includes:
  - Standard extraction pipeline (process_stay, process_corpus)
  - targeted_search(): called by Agent 2 feedback loop to re-search
    ALL records for specific missing dimensions
"""
from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rank_bm25 import BM25Okapi

from phenorag.agents.configs import Agent1Config
from phenorag.agents.interfaces import StayFacts
from phenorag.utils.llm_client import LLMClient
from phenorag.utils.prompt_loader import PromptLoader


# ════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════

def _timing() -> Dict[str, float]:
    return dict(prefilter_ms=0, inter_stay_rag_ms=0, llm_ms=0, total_ms=0)

def _rag_meta() -> Dict[str, Any]:
    return dict(rag_score_max=0.0, rag_score_sum_top3=0.0,
                kb_slots_used=[], retrieved_previous_stays=[],
                temporal_context_used=False)


# ════════════════════════════════════════════════════════════════
# CORPUS LOADER
# ════════════════════════════════════════════════════════════════

class IOACorpusLoader:

    @staticmethod
    def load(path: str) -> List[Dict]:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        # Try JSON array first, fallback to JSONL
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return [json.loads(line) for line in content.splitlines() if line.strip()]

    @staticmethod
    def build_patient_index(corpus: List[Dict]) -> Dict[str, List[Dict]]:
        return {
            p["patient_id"]: sorted(p.get("stays", []), key=lambda s: s.get("date", ""))
            for p in corpus
        }


# ════════════════════════════════════════════════════════════════
# KB MANAGER (BM25 built once)
# ════════════════════════════════════════════════════════════════

class KBManager:

    def __init__(self, kb_path: str):
        kb = json.loads(Path(kb_path).read_text("utf-8"))
        self.slots = {s["slot"]: s for s in kb.get("slots", []) if s.get("slot")}
        self._bm25, self._names = self._index()

    def _index(self) -> Tuple[Optional[BM25Okapi], List[str]]:
        corpus, names = [], []
        for name, d in self.slots.items():
            parts = [d.get("description", "")]
            for v in d.get("patterns", {}).values():
                if isinstance(v, list): parts.extend(v)
                elif isinstance(v, dict):
                    for sub in v.values():
                        if isinstance(sub, list): parts.extend(sub)
            corpus.append(" ".join(str(x).lower() for x in parts))
            names.append(name)
        if not corpus: return None, []
        return BM25Okapi([c.split() for c in corpus]), names

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        if not self._bm25: return []
        sc = self._bm25.get_scores(query.lower().split())
        top = sorted(range(len(sc)), key=lambda i: -sc[i])[:top_k]
        return [self.slots[self._names[i]] for i in top if sc[i] > 0]

    @staticmethod
    def slot_text(slot: Dict) -> str:
        pats = slot.get("patterns", {})
        pos = pats.get("positive", pats.get("anchor", []))
        lines = [f"**{slot.get('slot','?')}**: {slot.get('description','')}"]
        if pos: lines.append(f"  Look for: {', '.join(pos[:5])}")
        return "\n".join(lines)


# ════════════════════════════════════════════════════════════════
# PREFILTER
# ════════════════════════════════════════════════════════════════

_ANCHOR_RX = re.compile(
    r"|".join([
        r"\bpolyarthrite\s+rhumato[ïi]de\b", r"\bPR\b", r"\bRA\b",
        r"\bRF\b", r"\bfacteur\s+rhumato[ïi]de\b", r"\banti[- ]?CCP\b",
        r"\bdouleur\b", r"\braideur\b", r"\barticulation\b", r"\bsynovite\b",
        r"\bCRP\b", r"\bVS\b",
        r"\bm[eé]thotrexate\b", r"\bMTX\b", r"\badalimumab\b",
    ]), re.IGNORECASE)


def prefilter(records: List[Dict]) -> List[Dict]:
    return [r for r in records
            if _ANCHOR_RX.search(f"{r.get('LIBELLE','')} {r.get('REPONSE','')}")]


# ════════════════════════════════════════════════════════════════
# INTER-STAY RAG
# ════════════════════════════════════════════════════════════════

class InterStayRAG:

    def __init__(self, top_k: int = 2):
        self.top_k = top_k

    def retrieve(self, cur: Dict, all_stays: List[Dict],
                 focus: List[str]) -> Tuple[List[Dict], List[float]]:
        cd, cv = cur.get("date", ""), cur.get("visit_number", 999)
        prev = [s for s in all_stays
                if s.get("date", "") < cd or s.get("visit_number", 0) < cv]
        if not prev: return [], []
        corpus = [" ".join(f"{r.get('LIBELLE','')} {r.get('REPONSE','')}"
                           for r in s.get("records", [])) for s in prev]
        bm25 = BM25Okapi([c.lower().split() for c in corpus])
        sc = bm25.get_scores(" ".join(focus).lower().split())
        top = sorted(range(len(sc)), key=lambda i: -sc[i])[:self.top_k]
        return [prev[i] for i in top], [float(sc[i]) for i in top]

    @staticmethod
    def compact_context(retrieved: List[Dict], max_lines: int = 3) -> str:
        if not retrieved: return ""
        parts = []
        for s in retrieved:
            lines = [f"[Prev {s.get('stay_id','?')} {s.get('date','?')[:10]}]"]
            cs = s.get("clinical_state", {})
            if cs.get("crp") is not None: lines.append(f"  CRP:{cs['crp']}")
            if cs.get("treatment"): lines.append(f"  Tx:{cs['treatment']}")
            count = 0
            for r in s.get("records", []):
                if count >= max_lines: break
                lib = r.get("LIBELLE", "").lower()
                if any(k in lib for k in ("crp", "douleur", "traitement", "evolution")):
                    lines.append(f"  {r.get('LIBELLE','')}: {r.get('REPONSE','')}"[:80])
                    count += 1
            parts.append("\n".join(lines))
        return "\n".join(parts)


# ════════════════════════════════════════════════════════════════
# REGEX PATTERNS (precompiled)
# ════════════════════════════════════════════════════════════════

class _Rx:
    PR  = re.compile(r"polyarthrite\s+rhumato[ïi]de|PR\b", re.I)
    RF  = re.compile(r"\b(RF|facteur\s+rhumato[ïi]de)", re.I)
    CCP = re.compile(r"anti[- ]?CCP|ACPA", re.I)
    CRP = re.compile(r"CRP\s*:?\s*(\d+\.?\d*)\s*mg/L", re.I)
    ESR = re.compile(r"VS\s*:?\s*(\d+)\s*mm", re.I)
    MTX = re.compile(r"m[eé]thotrexate|MTX", re.I)
    BIO = re.compile(r"adalimumab|rituximab|tocilizumab|anti[- ]?TNF|etanercept|infliximab", re.I)
    JAK = re.compile(r"tofacitinib|baricitinib|upadacitinib", re.I)
    POS = re.compile(r"positif|positive|\+", re.I)
    NEG = re.compile(r"n[eé]gatif|negative|\u2212", re.I)

    # Joint patterns (for targeted search)
    JOINT_SMALL = re.compile(r"\bmcp\b|\bpip\b|\bmtp\b|\bpoignet|\bwrist", re.I)
    JOINT_ANY = re.compile(r"\barticul|\bjoint|\bsynovit|\bgonfle|\btumef", re.I)
    DURATION = re.compile(r"depuis\s+\d+\s*(semaines|mois|ans)|>\s*6\s*semaines|chronique|suivi\s+depuis", re.I)
    INFLAM_NEG = re.compile(r"bilan\s+(biologique\s+)?normal|pas\s+de\s+syndrome\s+inflammatoire|CRP\s+normal", re.I)


def regex_extract(records: List[Dict], stay_id: str) -> Dict:
    res: Dict[str, list] = {"disease_mentions": [], "labs": [], "drugs": []}
    for i, r in enumerate(records, 1):
        lib, rep = r.get("LIBELLE", ""), r.get("REPONSE", "")
        comb = f"{lib}: {rep}"
        ev = {"stay_id": stay_id, "line_no": i, "snippet": comb[:100]}

        if _Rx.PR.search(comb):
            res["disease_mentions"].append({"entity": "polyarthrite rhumatoide", "status": "mentioned", "evidence": ev})
        if _Rx.RF.search(lib):
            p = "positive" if _Rx.POS.search(rep) else "negative" if _Rx.NEG.search(rep) else "unknown"
            res["labs"].append({"test": "RF", "polarity": p, "evidence": ev})
        if _Rx.CCP.search(lib):
            p = "positive" if _Rx.POS.search(rep) else "negative" if _Rx.NEG.search(rep) else "unknown"
            res["labs"].append({"test": "anti-CCP", "polarity": p, "evidence": ev})
        m = _Rx.CRP.search(rep)
        if m:
            v = float(m.group(1))
            res["labs"].append({"test": "CRP", "value": f"{v} mg/L",
                                "polarity": "positive" if v > 10 else "negative", "evidence": ev})
        m2 = _Rx.ESR.search(rep)
        if m2:
            v = float(m2.group(1))
            res["labs"].append({"test": "ESR", "value": f"{v} mm/h",
                                "polarity": "positive" if v > 20 else "negative", "evidence": ev})
        if _Rx.MTX.search(comb):
            res["drugs"].append({"name": "methotrexate", "category": "csDMARD", "evidence": ev})
        if _Rx.BIO.search(comb):
            res["drugs"].append({"name": _Rx.BIO.search(comb).group(), "category": "bDMARD", "evidence": ev})
        if _Rx.JAK.search(comb):
            res["drugs"].append({"name": _Rx.JAK.search(comb).group(), "category": "tsDMARD", "evidence": ev})
    return res


# ════════════════════════════════════════════════════════════════
# TARGETED REGEX SEARCH (per dimension)
# ════════════════════════════════════════════════════════════════

# Maps dimension names to what regex patterns to look for
_DIM_REGEX_MAP = {
    "serology": {
        "patterns": [_Rx.RF, _Rx.CCP],
        "extract": lambda lib, rep, ev: [
            {"test": "RF",
             "polarity": "positive" if _Rx.POS.search(rep) else "negative" if _Rx.NEG.search(rep) else "unknown",
             "evidence": ev}
        ] if _Rx.RF.search(f"{lib} {rep}") else (
            [{"test": "anti-CCP",
              "polarity": "positive" if _Rx.POS.search(rep) else "negative" if _Rx.NEG.search(rep) else "unknown",
              "evidence": ev}]
            if _Rx.CCP.search(f"{lib} {rep}") else []
        ),
        "target_key": "labs",
    },
    "acute_phase": {
        "patterns": [_Rx.CRP, _Rx.ESR, _Rx.INFLAM_NEG],
        "target_key": "labs",
    },
    "joint": {
        "patterns": [_Rx.JOINT_SMALL, _Rx.JOINT_ANY],
        "target_key": "disease_mentions",
    },
    "duration": {
        "patterns": [_Rx.DURATION],
        "target_key": "disease_mentions",
    },
}


def targeted_regex_search(all_records: List[Dict], stay_id: str,
                          missing_dims: List[str]) -> Dict[str, Any]:
    """
    Search ALL records (unfiltered) for specific missing dimensions.
    Returns new facts found + list of confirmed absent dimensions.
    """
    new_facts: Dict[str, list] = {"disease_mentions": [], "labs": [], "drugs": []}
    found_dims: List[str] = []

    for dim in missing_dims:
        dim_config = _DIM_REGEX_MAP.get(dim)
        if not dim_config:
            continue

        dim_found = False
        for i, r in enumerate(all_records, 1):
            lib = r.get("LIBELLE", "")
            rep = r.get("REPONSE", "")
            combined = f"{lib} {rep}"
            ev = {"stay_id": stay_id, "line_no": i, "snippet": f"{lib}: {rep}"[:100]}

            for pat in dim_config["patterns"]:
                if pat.search(combined):
                    dim_found = True
                    target = dim_config["target_key"]

                    # Dimension-specific extraction
                    if dim == "serology":
                        if _Rx.RF.search(combined):
                            pol = "positive" if _Rx.POS.search(rep) else "negative" if _Rx.NEG.search(rep) else "unknown"
                            new_facts["labs"].append({"test": "RF", "polarity": pol, "evidence": ev})
                        if _Rx.CCP.search(combined):
                            pol = "positive" if _Rx.POS.search(rep) else "negative" if _Rx.NEG.search(rep) else "unknown"
                            new_facts["labs"].append({"test": "anti-CCP", "polarity": pol, "evidence": ev})

                    elif dim == "acute_phase":
                        m = _Rx.CRP.search(rep)
                        if m:
                            v = float(m.group(1))
                            new_facts["labs"].append({"test": "CRP", "value": f"{v} mg/L",
                                                      "polarity": "positive" if v > 10 else "negative", "evidence": ev})
                        elif _Rx.INFLAM_NEG.search(combined):
                            new_facts["labs"].append({"test": "CRP", "polarity": "negative",
                                                      "value": "normal", "evidence": ev})
                        m2 = _Rx.ESR.search(rep)
                        if m2:
                            v = float(m2.group(1))
                            new_facts["labs"].append({"test": "ESR", "value": f"{v} mm/h",
                                                      "polarity": "positive" if v > 20 else "negative", "evidence": ev})

                    elif dim == "joint":
                        if _Rx.JOINT_SMALL.search(combined):
                            new_facts["disease_mentions"].append({
                                "entity": "small joint involvement", "status": "mentioned", "evidence": ev})
                        elif _Rx.JOINT_ANY.search(combined):
                            new_facts["disease_mentions"].append({
                                "entity": "joint involvement", "status": "mentioned", "evidence": ev})

                    elif dim == "duration":
                        new_facts["disease_mentions"].append({
                            "entity": "symptom duration >= 6 weeks", "status": "mentioned", "evidence": ev})

                    break  # Found for this record, move to next

        if dim_found:
            found_dims.append(dim)

    confirmed_absent = [d for d in missing_dims if d not in found_dims]

    return {
        "new_facts": new_facts,
        "found_dims": found_dims,
        "confirmed_absent": confirmed_absent,
    }


# ════════════════════════════════════════════════════════════════
# PIPELINE
# ════════════════════════════════════════════════════════════════

SMART_SKIP_THRESHOLD = 2


class Agent1Pipeline:

    def __init__(self, *, llm: LLMClient, prompts: PromptLoader,
                 cfg: Agent1Config, kb_path: Optional[str] = None):
        self.llm = llm
        self.prompts = prompts
        self.cfg = cfg

        self.kb = KBManager(kb_path) if (cfg.use_kb_guidance and kb_path) else None
        self.rag = InterStayRAG(top_k=cfg.inter_stay_top_k) if cfg.use_inter_stay_rag else None

        self._kb_slots = self.kb.retrieve(cfg.kb_query, 5) if self.kb else []
        self._kb_names = [s["slot"] for s in self._kb_slots]
        self._system_prompt = self._build_system_prompt()
        self._warmup()

    def _build_system_prompt(self) -> str:
        base = self.prompts.load("agent1/system")
        if self._kb_slots and self.kb:
            kb_text = "\n".join(self.kb.slot_text(s) for s in self._kb_slots)
            return f"{base}\n\nKB Guidance:\n{kb_text}"
        return base

    def _warmup(self):
        try:
            self.llm.generate(prompt="warmup", model=self.cfg.model,
                              temperature=0, timeout_s=10,
                              extra={"keep_alive": "10m"})
        except Exception:
            pass

    # ── Standard extraction ──────────────────────────────────

    def process_stay(self, stay: Dict, patient_id: str,
                     all_stays: List[Dict]) -> Dict:
        tm = _timing()
        t0_total = time.time()

        stay_id = stay.get("stay_id", "unknown")
        visit_date = stay.get("date", "")[:10]
        visit_num = stay.get("visit_number", 0)
        records = stay.get("records", [])

        t0 = time.time()
        filtered = prefilter(records) if self.cfg.use_prefilter else records
        tm["prefilter_ms"] = (time.time() - t0) * 1000

        if not filtered:
            tm["total_ms"] = (time.time() - t0_total) * 1000
            return self._result(stay_id, patient_id, visit_date, visit_num,
                                {}, _rag_meta(), tm, "empty")

        if len(filtered) <= SMART_SKIP_THRESHOLD and self.cfg.use_regex_fallback:
            extracted = regex_extract(filtered, stay_id)
            if any(extracted[k] for k in ("disease_mentions", "labs", "drugs")):
                tm["total_ms"] = (time.time() - t0_total) * 1000
                return self._result(stay_id, patient_id, visit_date, visit_num,
                                    extracted, _rag_meta(), tm, "regex_smart_skip")

        t0 = time.time()
        temporal_ctx = ""
        rmeta = _rag_meta()
        rmeta["kb_slots_used"] = self._kb_names
        if self.rag:
            focus = self.cfg.query_focus or ["CRP", "traitement", "evolution", "douleur"]
            retrieved, scores = self.rag.retrieve(stay, all_stays, focus)
            if retrieved:
                temporal_ctx = InterStayRAG.compact_context(retrieved)
                rmeta["retrieved_previous_stays"] = [s["stay_id"] for s in retrieved]
                rmeta["temporal_context_used"] = True
                rmeta["rag_score_max"] = max(scores) if scores else 0
        tm["inter_stay_rag_ms"] = (time.time() - t0) * 1000

        t0 = time.time()
        extracted = self._llm_extract(filtered, stay_id, temporal_ctx)
        tm["llm_ms"] = (time.time() - t0) * 1000

        mode = "llm"
        if extracted is None:
            extracted = regex_extract(filtered, stay_id) if self.cfg.use_regex_fallback \
                else {"disease_mentions": [], "labs": [], "drugs": []}
            mode = "regex_fallback" if self.cfg.use_regex_fallback else "llm_failed"

        tm["total_ms"] = (time.time() - t0_total) * 1000
        return self._result(stay_id, patient_id, visit_date, visit_num,
                            extracted, rmeta, tm, mode)

    def _llm_extract(self, records: List[Dict], stay_id: str,
                     temporal_ctx: str) -> Optional[Dict]:
        lines = [f"{i}| {r.get('LIBELLE','')}: {r.get('REPONSE','')}"
                 for i, r in enumerate(records[:self.cfg.max_records_in_prompt], 1)]

        kb_guidance = "No specific KB slots provided."
        if self._kb_slots and self.kb:
            kb_guidance = "\n\n".join(self.kb.slot_text(s) for s in self._kb_slots)

        prompt = self.prompts.render(
            self.cfg.prompt_extraction,
            stay_id=stay_id, temporal_context=temporal_ctx,
            context_records="\n".join(lines))

        parsed, _ = self.llm.generate(
            prompt=prompt, model=self.cfg.model,
            temperature=self.cfg.temperature, timeout_s=self.cfg.timeout_s,
            extra={"system": self._system_prompt, "keep_alive": "10m"})

        if parsed and all(k in parsed and isinstance(parsed[k], list)
                          for k in ("disease_mentions", "labs", "drugs")):
            return parsed
        return None

    # ══════════════════════════════════════════════════════════
    # TARGETED SEARCH (called by Agent 2 feedback loop)
    # ══════════════════════════════════════════════════════════

    def targeted_search(self, raw_stay: Dict, missing_dims: List[str]) -> Dict[str, Any]:
        """
        Re-search ALL records of a stay for specific missing dimensions.

        Strategy: regex first, LLM if regex finds nothing.

        Args:
            raw_stay: original stay dict with ALL records
            missing_dims: ["serology", "joint", "acute_phase", "duration"]

        Returns:
            {
                "new_facts": {disease_mentions, labs, drugs},
                "found_dims": [...],
                "confirmed_absent": [...],
                "search_method": "regex" | "llm" | "both"
            }
        """
        stay_id = raw_stay.get("stay_id", "?")
        all_records = raw_stay.get("records", [])

        if not all_records or not missing_dims:
            return {"new_facts": {"disease_mentions": [], "labs": [], "drugs": []},
                    "found_dims": [], "confirmed_absent": missing_dims,
                    "search_method": "none"}

        # Step 1: Targeted regex on ALL records
        regex_result = targeted_regex_search(all_records, stay_id, missing_dims)

        still_missing = regex_result["confirmed_absent"]

        if not still_missing:
            # Regex found everything
            return {**regex_result, "search_method": "regex"}

        # Step 2: LLM targeted search for remaining missing dims
        llm_result = self._llm_targeted_search(all_records, stay_id, still_missing)

        # Merge results
        merged_facts = regex_result["new_facts"].copy()
        merged_found = list(regex_result["found_dims"])

        if llm_result:
            for key in ("disease_mentions", "labs", "drugs"):
                merged_facts[key] = merged_facts.get(key, []) + llm_result.get("new_facts", {}).get(key, [])
            merged_found.extend(llm_result.get("found_dims", []))
            still_missing = [d for d in still_missing if d not in llm_result.get("found_dims", [])]

        return {
            "new_facts": merged_facts,
            "found_dims": merged_found,
            "confirmed_absent": still_missing,
            "search_method": "regex" if not llm_result else "both",
        }

    def _llm_targeted_search(self, all_records: List[Dict], stay_id: str,
                             missing_dims: List[str]) -> Optional[Dict]:
        """LLM targeted search for specific missing dimensions."""

        # Build search instructions per dimension
        instructions = []
        for dim in missing_dims:
            if dim == "serology":
                instructions.append("- SEROLOGY: Look for RF (Facteur Rhumatoide), anti-CCP, ACPA. "
                                    "Any mention of these tests, even if negative.")
            elif dim == "acute_phase":
                instructions.append("- INFLAMMATION: Look for CRP, VS/ESR values. "
                                    "Also 'bilan normal', 'pas de syndrome inflammatoire' = negative evidence.")
            elif dim == "joint":
                instructions.append("- JOINTS: Look for MCP, PIP, MTP, poignet, synovite, gonflement, "
                                    "douleur articulaire, articulations touchees.")
            elif dim == "duration":
                instructions.append("- DURATION: Look for 'depuis X mois/semaines/ans', "
                                    "'chronique', 'suivi depuis', debut des symptomes.")

        search_text = "\n".join(instructions)

        # Format ALL records
        lines = [f"{i}| {r.get('LIBELLE','')}: {r.get('REPONSE','')}"
                 for i, r in enumerate(all_records[:40], 1)]

        prompt = self.prompts.render(
            "agent1/targeted_search",
            stay_id=stay_id,
            search_instructions=search_text,
            all_records="\n".join(lines))

        parsed, _ = self.llm.generate(
            prompt=prompt, model=self.cfg.model,
            temperature=self.cfg.temperature,
            timeout_s=self.cfg.timeout_s,
            extra={"keep_alive": "10m"})

        if not parsed:
            return None

        # Determine what was found
        new_facts = {"disease_mentions": [], "labs": [], "drugs": []}
        found = []

        for key in ("disease_mentions", "labs", "drugs"):
            items = parsed.get(key, [])
            if isinstance(items, list):
                new_facts[key] = [x for x in items if isinstance(x, dict)]

        # Check which dims were resolved
        if any(isinstance(l, dict) and l.get("test", "").lower() in ("rf", "anti-ccp", "acpa")
               for l in new_facts["labs"]):
            found.append("serology")
        if any(isinstance(l, dict) and l.get("test", "").lower() in ("crp", "esr", "vs")
               for l in new_facts["labs"]):
            found.append("acute_phase")
        if any(isinstance(d, dict) and ("joint" in (d.get("entity") or "").lower() or "articu" in (d.get("entity") or "").lower())
               for d in new_facts["disease_mentions"]):
            found.append("joint")
        if any(isinstance(d, dict) and ("duration" in (d.get("entity") or "").lower() or "semaine" in (d.get("entity") or "").lower())
               for d in new_facts["disease_mentions"]):
            found.append("duration")

        search_result = parsed.get("search_result", "found" if found else "confirmed_absent")

        return {
            "new_facts": new_facts,
            "found_dims": found,
            "confirmed_absent": [d for d in missing_dims if d not in found],
        }

    # ── Helpers ──────────────────────────────────────────────

    @staticmethod
    def _result(stay_id, pid, vdate, vnum, extracted, rmeta, tm, mode) -> Dict:
        return asdict(StayFacts(
            stay_id=stay_id, patient_id=pid,
            visit_date=vdate, visit_number=vnum,
            disease_mentions=extracted.get("disease_mentions", []),
            labs=extracted.get("labs", []),
            drugs=extracted.get("drugs", []),
            _rag_meta=rmeta, _timing=tm, _extraction_mode=mode))

    def _process_patient(self, pid: str, stays: List[Dict]) -> List[Dict]:
        return [self.process_stay(s, pid, stays) for s in stays]

    def process_corpus(self, corpus_path: str, output_path: str):
        corpus = IOACorpusLoader.load(corpus_path)
        idx = IOACorpusLoader.build_patient_index(corpus)
        total = sum(len(v) for v in idx.values())
        print(f"Patients: {len(idx)} | Stays: {total}")
        print("-" * 60)

        all_facts, stats, done = [], defaultdict(int), 0

        if self.cfg.max_workers <= 1:
            for pid, stays in idx.items():
                for sf in self._process_patient(pid, stays):
                    done += 1
                    stats[sf["_extraction_mode"]] += 1
                    if sf["_rag_meta"].get("temporal_context_used"): stats["temporal"] += 1
                    sym = {"llm":"+","regex_fallback":"!","regex_smart_skip":"~","empty":"-"}.get(sf["_extraction_mode"],"?")
                    print(f"  [{done}/{total}] {pid}/{sf['stay_id']} {sym} ({sf['_timing']['total_ms']:.0f}ms)")
                    all_facts.append(sf)
        else:
            with ThreadPoolExecutor(max_workers=self.cfg.max_workers) as pool:
                futs = {pool.submit(self._process_patient, pid, stays): pid for pid, stays in idx.items()}
                for fut in as_completed(futs):
                    for sf in fut.result():
                        done += 1; stats[sf["_extraction_mode"]] += 1
                        all_facts.append(sf)
                    print(f"  [{done}/{total}] Patient {futs[fut]} done")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for sf in all_facts:
                f.write(json.dumps(sf, ensure_ascii=False) + "\n")

        n = len(all_facts)
        pct = lambda k: 100 * stats[k] / n if n else 0
        print(f"\n{'='*60}\nEXTRACTION ({n} stays)\n{'='*60}")
        for k in ("llm", "regex_fallback", "regex_smart_skip", "empty"):
            print(f"  {k:<20s} {stats[k]:>4d} ({pct(k):.1f}%)")
        print(f"  temporal ctx       {stats['temporal']:>4d} ({pct('temporal'):.1f}%)")
        print(f"{'='*60}\n-> {output_path}")
