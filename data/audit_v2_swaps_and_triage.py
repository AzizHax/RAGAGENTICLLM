#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import json
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# =============================================================================
# HARDCODED PATHS (Windows-safe)
# =============================================================================

DATA_DIR = Path(r"C:\Users\Mega-Pc\Desktop\Projects\PhenoRag\fantôme\Data")

CORPUS_PATH = DATA_DIR / "phantom_ehr_corpus.txt"
ANNOTATIONS_PATH = DATA_DIR / "phantom_annotations_minimal.txt"

# Agent 1 patient-level facts (expected JSONL)
AGENT1_FACTS_PATH = DATA_DIR / "facts_agent1_patient.jsonl"

OUTPUT_DIR = DATA_DIR / "audit_out_v2"


# =============================================================================
# Regex (corpus + annotations)
# =============================================================================

RE_PATIENT = re.compile(r"^\s*PATIENT_ID:\s*(\d+)\s*$")
RE_STAY_HDR = re.compile(
    r"^\s*===\s*STAY_ID:\s*(\S+)\s*\|\s*DATE:\s*([0-9]{4}-[0-9]{2}-[0-9]{2})\s*\|\s*SERVICE:\s*(.+?)\s*===\s*$"
)

RE_RF_VAL = re.compile(
    r"\bRF\b\s*[:=]?\s*(positif|positive|n[ée]gatif|negative)?\s*(?:à|=|\()?[\s]*([0-9]+(?:[.,][0-9]+)?)\s*(?:UI/mL|IU/mL)?",
    re.IGNORECASE,
)

RE_CCP_VAL = re.compile(
    r"\banti[-\s]?CCP\b\s*[:=]?\s*(positif|positive|n[ée]gatif|negative)?\s*(?:à|=|\()?[\s]*([<>]?\s*[0-9]+(?:[.,][0-9]+)?)\s*(?:U/mL|U\/mL)?",
    re.IGNORECASE,
)

RE_RA_CONFIRMED = re.compile(
    r"\b(polyarthrite\s+rhumato[iï]de|arthrite\s+rhumato[iï]de|PR)\s+(confirm[ée]e|confirm[ée])\b",
    re.IGNORECASE,
)
RE_RA_PROBABLE = re.compile(
    r"\b(polyarthrite\s+rhumato[iï]de|arthrite\s+rhumato[iï]de|PR)\s+(probable|possible|[ée]voqu[ée]e)\b",
    re.IGNORECASE,
)
RE_RA_NEGATED = re.compile(
    r"\b(pas\s+de\s+(polyarthrite\s+rhumato[iï]de|arthrite\s+rhumato[iï]de|PR)|PR\s+[ée]cart[ée]e|diagnostic\s+non\s+retenu|ruled\s+out)\b",
    re.IGNORECASE,
)

RE_MTX = re.compile(r"\b(m[ée]thotrexate|methotrexate|MTX)\b", re.IGNORECASE)
RE_BIOLOGIC = re.compile(r"\b(adalimumab|etanercept|infliximab|tocilizumab|abatacept|rituximab)\b", re.IGNORECASE)
RE_JAK = re.compile(r"\b(tofacitinib|baricitinib|upadacitinib)\b", re.IGNORECASE)

KEYWORDS = {
    "arthrose": re.compile(r"\barthrose\b", re.IGNORECASE),
    "diverticulite": re.compile(r"\bdiverticulite\b", re.IGNORECASE),
    "aspergillose": re.compile(r"\baspergillose\b|\baspergillus\b", re.IGNORECASE),
    "sjogren": re.compile(r"\bSjögren\b|\bsjogren\b", re.IGNORECASE),
    "infection": re.compile(r"\binfection\b|\bpneumonie\b|\bpy[ée]lon[ée]phrite\b", re.IGNORECASE),
}


# =============================================================================
# Data models
# =============================================================================

@dataclass
class StayInfo:
    stay_id: str
    date: Optional[str]
    service: Optional[str]
    text: str


@dataclass
class PatientSignature:
    patient_id: str
    n_stays: int
    stay_ids: List[str]
    dates: List[str]

    rf_max_value: Optional[float]
    rf_pol: str

    ccp_max_value: Optional[float]
    ccp_pol: str

    has_ra_confirmed: bool
    has_ra_probable: bool
    has_ra_negated: bool

    has_mtx: bool
    has_biologic: bool
    has_jak: bool

    keywords_present: List[str]


@dataclass
class AnnotationRow:
    patient_id: str
    label_binary: int
    comment: str
    rf_comment: Optional[float]
    ccp_comment: Optional[float]
    keywords_comment: List[str]


@dataclass
class SwapSuggestion:
    ann_patient_id: str
    candidate_patient_id: str
    sim_score: float
    reasons: List[str]


@dataclass
class TriageRow:
    patient_id: str
    gt_label: int
    agent1_pred: Optional[int]
    agent1_score: Optional[float]
    risk_priority: str
    notes: str


# =============================================================================
# Utils
# =============================================================================

def to_float(s: Optional[str]) -> Optional[float]:
    if s is None:
        return None
    s = s.strip().replace(",", ".")
    s = re.sub(r"^[<>]\s*", "", s)
    try:
        return float(s)
    except Exception:
        return None


def infer_pol(word: Optional[str]) -> str:
    if not word:
        return "unknown"
    w = word.lower()
    if "posit" in w:
        return "positive"
    if "nég" in w or "neg" in w:
        return "negative"
    return "unknown"


def safe_int(x) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


# =============================================================================
# Parsing
# =============================================================================

def parse_corpus(corpus_text: str) -> Dict[str, List[StayInfo]]:
    patients: Dict[str, List[StayInfo]] = {}
    cur_pid: Optional[str] = None
    cur_stay_id: Optional[str] = None
    cur_date: Optional[str] = None
    cur_service: Optional[str] = None
    buf: List[str] = []

    def flush():
        nonlocal cur_pid, cur_stay_id, cur_date, cur_service, buf
        if cur_pid and cur_stay_id:
            patients.setdefault(cur_pid, []).append(
                StayInfo(stay_id=cur_stay_id, date=cur_date, service=cur_service, text="\n".join(buf).strip())
            )
        buf = []

    for line in corpus_text.splitlines():
        m_pid = RE_PATIENT.match(line)
        if m_pid:
            flush()
            cur_pid = m_pid.group(1).strip()
            cur_stay_id, cur_date, cur_service = None, None, None
            continue

        m_stay = RE_STAY_HDR.match(line)
        if m_stay:
            flush()
            cur_stay_id = m_stay.group(1).strip()
            cur_date = m_stay.group(2).strip()
            cur_service = m_stay.group(3).strip()
            continue

        if cur_pid and cur_stay_id:
            buf.append(line)

    flush()
    return patients


def parse_annotations(ann_text: str) -> Dict[str, AnnotationRow]:
    line_re = re.compile(
        r"PATIENT_ID:\s*(\d+)\s*\|\s*LABEL_BINARY:\s*([01])\s*\|\s*COMMENT:\s*(.*)$"
    )
    out: Dict[str, AnnotationRow] = {}

    for raw in ann_text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        m = line_re.search(raw)
        if not m:
            continue
        pid = m.group(1).strip()
        label = int(m.group(2))
        comment = m.group(3).strip()

        rf_c = extract_rf_from_comment(comment)
        ccp_c = extract_ccp_from_comment(comment)
        kw_c = [k for k, rx in KEYWORDS.items() if rx.search(comment)]

        out[pid] = AnnotationRow(
            patient_id=pid,
            label_binary=label,
            comment=comment,
            rf_comment=rf_c,
            ccp_comment=ccp_c,
            keywords_comment=kw_c
        )

    return out


def load_agent1_patient_facts(path: Path) -> Dict[str, Dict]:
    """
    Load Agent1 patient-level JSONL.
    Returns dict: patient_id -> raw json dict
    """
    if not path.exists():
        return {}

    out: Dict[str, Dict] = {}
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            pid = str(obj.get("patient_id", "")).strip()
            if pid:
                out[pid] = obj
    return out


# =============================================================================
# Feature extraction (corpus)
# =============================================================================

def extract_rf_features(text: str) -> Tuple[Optional[float], str]:
    vals: List[Tuple[float, str]] = []
    for m in RE_RF_VAL.finditer(text):
        pol = infer_pol(m.group(1))
        v = to_float(m.group(2))
        if v is not None:
            vals.append((v, pol))
    if not vals:
        return None, "absent"
    max_v, pol = sorted(vals, key=lambda x: x[0], reverse=True)[0]
    return max_v, pol if pol != "unknown" else "unknown"


def extract_ccp_features(text: str) -> Tuple[Optional[float], str]:
    vals: List[Tuple[float, str]] = []
    for m in RE_CCP_VAL.finditer(text):
        pol = infer_pol(m.group(1))
        v = to_float(m.group(2))
        if v is not None:
            vals.append((v, pol))
    if not vals:
        return None, "absent"
    max_v, pol = sorted(vals, key=lambda x: x[0], reverse=True)[0]
    return max_v, pol if pol != "unknown" else "unknown"


def extract_ra_flags(text: str) -> Tuple[bool, bool, bool]:
    return bool(RE_RA_CONFIRMED.search(text)), bool(RE_RA_PROBABLE.search(text)), bool(RE_RA_NEGATED.search(text))


def extract_drug_flags(text: str) -> Tuple[bool, bool, bool]:
    return bool(RE_MTX.search(text)), bool(RE_BIOLOGIC.search(text)), bool(RE_JAK.search(text))


def extract_keywords(text: str) -> List[str]:
    return [k for k, rx in KEYWORDS.items() if rx.search(text)]


def build_signature(pid: str, stays: List[StayInfo]) -> PatientSignature:
    full_text = "\n".join(
        [f"=== {s.stay_id} {s.date or ''} {s.service or ''} ===\n{s.text}" for s in stays]
    )

    rf_v, rf_pol = extract_rf_features(full_text)
    ccp_v, ccp_pol = extract_ccp_features(full_text)
    ra_conf, ra_prob, ra_neg = extract_ra_flags(full_text)
    has_mtx, has_bio, has_jak = extract_drug_flags(full_text)
    kws = extract_keywords(full_text)

    stay_ids = [s.stay_id for s in stays]
    dates = [s.date for s in stays if s.date]

    return PatientSignature(
        patient_id=pid,
        n_stays=len(stays),
        stay_ids=stay_ids,
        dates=dates,
        rf_max_value=rf_v,
        rf_pol=rf_pol,
        ccp_max_value=ccp_v,
        ccp_pol=ccp_pol,
        has_ra_confirmed=ra_conf,
        has_ra_probable=ra_prob,
        has_ra_negated=ra_neg,
        has_mtx=has_mtx,
        has_biologic=has_bio,
        has_jak=has_jak,
        keywords_present=kws
    )


# =============================================================================
# Comment parsing (annotation)
# =============================================================================

def extract_rf_from_comment(comment: str) -> Optional[float]:
    m = re.search(r"\bRF\+?\s*(?:à|=)?\s*([0-9]+(?:[.,][0-9]+)?)", comment, re.IGNORECASE)
    return to_float(m.group(1)) if m else None


def extract_ccp_from_comment(comment: str) -> Optional[float]:
    m = re.search(r"\banti[-\s]?CCP\+?\s*(?:à|=)?\s*([0-9]+(?:[.,][0-9]+)?)", comment, re.IGNORECASE)
    return to_float(m.group(1)) if m else None


# =============================================================================
# Similarity scoring (swap detector)
# =============================================================================

def sim_bool(a: bool, b: bool) -> float:
    return 1.0 if a == b else 0.0


def sim_value(comment_val: Optional[float], corpus_val: Optional[float], tol: float = 5.0) -> float:
    """
    Score 1 if close, 0.5 if present but far, 0 if missing on one side.
    """
    if comment_val is None:
        return 0.5  # neutral (comment didn't mention it)
    if corpus_val is None:
        return 0.0
    if abs(corpus_val - comment_val) <= tol:
        return 1.0
    return 0.5


def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def compute_similarity(ann: AnnotationRow, sig: PatientSignature) -> Tuple[float, List[str]]:
    reasons: List[str] = []

    # components (weights sum to 1)
    w_rf = 0.20
    w_ccp = 0.25
    w_kw = 0.25
    w_drugs = 0.15
    w_raflags = 0.15

    rf_s = sim_value(ann.rf_comment, sig.rf_max_value, tol=5.0)
    ccp_s = sim_value(ann.ccp_comment, sig.ccp_max_value, tol=5.0)
    kw_s = jaccard(ann.keywords_comment, sig.keywords_present)

    drugs_s = (sim_bool("mtx" in ann.comment.lower(), sig.has_mtx) +
               sim_bool(any(x in ann.comment.lower() for x in ["biothérapie", "biologique", "etanercept", "adalimumab", "tocilizumab", "abatacept", "rituximab"]),
                        sig.has_biologic) +
               sim_bool(any(x in ann.comment.lower() for x in ["jak", "tofacitinib", "baricitinib", "upadacitinib"]),
                        sig.has_jak)) / 3.0

    raflags_s = (sim_bool("confirm" in ann.comment.lower(), sig.has_ra_confirmed) +
                 sim_bool(any(x in ann.comment.lower() for x in ["probable", "possible", "évoqué", "evoque"]), sig.has_ra_probable) +
                 sim_bool(any(x in ann.comment.lower() for x in ["écartée", "ecartee", "pas de"]), sig.has_ra_negated)) / 3.0

    score = (w_rf * rf_s + w_ccp * ccp_s + w_kw * kw_s + w_drugs * drugs_s + w_raflags * raflags_s)

    # Reasons for interpretability
    if ann.rf_comment is not None and sig.rf_max_value is not None and abs(sig.rf_max_value - ann.rf_comment) <= 5:
        reasons.append(f"rf≈{ann.rf_comment}")
    if ann.ccp_comment is not None and sig.ccp_max_value is not None and abs(sig.ccp_max_value - ann.ccp_comment) <= 5:
        reasons.append(f"ccp≈{ann.ccp_comment}")
    inter_kw = sorted(list(set(ann.keywords_comment) & set(sig.keywords_present)))
    if inter_kw:
        reasons.append(f"kw:{','.join(inter_kw)}")
    if sig.has_biologic:
        reasons.append("biologic")
    if sig.has_mtx:
        reasons.append("mtx")
    if sig.has_ra_confirmed:
        reasons.append("ra_confirmed")

    return score, reasons


def top_k_swap_suggestions(ann: AnnotationRow, signatures: Dict[str, PatientSignature], k: int = 3) -> List[SwapSuggestion]:
    scored: List[Tuple[str, float, List[str]]] = []
    for pid, sig in signatures.items():
        sc, reasons = compute_similarity(ann, sig)
        scored.append((pid, sc, reasons))
    scored.sort(key=lambda x: x[1], reverse=True)

    out: List[SwapSuggestion] = []
    for cand_pid, sc, reasons in scored[:k]:
        out.append(SwapSuggestion(
            ann_patient_id=ann.patient_id,
            candidate_patient_id=cand_pid,
            sim_score=float(sc),
            reasons=reasons
        ))
    return out


# =============================================================================
# Agent1 quick prediction helper (RA ever)
# =============================================================================

def agent1_predict_ra_ever(agent1_obj: Dict) -> Tuple[Optional[int], Optional[float]]:
    """
    Heuristic: RA+ if any evidence of confirmed RA or biologic/JAK or (anti-CCP positive) or (RF high).
    Returns (pred_label, score_proxy).
    This is NOT your real scorer; it's only for triage.
    """
    if not agent1_obj:
        return None, None

    score = 0.0

    # disease mentions
    for dm in agent1_obj.get("disease_mentions", []) or []:
        ent = str(dm.get("entity", "")).lower()
        status = str(dm.get("status", "")).lower()
        if "polyarth" in ent or "arthrite rhumato" in ent or ent.strip() in ["pr", "ra", "rheumatoid arthritis"]:
            if status in ["confirmed"]:
                score += 3.0
            elif status in ["suspected", "possible", "probable"]:
                score += 1.0
            elif status in ["negated"]:
                score -= 2.0

    # labs
    for lab in agent1_obj.get("labs", []) or []:
        test = str(lab.get("test", "")).lower()
        pol = str(lab.get("polarity", "")).lower()
        val = str(lab.get("value", "")).lower()
        if "anti" in test and "ccp" in test:
            if pol == "positive" or "posit" in val:
                score += 3.0
        if test in ["rf", "facteur rhumatoïde", "rheumatoid factor"] or "rf" == test:
            if pol == "positive" or "posit" in val:
                score += 2.0

    # drugs
    for drug in agent1_obj.get("drugs", []) or []:
        name = str(drug.get("name", "")).lower()
        cat = str(drug.get("category", "")).lower()
        if "methotrex" in name or "mtx" in name:
            score += 1.0
        if cat in ["bdmard", "tsdmard"] or any(x in name for x in ["adalimumab", "etanercept", "tocilizumab", "abatacept", "rituximab", "tofacitinib", "baricitinib", "upadacitinib"]):
            score += 3.0

    pred = 1 if score >= 3.0 else 0
    return pred, score


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 80)
    print("AUDIT v2: Swap detector + Triage (RA ever) + Compare Agent1")
    print("=" * 80)

    if not CORPUS_PATH.exists():
        print(f"[ERROR] Corpus not found:\n  {CORPUS_PATH}")
        return
    if not ANNOTATIONS_PATH.exists():
        print(f"[ERROR] Annotations not found:\n  {ANNOTATIONS_PATH}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    corpus_text = CORPUS_PATH.read_text(encoding="utf-8", errors="replace")
    ann_text = ANNOTATIONS_PATH.read_text(encoding="utf-8", errors="replace")

    corpus_patients = parse_corpus(corpus_text)
    annotations = parse_annotations(ann_text)

    signatures: Dict[str, PatientSignature] = {}
    for pid, stays in corpus_patients.items():
        signatures[pid] = build_signature(pid, stays)

    agent1 = load_agent1_patient_facts(AGENT1_FACTS_PATH)

    # -------------------------------------------------------------------------
    # 1) Swap suggestions for each annotation row
    # -------------------------------------------------------------------------
    swaps_csv = OUTPUT_DIR / "swap_suggestions_top3.csv"
    swaps_jsonl = OUTPUT_DIR / "swap_suggestions_top3.jsonl"

    swap_rows: List[Dict] = []

    for pid, ann in annotations.items():
        top3 = top_k_swap_suggestions(ann, signatures, k=3)

        # we flag potential swap if best candidate isn't itself AND score gap is strong
        best = top3[0]
        second = top3[1] if len(top3) > 1 else None
        self_score = None
        for s in top3:
            if s.candidate_patient_id == pid:
                self_score = s.sim_score

        swap_flag = False
        if best.candidate_patient_id != pid:
            gap = best.sim_score - (self_score if self_score is not None else 0.0)
            # conservative thresholds
            if best.sim_score >= 0.65 and gap >= 0.10:
                swap_flag = True

        rec = {
            "ann_patient_id": pid,
            "label_binary": ann.label_binary,
            "swap_flag": swap_flag,
            "best_candidate": best.candidate_patient_id,
            "best_score": round(best.sim_score, 4),
            "best_reasons": ",".join(best.reasons),
            "self_score": round(self_score, 4) if self_score is not None else "",
            "cand2": top3[1].candidate_patient_id if len(top3) > 1 else "",
            "cand2_score": round(top3[1].sim_score, 4) if len(top3) > 1 else "",
            "cand3": top3[2].candidate_patient_id if len(top3) > 2 else "",
            "cand3_score": round(top3[2].sim_score, 4) if len(top3) > 2 else "",
            "comment": ann.comment
        }
        swap_rows.append(rec)

    # write CSV
    with swaps_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(swap_rows[0].keys()))
        w.writeheader()
        w.writerows(swap_rows)

    # write JSONL
    with swaps_jsonl.open("w", encoding="utf-8") as f:
        for r in swap_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # -------------------------------------------------------------------------
    # 2) Auto-proposed remap file (only for swap_flag==True)
    # -------------------------------------------------------------------------
    remap_path = OUTPUT_DIR / "proposed_remap.tsv"
    flagged = [r for r in swap_rows if r["swap_flag"]]

    with remap_path.open("w", encoding="utf-8") as f:
        f.write("ann_patient_id\tbest_candidate\tbest_score\tcomment\n")
        for r in flagged:
            f.write(f"{r['ann_patient_id']}\t{r['best_candidate']}\t{r['best_score']}\t{r['comment']}\n")

    # -------------------------------------------------------------------------
    # 3) Triage list: compare GT vs Agent1 (quick proxy) + prioritize
    # -------------------------------------------------------------------------
    triage_csv = OUTPUT_DIR / "triage_gt_vs_agent1.csv"
    triage_jsonl = OUTPUT_DIR / "triage_gt_vs_agent1.jsonl"

    triage_rows: List[TriageRow] = []
    for pid, ann in annotations.items():
        a1_obj = agent1.get(pid, {})
        a1_pred, a1_score = agent1_predict_ra_ever(a1_obj)

        # priority: high if mismatch OR swap_flag
        swap_flag = any(r["ann_patient_id"] == pid and r["swap_flag"] for r in swap_rows)
        mismatch = (a1_pred is not None and a1_pred != ann.label_binary)

        if swap_flag or mismatch:
            risk = "HIGH"
        else:
            risk = "LOW"

        notes = []
        if swap_flag:
            rec = next(r for r in swap_rows if r["ann_patient_id"] == pid)
            notes.append(f"swap? best={rec['best_candidate']} score={rec['best_score']} self={rec['self_score']}")
        if mismatch:
            notes.append("agent1_vs_gt_mismatch")
        triage_rows.append(TriageRow(
            patient_id=pid,
            gt_label=ann.label_binary,
            agent1_pred=a1_pred,
            agent1_score=round(a1_score, 3) if a1_score is not None else None,
            risk_priority=risk,
            notes=" | ".join(notes) if notes else ""
        ))

    # save triage
    with triage_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["patient_id", "gt_label", "agent1_pred", "agent1_score", "risk_priority", "notes"])
        for t in triage_rows:
            w.writerow([t.patient_id, t.gt_label, t.agent1_pred, t.agent1_score, t.risk_priority, t.notes])

    with triage_jsonl.open("w", encoding="utf-8") as f:
        for t in triage_rows:
            f.write(json.dumps(asdict(t), ensure_ascii=False) + "\n")

    # -------------------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------------------
    print(f"\n[OK] Saved swap suggestions CSV:  {swaps_csv}")
    print(f"[OK] Saved swap suggestions JSONL:{swaps_jsonl}")
    print(f"[OK] Saved proposed remap TSV:    {remap_path}")
    print(f"[OK] Saved triage CSV:            {triage_csv}")
    print(f"[OK] Saved triage JSONL:          {triage_jsonl}")

    print("\n--- Potential swaps (swap_flag==True) ---")
    if not flagged:
        print("None flagged.")
    else:
        for r in flagged[:10]:
            print(f"- ann={r['ann_patient_id']} -> best={r['best_candidate']} score={r['best_score']} (self={r['self_score']}) reasons={r['best_reasons']}")

    high = [t for t in triage_rows if t.risk_priority == "HIGH"]
    print("\n--- Triage (HIGH priority) ---")
    if not high:
        print("None high priority.")
    else:
        for t in high[:15]:
            print(f"- {t.patient_id}: GT={t.gt_label} A1={t.agent1_pred} score={t.agent1_score} notes={t.notes}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
