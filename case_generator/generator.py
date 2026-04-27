#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PHANTOM GENERATOR v3 (robust)
- Forces STAY_ID allocation (S0001..)
- Two-step generation per batch:
  (1) Generate EHR stays only
  (2) Generate gold annotations only (based on generated EHR)
- Retries + optional batch-size backoff

Outputs:
- phantom_ehr_corpus_v3.txt
- phantom_annotations_minimal_v3.txt
- generation_log_v3.jsonl
"""

import json
import re
import time
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import requests


# =============================================================================
# CONFIG
# =============================================================================

DATA_DIR = Path(r"C:\Users\Mega-Pc\Desktop\Projects\PhenoRag\fantôme\Data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

OUT_CORPUS = DATA_DIR / "phantom_ehr_corpus_v3.txt"
OUT_ANN = DATA_DIR / "phantom_annotations_minimal_v3.txt"
OUT_LOG = DATA_DIR / "generation_log_v3.jsonl"

OLLAMA_URL = "http://localhost:11434"
MODEL = "qwen2.5:3b-instruct"

N_PATIENTS = 100
TARGET_STAYS = 300
MIN_STAYS_PER_PAT = 1
MAX_STAYS_PER_PAT = 5

BATCH_SIZE = 10           # will auto-backoff to 5/3 if failures
TEMPERATURE = 0.25
TIMEOUT_S = 240
MAX_RETRIES = 3


# =============================================================================
# OLLAMA
# =============================================================================

def ollama_generate(prompt: str,
                    model: str = MODEL,
                    temperature: float = TEMPERATURE,
                    timeout_s: int = TIMEOUT_S) -> str:
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "temperature": temperature,
        },
        timeout=timeout_s
    )
    r.raise_for_status()
    return (r.json().get("response") or "").strip()


def write_log(obj: Dict[str, Any]):
    with open(OUT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# =============================================================================
# PLANNING
# =============================================================================

def plan_stays(n_patients: int,
               target_stays: int,
               min_stays: int,
               max_stays: int,
               rng: random.Random) -> Dict[str, int]:
    pids = [str(i).zfill(4) for i in range(1, n_patients + 1)]
    plan = {pid: rng.randint(min_stays, max_stays) for pid in pids}
    total = sum(plan.values())

    while total < target_stays:
        pid = rng.choice(pids)
        if plan[pid] < max_stays:
            plan[pid] += 1
            total += 1
    while total > target_stays:
        pid = rng.choice(pids)
        if plan[pid] > min_stays:
            plan[pid] -= 1
            total -= 1

    return plan


def allocate_stay_ids(stays_plan: Dict[str, int]) -> Dict[str, List[str]]:
    all_patients = sorted(stays_plan.keys())
    stay_counter = 1
    alloc: Dict[str, List[str]] = {}
    for pid in all_patients:
        alloc[pid] = []
        for _ in range(stays_plan[pid]):
            alloc[pid].append(f"S{stay_counter:04d}")
            stay_counter += 1
    return alloc


# =============================================================================
# PARSING / VALIDATION
# =============================================================================

PATIENT_HEADER_RE = re.compile(r"^PATIENT_ID:\s*(\d{4})\s*$")
STAY_HEADER_RE = re.compile(r"^===\s*STAY_ID:\s*(S\d{4})\s*===\s*$")
ANN_LINE_RE = re.compile(
    r"^PATIENT_ID:\s*(\d{4})\s*\|\s*LABEL_BINARY:\s*([01])\s*\|\s*ACR_EULAR_SCORE:\s*(\d{1,2})/10\s*\|\s*COMMENT:\s*(.+)$"
)

def parse_corpus(text: str) -> Dict[str, List[Tuple[str, str]]]:
    lines = [ln.rstrip("\n") for ln in text.splitlines()]
    out: Dict[str, List[Tuple[str, str]]] = {}
    current_pid: Optional[str] = None
    current_stay: Optional[str] = None
    buf: List[str] = []

    def flush_stay():
        nonlocal buf, current_pid, current_stay
        if current_pid and current_stay:
            out.setdefault(current_pid, []).append((current_stay, "\n".join(buf).strip()))
        buf = []

    for ln in lines:
        m_pid = PATIENT_HEADER_RE.match(ln.strip())
        if m_pid:
            flush_stay()
            current_pid = m_pid.group(1)
            current_stay = None
            continue

        m_stay = STAY_HEADER_RE.match(ln.strip())
        if m_stay:
            flush_stay()
            current_stay = m_stay.group(1)
            continue

        if current_pid is not None:
            buf.append(ln)

    flush_stay()
    return out


def parse_annotations(ann_text: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for ln in ann_text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        m = ANN_LINE_RE.match(ln)
        if not m:
            continue
        pid = m.group(1)
        label = int(m.group(2))
        acr = max(0, min(10, int(m.group(3))))
        comment = m.group(4).strip()
        out[pid] = {"label_binary": label, "acr_eular_score": acr, "comment": comment}
    return out


def validate_ehr_batch(patient_ids: List[str],
                       stay_ids_alloc: Dict[str, List[str]],
                       corpus_map: Dict[str, List[Tuple[str, str]]]) -> List[str]:
    errors = []

    for pid in patient_ids:
        if pid not in corpus_map:
            errors.append(f"Missing patient in corpus: {pid}")
            continue

        expected_ids = stay_ids_alloc[pid]
        found_ids = [sid for sid, _ in corpus_map[pid]]

        if len(found_ids) != len(expected_ids):
            errors.append(f"Stay count mismatch for {pid}: expected {len(expected_ids)}, found {len(found_ids)}")

        # ensure the exact stay ids are present, in any order (better: exact order)
        missing = [sid for sid in expected_ids if sid not in found_ids]
        extra = [sid for sid in found_ids if sid not in expected_ids]
        if missing:
            errors.append(f"{pid}: Missing STAY_ID(s): {missing[:5]}")
        if extra:
            errors.append(f"{pid}: Extra STAY_ID(s): {extra[:5]}")

        for sid, txt in corpus_map[pid]:
            if not sid or not re.match(r"^S\d{4}$", sid):
                errors.append(f"Invalid STAY_ID for {pid}: {sid}")
            if not txt.strip():
                errors.append(f"Empty stay text for {pid}/{sid}")

    # global uniqueness
    all_ids = []
    for pid in patient_ids:
        for sid, _ in corpus_map.get(pid, []):
            all_ids.append(sid)
    if len(all_ids) != len(set(all_ids)):
        errors.append("Duplicate STAY_ID inside batch")

    return errors


def validate_ann_batch(patient_ids: List[str], ann_map: Dict[str, Dict[str, Any]]) -> List[str]:
    errors = []
    for pid in patient_ids:
        if pid not in ann_map:
            errors.append(f"Missing patient in annotations: {pid}")
            continue
        acr = ann_map[pid].get("acr_eular_score")
        if not isinstance(acr, int) or acr < 0 or acr > 10:
            errors.append(f"Invalid ACR score for {pid}: {acr}")
    return errors


# =============================================================================
# PROMPTS (2 phases)
# =============================================================================

def build_ehr_prompt(patient_ids: List[str],
                     stay_ids_alloc: Dict[str, List[str]],
                     seed: int) -> str:
    plan_lines = []
    for pid in patient_ids:
        plan_lines.append(f"- PATIENT_ID: {pid} | STAYS: {', '.join(stay_ids_alloc[pid])}")
    plan_text = "\n".join(plan_lines)

    return f"""
You generate realistic hospital EHR notes (French with occasional English).
You MUST follow the exact patient plan (patient IDs + exact STAY_ID list).

RANDOM SEED: {seed}

PATIENT PLAN (MUST FOLLOW EXACTLY):
{plan_text}

OUTPUT FORMAT ONLY (no extra text):

PATIENT_ID: 0001
=== STAY_ID: S0001 ===
<clinical note>

=== STAY_ID: S0002 ===
<clinical note>

Rules:
- Use EXACTLY the STAY_IDs given in the plan for each patient.
- Each stay 50–200 words usually; some shorter; rare longer.
- Strong heterogeneity:
  - Some stays have NO RA signal (pure noise).
  - Some have RA negations / differential diagnoses.
  - Some have contradictions across stays (e.g., RF+ then RF−).
  - Not everyone has drugs; not everyone has labs.
- Include labs sometimes: RF, anti-CCP/ACPA, CRP, ESR/VHS with numeric values.
- Include treatments sometimes: MTX, biologics, steroids, NSAIDs; MTX sometimes for non-RA.

CRITICAL:
- Do not invent extra patients.
- Do not invent extra STAY_ID.
- Do not output any section headers, only the corpus.
""".strip()


def build_ann_prompt(patient_ids: List[str],
                     ehr_text: str,
                     seed: int) -> str:
    # keep prompt small-ish: only include the chunk relevant to this batch
    return f"""
You are a rheumatologist creating GOLD annotations for RA phenotyping from EHR notes.

RANDOM SEED: {seed}

EHR INPUT:
{ehr_text}

TASK:
For each patient listed below, output exactly one line:

PATIENT_ID: 0001 | LABEL_BINARY: 1 | ACR_EULAR_SCORE: X/10 | COMMENT: <short justification>

Rules:
- LABEL_BINARY: 1 = RA+ ; 0 = RA−
- ACR_EULAR_SCORE is integer 0-10.
- COMMENT short but specific (serology/joints/negation/treatment).
- Be consistent with the EHR above.
- If uncertain, keep RA− unless strong RA evidence.

Patients to annotate (MUST include all):
{", ".join(patient_ids)}

CRITICAL:
- Output ONLY the annotation lines. No extra text.
""".strip()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("PHANTOM GENERATOR v3 (robust, 2-phase) - Ollama")
    print("=" * 80)
    print(f"[Config] Model: {MODEL}")
    print(f"[Config] Output corpus: {OUT_CORPUS}")
    print(f"[Config] Output ann:    {OUT_ANN}")
    print(f"[Config] Output log:    {OUT_LOG}")
    print()

    # reset
    for p in [OUT_CORPUS, OUT_ANN, OUT_LOG]:
        if p.exists():
            p.unlink()

    rng = random.Random(1337)
    stays_plan = plan_stays(N_PATIENTS, TARGET_STAYS, MIN_STAYS_PER_PAT, MAX_STAYS_PER_PAT, rng)
    stay_ids_alloc = allocate_stay_ids(stays_plan)

    all_patient_ids = [str(i).zfill(4) for i in range(1, N_PATIENTS + 1)]

    # dynamic batch sizes if needed
    batch_sizes_to_try = [BATCH_SIZE, 5, 3]

    corpus_chunks: List[str] = []
    ann_lines: List[str] = []

    i = 0
    batch_index = 0
    while i < len(all_patient_ids):
        batch_index += 1

        # choose batch size (start with default)
        bs = batch_sizes_to_try[0]
        patient_ids = all_patient_ids[i:i+bs]

        def run_batch(patient_ids_local: List[str]) -> Tuple[str, str]:
            # Phase A: EHR
            seed = 2000 + batch_index
            ehr_prompt = build_ehr_prompt(patient_ids_local, stay_ids_alloc, seed)
            ehr_text = ollama_generate(ehr_prompt)
            corpus_map = parse_corpus(ehr_text)
            ehr_errors = validate_ehr_batch(patient_ids_local, stay_ids_alloc, corpus_map)

            if ehr_errors:
                raise ValueError(f"EHR validation failed: {ehr_errors[0]}")

            # Phase B: Annotations
            seed2 = 3000 + batch_index
            ann_prompt = build_ann_prompt(patient_ids_local, ehr_text, seed2)
            ann_text = ollama_generate(ann_prompt)
            ann_map = parse_annotations(ann_text)
            ann_errors = validate_ann_batch(patient_ids_local, ann_map)

            if ann_errors:
                raise ValueError(f"ANN validation failed: {ann_errors[0]}")

            return ehr_text.strip(), ann_text.strip()

        success = False
        for bs_try in batch_sizes_to_try:
            patient_ids = all_patient_ids[i:i+bs_try]
            if not patient_ids:
                break

            print(f"[Batch {batch_index}] Patients {patient_ids[0]}..{patient_ids[-1]} (size={len(patient_ids)})")

            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    t0 = time.time()
                    ehr_text, ann_text = run_batch(patient_ids)
                    dt = time.time() - t0

                    write_log({
                        "batch_index": batch_index,
                        "patients": patient_ids,
                        "attempt": attempt,
                        "batch_size": len(patient_ids),
                        "elapsed_s": round(dt, 2),
                        "status": "ok"
                    })

                    corpus_chunks.append(ehr_text + "\n")
                    for ln in ann_text.splitlines():
                        ln = ln.strip()
                        if ANN_LINE_RE.match(ln):
                            ann_lines.append(ln)

                    print(f"  ✓ OK | elapsed {dt:.1f}s")
                    success = True
                    break

                except Exception as e:
                    write_log({
                        "batch_index": batch_index,
                        "patients": patient_ids,
                        "attempt": attempt,
                        "batch_size": len(patient_ids),
                        "status": "error",
                        "error": f"{type(e).__name__}: {e}"
                    })
                    print(f"  ✗ attempt {attempt}/{MAX_RETRIES}: {type(e).__name__}: {e}")
                    time.sleep(1.5)

            if success:
                i += len(patient_ids)
                break
            else:
                print(f"  ⚠ Switching to smaller batch size (next={bs_try})")

        if not success:
            raise RuntimeError(f"Batch {batch_index} failed after retries and backoff.")

    # Write final outputs
    corpus_final = "\n".join(corpus_chunks).strip() + "\n"
    ann_final = "\n".join(ann_lines).strip() + "\n"

    with open(OUT_CORPUS, "w", encoding="utf-8") as f:
        f.write(corpus_final)
    with open(OUT_ANN, "w", encoding="utf-8") as f:
        f.write(ann_final)

    # Global checks
    corpus_map_all = parse_corpus(corpus_final)
    ann_map_all = parse_annotations(ann_final)
    total_stays = sum(len(v) for v in corpus_map_all.values())
    all_stay_ids = [sid for stays in corpus_map_all.values() for sid, _ in stays]
    dup_stays = len(all_stay_ids) - len(set(all_stay_ids))

    missing_corpus = [pid for pid in all_patient_ids if pid not in corpus_map_all]
    missing_ann = [pid for pid in all_patient_ids if pid not in ann_map_all]

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"[Stats] patients expected: {N_PATIENTS}")
    print(f"[Stats] patients in corpus: {len(corpus_map_all)} (missing: {len(missing_corpus)})")
    print(f"[Stats] patients in ann:    {len(ann_map_all)} (missing: {len(missing_ann)})")
    print(f"[Stats] stays total:        {total_stays} (target ~{TARGET_STAYS})")
    print(f"[Stats] stay_id duplicates: {dup_stays}")
    print(f"[Output] corpus: {OUT_CORPUS}")
    print(f"[Output] ann:    {OUT_ANN}")
    print(f"[Output] log:    {OUT_LOG}")

    if missing_corpus or missing_ann or dup_stays:
        print("\n⚠ WARNING: global issues detected")
        if missing_corpus:
            print(f"  - Missing in corpus: {missing_corpus[:10]}")
        if missing_ann:
            print(f"  - Missing in ann: {missing_ann[:10]}")
        if dup_stays:
            print("  - Duplicate STAY_ID detected (should not happen with enforced plan).")


if __name__ == "__main__":
    main()
