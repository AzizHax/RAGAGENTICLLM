#!/usr/bin/env python3
"""
re_evaluate.py — Ré-évaluation post-hoc des stratégies d'agrégation
====================================================================

Prend un `decisions.jsonl` déjà produit + un GT CSV séjour-level,
recalcule les métriques pour différentes stratégies d'agrégation
patient SANS relancer le LLM.

Évalue à 2 niveaux : séjour (NDA) et patient (NIP).

Usage
-----
    python re_evaluate.py \
        --decisions runs/b1/decisions.jsonl \
        --gt-csv data/real/ground_truth.csv \
        --output runs/b1/re_evaluation.json

Sortie : un JSON + un tableau ASCII en console comparant toutes les
stratégies d'agrégation sur les mêmes prédictions séjour.

Format CSV GT attendu :
    id,NIP,NDA,PR
    1,123456,789012,F
    2,123456,789013,T
    3,234567,890123,D

PR ∈ {T, F, D}. D est traité comme F (RA-) pour l'évaluation,
mais rapporté séparément.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


# ════════════════════════════════════════════════════════════════
# CHARGEMENT GT
# ════════════════════════════════════════════════════════════════

def load_gt(csv_path: str) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    """
    Charge le GT séjour. Retourne :
      - stay_labels : {NDA: "RA+"|"RA-"}
      - raw_pr      : {NDA: "T"|"F"|"D"}
      - nip_by_nda  : {NDA: NIP}
    """
    PR_MAP = {"T": "RA+", "F": "RA-", "D": "RA-"}   # D traité comme RA-
    stay_labels: Dict[str, str] = {}
    raw_pr: Dict[str, str] = {}
    nip_by_nda: Dict[str, str] = {}

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            nip = (row.get("NIP") or "").strip()
            nda = (row.get("NDA") or "").strip()
            pr = (row.get("PR") or "").strip().upper().strip('"')
            if not nip or not nda or pr not in PR_MAP:
                continue
            stay_labels[nda] = PR_MAP[pr]
            raw_pr[nda] = pr
            nip_by_nda[nda] = nip
    return stay_labels, raw_pr, nip_by_nda


def derive_patient_gt(stay_labels: Dict[str, str], nip_by_nda: Dict[str, str],
                       strategy: str = "any_positive") -> Dict[str, str]:
    """Agrège séjour → patient pour le ground truth."""
    nip_to_stays: Dict[str, List[str]] = defaultdict(list)
    for nda, label in stay_labels.items():
        nip = nip_by_nda[nda]
        nip_to_stays[nip].append(label)

    patient_gt: Dict[str, str] = {}
    for nip, labels in nip_to_stays.items():
        n_pos = labels.count("RA+")
        n_total = len(labels)
        if strategy == "any_positive":
            patient_gt[nip] = "RA+" if n_pos > 0 else "RA-"
        elif strategy == "majority":
            patient_gt[nip] = "RA+" if n_pos > n_total / 2 else "RA-"
        elif strategy == "all_positive":
            patient_gt[nip] = "RA+" if n_pos == n_total else "RA-"
    return patient_gt


# ════════════════════════════════════════════════════════════════
# AGRÉGATION DES PRÉDICTIONS PATIENT
# ════════════════════════════════════════════════════════════════

def aggregate_patient(stay_details: List[Dict[str, Any]], strategy: str,
                       proportion_threshold: float = 0.5) -> str:
    """
    Recalcule le label patient à partir des stay_details.

    Stratégies :
      - any_positive  : RA+ si ≥ 1 séjour RA+
      - majority      : RA+ si > 50% des séjours RA+
      - confirmed     : RA+ si ≥ 1 séjour avec ACR ≥ 8 (haute confiance clinique)
      - proportion    : RA+ si proportion ≥ seuil
    """
    if not stay_details:
        return "RA-"

    n = len(stay_details)
    n_pos = sum(1 for s in stay_details if s.get("label") == "RA+")

    if strategy == "any_positive":
        return "RA+" if n_pos > 0 else "RA-"

    if strategy == "majority":
        return "RA+" if n_pos > n / 2 else "RA-"

    if strategy == "confirmed":
        # Au moins 1 séjour avec ACR ≥ 8 (critère clinique strict)
        for s in stay_details:
            if s.get("acr_score") is not None and s["acr_score"] >= 8:
                return "RA+"
        return "RA-"

    if strategy == "proportion":
        return "RA+" if (n_pos / n) >= proportion_threshold else "RA-"

    raise ValueError(f"Stratégie inconnue : {strategy}")


# ════════════════════════════════════════════════════════════════
# MÉTRIQUES
# ════════════════════════════════════════════════════════════════

def compute_metrics(tp: int, fp: int, tn: int, fn: int) -> Dict[str, float]:
    total = tp + fp + tn + fn
    acc = (tp + tn) / total if total else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn, "n": total,
        "accuracy": round(acc, 4), "precision": round(prec, 4),
        "recall": round(rec, 4), "f1": round(f1, 4), "specificity": round(spec, 4),
    }


def evaluate_patient_level(decisions: List[Dict[str, Any]],
                            patient_gt: Dict[str, str],
                            strategy: str,
                            proportion_threshold: float = 0.5) -> Dict[str, Any]:
    tp = fp = tn = fn = 0
    for d in decisions:
        nip = str(d.get("patient_id", ""))
        if nip not in patient_gt:
            continue
        true = patient_gt[nip]
        pred = aggregate_patient(d.get("stay_details", []), strategy, proportion_threshold)
        if pred == "RA+" and true == "RA+": tp += 1
        elif pred == "RA+" and true == "RA-": fp += 1
        elif pred == "RA-" and true == "RA-": tn += 1
        elif pred == "RA-" and true == "RA+": fn += 1
    m = compute_metrics(tp, fp, tn, fn)
    m["strategy"] = strategy
    if strategy == "proportion":
        m["threshold"] = proportion_threshold
    return m


def evaluate_stay_level(decisions: List[Dict[str, Any]],
                         stay_gt: Dict[str, str],
                         raw_pr: Dict[str, str]) -> Dict[str, Any]:
    """
    Stay-level : stratégie unique (label séjour tel quel), mais on rapporte
    séparément les cas D pour analyse.
    """
    tp = fp = tn = fn = 0
    fp_doubt = fn_doubt = 0   # FP/FN qui correspondent à des cas D dans le GT

    for d in decisions:
        for s in d.get("stay_details", []):
            nda = str(s.get("stay_id", ""))
            if nda not in stay_gt:
                continue
            pred = s.get("label", "RA-")
            true = stay_gt[nda]
            is_doubt = raw_pr.get(nda) == "D"

            if pred == "RA+" and true == "RA+": tp += 1
            elif pred == "RA+" and true == "RA-":
                fp += 1
                if is_doubt: fp_doubt += 1
            elif pred == "RA-" and true == "RA-": tn += 1
            elif pred == "RA-" and true == "RA+":
                fn += 1
                if is_doubt: fn_doubt += 1

    m = compute_metrics(tp, fp, tn, fn)
    m["fp_from_doubt"] = fp_doubt
    m["fn_from_doubt"] = fn_doubt
    m["fp_excluding_doubt"] = fp - fp_doubt
    return m


# ════════════════════════════════════════════════════════════════
# AFFICHAGE
# ════════════════════════════════════════════════════════════════

def print_table(rows: List[Dict[str, Any]], title: str) -> None:
    print(f"\n{'='*78}")
    print(f"  {title}")
    print(f"{'='*78}")
    header = f"  {'Stratégie':<22} {'TP':>5} {'FP':>5} {'TN':>5} {'FN':>5} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'Spec':>7}"
    print(header)
    print("  " + "-" * 76)
    for r in rows:
        label = r["strategy"]
        if "threshold" in r:
            label = f"{label}({r['threshold']})"
        print(f"  {label:<22} "
              f"{r['tp']:>5} {r['fp']:>5} {r['tn']:>5} {r['fn']:>5} "
              f"{r['accuracy']:>7.1%} {r['precision']:>7.1%} {r['recall']:>7.1%} "
              f"{r['f1']:>7.1%} {r['specificity']:>7.1%}")


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Ré-évaluation post-hoc")
    ap.add_argument("--decisions", required=True,
                    help="Chemin du decisions.jsonl produit par le pipeline")
    ap.add_argument("--gt-csv", required=True,
                    help="GT CSV séjour-level (id,NIP,NDA,PR)")
    ap.add_argument("--output", default=None,
                    help="JSON de sortie (default: <decisions>/re_evaluation.json)")
    ap.add_argument("--gt-patient-agg", default="any_positive",
                    choices=["any_positive", "majority", "all_positive"],
                    help="Agrégation séjour→patient pour DÉRIVER le GT patient (défaut: any_positive)")
    args = ap.parse_args()

    decisions_path = Path(args.decisions)
    if not decisions_path.exists():
        print(f"ERROR: {decisions_path} introuvable")
        return

    # ── Chargement GT ─────────────────────────────────────────────
    stay_gt, raw_pr, nip_by_nda = load_gt(args.gt_csv)
    patient_gt = derive_patient_gt(stay_gt, nip_by_nda, args.gt_patient_agg)
    n_doubt = sum(1 for v in raw_pr.values() if v == "D")

    print(f"\n  GT chargé :")
    print(f"    Patients : {len(patient_gt)}")
    print(f"    Séjours  : {len(stay_gt)}")
    print(f"    Cas D    : {n_doubt} ({100*n_doubt/len(stay_gt):.1f}%)")
    print(f"    Distribution séjour : "
          f"RA+={sum(1 for v in stay_gt.values() if v=='RA+')}, "
          f"RA-={sum(1 for v in stay_gt.values() if v=='RA-')} "
          f"(dont {n_doubt} D)")
    print(f"    Distribution patient : "
          f"RA+={sum(1 for v in patient_gt.values() if v=='RA+')}, "
          f"RA-={sum(1 for v in patient_gt.values() if v=='RA-')} "
          f"(agg='{args.gt_patient_agg}')")

    # ── Chargement décisions (JSON liste OU JSONL ligne-par-ligne) ─
    with open(decisions_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    decisions: List[Dict[str, Any]] = []
    if not content:
        print(f"ERROR: {decisions_path} est vide")
        return

    # Cas 1 : JSON liste (commence par '[')
    if content.startswith("["):
        try:
            decisions = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"ERROR: JSON liste invalide : {e}")
            return
    # Cas 2 : JSON unique objet (commence par '{' avec un seul objet)
    elif content.startswith("{") and content.count("\n") < 3:
        try:
            obj = json.loads(content)
            decisions = [obj] if isinstance(obj, dict) else obj
        except json.JSONDecodeError as e:
            print(f"ERROR: JSON unique invalide : {e}")
            return
    # Cas 3 : JSONL (un objet par ligne)
    else:
        for i, line in enumerate(content.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                decisions.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [warn] ligne {i} ignorée : {e}")
                continue

    if not decisions:
        print(f"ERROR: aucune décision chargée depuis {decisions_path}")
        return

    print(f"\n  Décisions chargées : {len(decisions)} patients")

    # ── Stay-level (unique) ───────────────────────────────────────
    stay_metrics = evaluate_stay_level(decisions, stay_gt, raw_pr)
    stay_metrics["strategy"] = "stay-level (raw)"
    print_table([stay_metrics], "STAY-LEVEL")
    print(f"\n  Note : sur les {stay_metrics['fp']} FP, "
          f"{stay_metrics['fp_from_doubt']} correspondent à des cas D (doute) du GT.")
    print(f"         FP réels (excluant D) : {stay_metrics['fp_excluding_doubt']}")

    # ── Patient-level (multiple stratégies) ───────────────────────
    patient_results = []
    for strategy in ["any_positive", "majority", "confirmed"]:
        patient_results.append(evaluate_patient_level(decisions, patient_gt, strategy))
    # Stratégies par seuil de proportion
    for thr in [0.3, 0.4, 0.5, 0.6, 0.7]:
        patient_results.append(
            evaluate_patient_level(decisions, patient_gt, "proportion", thr))

    print_table(patient_results, "PATIENT-LEVEL (différentes stratégies)")

    # ── Meilleure stratégie par F1 ────────────────────────────────
    best = max(patient_results, key=lambda r: r["f1"])
    thr_label = f"(thr={best['threshold']})" if "threshold" in best else ""
    print(f"\n  Meilleure stratégie patient (F1) : {best['strategy']} {thr_label}"
          f" → F1={best['f1']:.1%}, Prec={best['precision']:.1%}, Rec={best['recall']:.1%}")

    # ── Sauvegarde ────────────────────────────────────────────────
    out_path = Path(args.output) if args.output else decisions_path.parent / "re_evaluation.json"
    output = {
        "decisions_file": str(decisions_path),
        "gt_csv": args.gt_csv,
        "gt_patient_aggregation": args.gt_patient_agg,
        "n_patients_gt": len(patient_gt),
        "n_stays_gt": len(stay_gt),
        "n_doubt_stays": n_doubt,
        "n_decisions_loaded": len(decisions),
        "stay_level": stay_metrics,
        "patient_level_strategies": patient_results,
        "best_patient_strategy": {
            "strategy": best["strategy"],
            "threshold": best.get("threshold"),
            "f1": best["f1"],
            "precision": best["precision"],
            "recall": best["recall"],
        },
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()