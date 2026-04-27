"""
phenorag/eval/evaluate_v2.py

Évaluation à 2 niveaux (stay-level + patient-level) avec :
  - Métriques classiques : TP, FP, TN, FN, Acc, Prec, Recall, F1, Spec
  - Décomposition des erreurs sur les cas "D" (doute) — ils sont comptés
    comme RA- mais on rapporte séparément ces cas pour analyse fine.
  - Sortie JSON unique avec les 2 niveaux + résumé.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .ground_truth import GroundTruth, load_ground_truth


def _confusion(pred: str, true: str) -> str:
    if pred == "RA+" and true == "RA+": return "TP"
    if pred == "RA+" and true == "RA-": return "FP"
    if pred == "RA-" and true == "RA-": return "TN"
    if pred == "RA-" and true == "RA+": return "FN"
    return "UNK"


def _metrics_from_counts(tp: int, fp: int, tn: int, fn: int) -> Dict[str, float]:
    total = tp + fp + tn + fn
    acc = (tp + tn) / total if total else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "n": total,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "specificity": round(spec, 4),
    }


def evaluate_two_levels(decisions_path: Path,
                         gt: GroundTruth,
                         patient_filter: set | None = None) -> Dict[str, Any]:
    """
    Évalue les décisions du pipeline aux 2 niveaux.

    Args:
        decisions_path: runs/.../decisions.jsonl (sortie Agent 3)
        gt: GroundTruth chargé via load_ground_truth()
        patient_filter: si fourni, n'évalue que ces NIP

    Returns:
        Dict avec les sections : stay_level, patient_level, doubt_analysis, errors
    """
    if not decisions_path.exists():
        return {"error": f"decisions.jsonl not found: {decisions_path}"}

    with open(decisions_path, "r", encoding="utf-8") as f:
        decisions = [json.loads(line) for line in f]

    # ── Patient-level ─────────────────────────────────────────────
    p_tp = p_fp = p_tn = p_fn = 0
    patient_errors: List[Dict[str, Any]] = []
    patients_evaluated: List[str] = []

    for d in decisions:
        nip = str(d["patient_id"])
        if patient_filter and nip not in patient_filter:
            continue
        true = gt.patient_labels.get(nip)
        if true is None:
            continue
        pred = d["final_label"]
        patients_evaluated.append(nip)
        kind = _confusion(pred, true)
        if kind == "TP": p_tp += 1
        elif kind == "FP": p_fp += 1
        elif kind == "TN": p_tn += 1
        elif kind == "FN": p_fn += 1
        if kind in ("FP", "FN"):
            patient_errors.append({
                "patient_id": nip, "pred": pred, "true": true, "type": kind,
                "confidence": d.get("final_confidence"),
            })

    # ── Stay-level ────────────────────────────────────────────────
    s_tp = s_fp = s_tn = s_fn = 0
    s_doubt_pred_pos = 0    # cas D prédits RA+ (intéressants à investiguer)
    s_doubt_pred_neg = 0    # cas D prédits RA-
    stay_errors: List[Dict[str, Any]] = []

    for d in decisions:
        nip = str(d["patient_id"])
        if patient_filter and nip not in patient_filter:
            continue
        for stay in d.get("stay_details", []):
            nda = str(stay.get("stay_id", ""))
            true = gt.stay_labels.get(nda)
            if true is None:
                continue
            pred = stay.get("label", "RA-")
            kind = _confusion(pred, true)
            if kind == "TP": s_tp += 1
            elif kind == "FP": s_fp += 1
            elif kind == "TN": s_tn += 1
            elif kind == "FN": s_fn += 1

            # Cas de doute (PR=D dans le GT) : rapporté séparément
            if gt.raw_pr.get(nda) == "D":
                if pred == "RA+":
                    s_doubt_pred_pos += 1
                else:
                    s_doubt_pred_neg += 1

            if kind in ("FP", "FN"):
                stay_errors.append({
                    "patient_id": nip, "stay_id": nda,
                    "pred": pred, "true": true, "type": kind,
                    "raw_gt": gt.raw_pr.get(nda),
                    "confidence": stay.get("confidence"),
                    "acr_score": stay.get("acr_score"),
                })

    return {
        "n_patients_evaluated": len(set(patients_evaluated)),
        "patient_level": _metrics_from_counts(p_tp, p_fp, p_tn, p_fn),
        "stay_level": _metrics_from_counts(s_tp, s_fp, s_tn, s_fn),
        "doubt_analysis": {
            "n_doubt_stays_in_gt": gt.doubt_count(),
            "predicted_positive": s_doubt_pred_pos,
            "predicted_negative": s_doubt_pred_neg,
            "note": "Cas D (doute) traités comme RA- pour le calcul mais rapportés ici",
        },
        "patient_errors": patient_errors,
        "stay_errors": stay_errors[:50],   # cap pour pas exploser le JSON
        "n_stay_errors_total": len(stay_errors),
    }


def print_metrics(metrics: Dict[str, Any]) -> None:
    """Affichage console formaté."""
    if "error" in metrics:
        print(f"  ERROR: {metrics['error']}")
        return

    print(f"\n  Patients évalués : {metrics['n_patients_evaluated']}")

    pl = metrics["patient_level"]
    print(f"\n  ── PATIENT-LEVEL ─────────────────────────")
    print(f"  TP={pl['tp']}  FP={pl['fp']}  TN={pl['tn']}  FN={pl['fn']}  (n={pl['n']})")
    print(f"  Accuracy:    {pl['accuracy']:.1%}")
    print(f"  Precision:   {pl['precision']:.1%}")
    print(f"  Recall:      {pl['recall']:.1%}")
    print(f"  F1-Score:    {pl['f1']:.1%}")
    print(f"  Specificity: {pl['specificity']:.1%}")

    sl = metrics["stay_level"]
    print(f"\n  ── STAY-LEVEL ────────────────────────────")
    print(f"  TP={sl['tp']}  FP={sl['fp']}  TN={sl['tn']}  FN={sl['fn']}  (n={sl['n']})")
    print(f"  Accuracy:    {sl['accuracy']:.1%}")
    print(f"  Precision:   {sl['precision']:.1%}")
    print(f"  Recall:      {sl['recall']:.1%}")
    print(f"  F1-Score:    {sl['f1']:.1%}")
    print(f"  Specificity: {sl['specificity']:.1%}")

    da = metrics["doubt_analysis"]
    if da["n_doubt_stays_in_gt"] > 0:
        print(f"\n  ── DOUBT (D) ─────────────────────────────")
        print(f"  Cas D dans GT:  {da['n_doubt_stays_in_gt']}")
        print(f"  Prédits RA+:    {da['predicted_positive']}")
        print(f"  Prédits RA-:    {da['predicted_negative']}")

    n_p_err = len(metrics["patient_errors"])
    if n_p_err:
        print(f"\n  Erreurs patient ({n_p_err}):")
        for e in metrics["patient_errors"][:15]:
            print(f"    NIP={e['patient_id']}: pred={e['pred']} true={e['true']} ({e['type']})")


def evaluate_and_save(decisions_path: str | Path,
                       gt_csv_path: str | Path,
                       output_path: str | Path,
                       patient_filter: set | None = None,
                       patient_aggregation: str = "any_positive") -> Dict[str, Any]:
    """Charge le GT, évalue, affiche, et sauve dans un JSON."""
    gt = load_ground_truth(gt_csv_path, patient_aggregation=patient_aggregation)
    print(f"\n  GT chargé: {gt.n_patients} patients, {gt.n_stays} séjours "
          f"({gt.doubt_count()} cas D)")

    metrics = evaluate_two_levels(Path(decisions_path), gt, patient_filter)
    print_metrics(metrics)

    out = Path(output_path)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\n  → {out}")
    return metrics
