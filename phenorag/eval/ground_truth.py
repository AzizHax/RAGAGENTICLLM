"""
phenorag/eval/ground_truth.py

Chargement du ground truth séjour-level (CSV) et dérivation patient-level.

Format CSV attendu :
    id,NIP,NDA,PR
    1,123456,789012,F
    2,123456,789013,T
    3,234567,890123,D
    ...

Légende PR :
    T = True  → RA+ (PR confirmée)
    F = False → RA- (pas de PR)
    D = Doute → traité comme RA-  (selon spec utilisateur)
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set


# Mapping PR → label binaire
PR_TO_LABEL = {
    "T": "RA+",
    "F": "RA-",
    "D": "RA-",   # doute traité comme négatif (cf. spec)
}


@dataclass
class GroundTruth:
    """Container pour les labels GT à 2 niveaux."""
    stay_labels: Dict[str, str]       # {NDA: "RA+"|"RA-"}
    patient_labels: Dict[str, str]    # {NIP: "RA+"|"RA-"}
    raw_pr: Dict[str, str]            # {NDA: "T"|"F"|"D"} pour audit
    nip_by_nda: Dict[str, str]        # {NDA: NIP}

    @property
    def n_patients(self) -> int:
        return len(self.patient_labels)

    @property
    def n_stays(self) -> int:
        return len(self.stay_labels)

    def patient_distribution(self) -> Dict[str, int]:
        from collections import Counter
        return dict(Counter(self.patient_labels.values()))

    def stay_distribution(self) -> Dict[str, int]:
        from collections import Counter
        return dict(Counter(self.stay_labels.values()))

    def doubt_count(self) -> int:
        return sum(1 for v in self.raw_pr.values() if v == "D")


def load_ground_truth(csv_path: str | Path,
                       patient_aggregation: str = "any_positive") -> GroundTruth:
    """
    Charge le GT séjour et dérive le GT patient par agrégation.

    Args:
        csv_path: chemin vers le CSV GT
        patient_aggregation: stratégie d'agrégation patient
            - "any_positive": un patient est RA+ si AU MOINS un séjour est RA+
            - "majority": RA+ si >50% des séjours sont RA+
            - "all_positive": RA+ seulement si tous les séjours sont RA+

    Returns:
        GroundTruth avec stay_labels, patient_labels, raw_pr, nip_by_nda
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"GT CSV not found: {csv_path}")

    stay_labels: Dict[str, str] = {}
    raw_pr: Dict[str, str] = {}
    nip_by_nda: Dict[str, str] = {}
    nip_to_stays: Dict[str, List[str]] = {}

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        # Tolérer des espaces dans les valeurs ("XXXX, " etc.)
        for row in reader:
            nip = (row.get("NIP") or "").strip()
            nda = (row.get("NDA") or "").strip()
            pr_raw = (row.get("PR") or "").strip().upper().strip('"')

            if not nip or not nda:
                continue
            if pr_raw not in PR_TO_LABEL:
                # Ligne malformée (ex: PR vide, valeur inattendue) — on skip
                continue

            label = PR_TO_LABEL[pr_raw]
            stay_labels[nda] = label
            raw_pr[nda] = pr_raw
            nip_by_nda[nda] = nip
            nip_to_stays.setdefault(nip, []).append(nda)

    # Agrégation patient
    patient_labels: Dict[str, str] = {}
    for nip, ndas in nip_to_stays.items():
        labels = [stay_labels[nda] for nda in ndas]
        n_pos = labels.count("RA+")
        n_total = len(labels)

        if patient_aggregation == "any_positive":
            patient_labels[nip] = "RA+" if n_pos > 0 else "RA-"
        elif patient_aggregation == "majority":
            patient_labels[nip] = "RA+" if n_pos > n_total / 2 else "RA-"
        elif patient_aggregation == "all_positive":
            patient_labels[nip] = "RA+" if n_pos == n_total else "RA-"
        else:
            raise ValueError(f"Unknown aggregation: {patient_aggregation}")

    return GroundTruth(
        stay_labels=stay_labels,
        patient_labels=patient_labels,
        raw_pr=raw_pr,
        nip_by_nda=nip_by_nda,
    )


def filter_first_n_patients(gt: GroundTruth, n: int) -> Set[str]:
    """
    Retourne le set des NIP correspondant aux N premiers patients
    rencontrés dans l'ordre d'insertion (= ordre du fichier CSV).
    """
    return set(list(gt.patient_labels.keys())[:n])
