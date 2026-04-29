"""
phenorag/preprocess.py

Parquet EHR → JSON corpus compatible with Agent 1.

Le patient_id (NIP) est résolu UNIQUEMENT via la jointure GT (NDA → NIP).
Le parquet n'a pas besoin de contenir NIPATIENT — seul NDA est utilisé comme clé.

Output:
[
  {
    "patient_id": "10001",       ← NIP du GT
    "stays": [
      {
        "stay_id": "1000100",    ← NDA pur (sans préfixe)
        "date": "",
        "visit_number": 1,
        "records": [
          {"LIBELLE": "Motif", "REPONSE": "Douleurs articulaires"},
          ...
        ]
      }
    ]
  }
]
"""
from __future__ import annotations

import csv as _csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional


def _clean_nda(val: str) -> str:
    """Retire le préfixe NDA et normalise en entier string. 'NDA1000100' → '1000100'"""
    val = str(val).strip().upper().replace("NDA", "")
    try:
        return str(int(float(val)))
    except (ValueError, TypeError):
        return val


def _load_gt_mapping(gt_path: str) -> Dict[str, str]:
    """Charge la table NDA → NIP depuis le GT CSV."""
    nda_to_nip: Dict[str, str] = {}
    with open(gt_path, "r", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        cols = [c.strip() for c in (reader.fieldnames or [])]
        col_map = {c.lower(): c for c in cols}
        nip_col = col_map.get("nip") or col_map.get("patient_id")
        nda_col = col_map.get("nda") or col_map.get("stay_id")
        if not nip_col or not nda_col:
            raise ValueError(f"GT CSV : colonnes NIP/NDA non trouvées. Colonnes : {cols}")
        for row in reader:
            nda = _clean_nda(str(row[nda_col]))
            nip = str(row[nip_col]).strip()
            if nda and nip:
                nda_to_nip[nda] = nip
    return nda_to_nip


def parquet_to_corpus(parquet_path: str,
                      stay_col: str = "NISEJOUR",
                      label_col: str = "LIBELLE",
                      response_col: str = "REPONSE",
                      nda_col: str = "NDA",
                      gt_path: Optional[str] = None,
                      output_path: Optional[str] = None) -> List[Dict]:
    """
    Convert a Parquet EHR file to JSON corpus format for Agent 1.

    NIP résolu via jointure GT (NDA → NIP). NIPATIENT ignoré.
    Séjours absents du GT ignorés (hors cohorte).
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pip install pandas pyarrow")

    print(f"Loading {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    print(f"  {len(df)} rows, columns: {list(df.columns)}")

    for col_name, col_val in [("stay", stay_col), ("label", label_col), ("response", response_col)]:
        if col_val not in df.columns:
            raise ValueError(f"Colonne '{col_val}' absente. Disponibles : {', '.join(df.columns)}")

    # ── Jointure GT : NDA → NIP ───────────────────────────────────
    nda_to_nip: Dict[str, str] = {}
    if gt_path and Path(gt_path).exists():
        nda_to_nip = _load_gt_mapping(gt_path)
        print(f"  [GT] {len(nda_to_nip)} NDA→NIP chargés depuis {gt_path}")
    else:
        print("  [GT] Aucun GT fourni — tous les séjours inclus avec NIP=UNKNOWN")

    # ── Groupement parquet par NDA ────────────────────────────────
    stays_dict: Dict[str, List[Dict]] = defaultdict(list)  # nda → [records]

    for _, row in df.iterrows():
        raw_nda = row.get(nda_col) if nda_col in df.columns else row.get(stay_col)
        if raw_nda is None or pd.isna(raw_nda):
            continue
        nda = _clean_nda(str(raw_nda))

        label    = "" if pd.isna(row.get(label_col))    else str(row[label_col]).strip()
        response = "" if pd.isna(row.get(response_col)) else str(row[response_col]).strip()
        if not label and not response:
            continue

        stays_dict[nda].append({"LIBELLE": label, "REPONSE": response})

    # ── Résolution NIP via GT ─────────────────────────────────────
    patients_tmp: Dict[str, List[str]] = defaultdict(list)  # nip → [nda]
    skipped = 0
    for nda in stays_dict:
        nip = nda_to_nip.get(nda)
        if nip is None:
            if nda_to_nip:  # GT fourni mais NDA absent → hors cohorte
                skipped += 1
                continue
            nip = "UNKNOWN"
        patients_tmp[nip].append(nda)

    if skipped:
        print(f"  [GT] {skipped} séjours ignorés (NDA absent du GT — hors cohorte)")

    # ── Construction corpus final ─────────────────────────────────
    corpus = []
    for nip in sorted(patients_tmp.keys()):
        ndas_sorted = sorted(patients_tmp[nip])
        stays = [
            {
                "stay_id":      nda,
                "date":         "",
                "visit_number": i,
                "records":      stays_dict[nda],
            }
            for i, nda in enumerate(ndas_sorted, 1)
        ]
        corpus.append({"patient_id": nip, "stays": stays})

    n_stays   = sum(len(p["stays"]) for p in corpus)
    n_records = sum(len(s["records"]) for p in corpus for s in p["stays"])
    print(f"  → {len(corpus)} patients, {n_stays} séjours, {n_records} records")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(corpus, f, ensure_ascii=False, indent=2)
        print(f"  → Saved to {output_path}")

    return corpus


def preprocess_parquet(parquet_path: str, output_dir: str,
                       stay_col: str = "NISEJOUR",
                       label_col: str = "LIBELLE",
                       response_col: str = "REPONSE",
                       nda_col: str = "NDA",
                       gt_path: Optional[str] = None) -> str:
    """
    Full preprocessing pipeline: Parquet → JSON corpus + report.
    gt_path : CSV ground truth (NIP, NDA) — source de vérité pour les IDs.
    Returns path to the output JSON corpus.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    corpus_path = str(out / "corpus.json")

    corpus = parquet_to_corpus(
        parquet_path,
        stay_col=stay_col, label_col=label_col,
        response_col=response_col, nda_col=nda_col,
        gt_path=gt_path, output_path=corpus_path,
    )

    report = {
        "input":      parquet_path,
        "output":     corpus_path,
        "n_patients": len(corpus),
        "n_stays":    sum(len(p["stays"]) for p in corpus),
        "n_records":  sum(len(s["records"]) for p in corpus for s in p["stays"]),
        "patients": [
            {"patient_id": p["patient_id"],
             "n_stays":    len(p["stays"]),
             "n_records":  sum(len(s["records"]) for s in p["stays"])}
            for p in corpus
        ],
    }
    report_path = str(out / "preprocessing_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  → Report: {report_path}")

    return corpus_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess Parquet EHR → PhenoRAG JSON")
    parser.add_argument("input",          help="Input Parquet file")
    parser.add_argument("--output-dir",   default="data/preprocessed")
    parser.add_argument("--stay-col",     default="NISEJOUR")
    parser.add_argument("--label-col",    default="LIBELLE")
    parser.add_argument("--response-col", default="REPONSE")
    parser.add_argument("--nda-col",      default="NDA")
    parser.add_argument("--gt",           default=None, help="CSV GT (NIP, NDA)")
    args = parser.parse_args()

    preprocess_parquet(args.input, args.output_dir,
                       stay_col=args.stay_col, label_col=args.label_col,
                       response_col=args.response_col, nda_col=args.nda_col,
                       gt_path=args.gt)