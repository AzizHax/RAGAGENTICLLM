"""
phenorag/preprocess.py

Parquet EHR → JSON corpus compatible with Agent 1.

Transforms questionnaire format (LIBELLE/REPONSE pairs) into
the JSON structure expected by IOACorpusLoader:

[
  {
    "patient_id": "P001",
    "stays": [
      {
        "stay_id": "S001",
        "date": "2024-01-15",
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

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

SECTION_KEYWORDS = {
    "identite": ["identité", "patient", "age", "sexe"],
    "motif": ["motif", "recours", "plainte", "admission"],
    "antecedents": ["antécédent", "atcd", "historique"],
    "traitement": ["traitement", "médicament", "ordonnance", "habituel"],
    "constantes": ["constante", "température", "tension", "fréquence", "saturation"],
    "examen_clinique": ["examen", "clinique", "cardiovasculaire", "neurologique"],
    "biologie": ["biologie", "bilan", "crp", "leucocyte", "hémoglobine"],
    "imagerie": ["imagerie", "radio", "scanner", "irm", "échographie"],
    "conclusion": ["conclusion", "diagnostic", "synthèse", "évolution"],
    "sortie": ["sortie", "orientation"],
}


def classify_section(label: str) -> str:
    label_lower = label.lower()
    for section, keywords in SECTION_KEYWORDS.items():
        for kw in keywords:
            if kw in label_lower:
                return section
    return "autre"


def parquet_to_corpus(parquet_path: str,
                      patient_col: str = "NIPATIENT",
                      stay_col: str = "NISEJOUR",
                      label_col: str = "LIBELLE",
                      response_col: str = "REPONSE",
                      nda_col: str = "NDA",
                      output_path: Optional[str] = None) -> List[Dict]:
    """
    Convert a Parquet EHR file to JSON corpus format for Agent 1.

    Returns list of patient dicts AND saves to output_path if provided.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pip install pandas pyarrow")

    print(f"Loading {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    print(f"  {len(df)} rows, columns: {list(df.columns)}")

    # Validate columns
    for col_name, col_val in [("patient", patient_col), ("stay", stay_col),
                               ("label", label_col), ("response", response_col)]:
        if col_val not in df.columns:
            available = ", ".join(df.columns)
            raise ValueError(f"Column '{col_val}' not found. Available: {available}")

    # Group by patient → stays → records
    patients_dict: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))

    for _, row in df.iterrows():
        pid = str(row[patient_col])
        sid = str(row[stay_col])
        label = str(row.get(label_col, "")) if not pd.isna(row.get(label_col)) else ""
        response = str(row.get(response_col, "")) if not pd.isna(row.get(response_col)) else ""

        if not label and not response:
            continue

        record = {"LIBELLE": label.strip(), "REPONSE": response.strip()}

        # Add NDA if available
        if nda_col in df.columns and not pd.isna(row.get(nda_col)):
            record["NDA"] = str(row[nda_col])

        patients_dict[pid][sid].append(record)

    # Build corpus structure
    corpus = []
    for pid in sorted(patients_dict.keys()):
        stays = []
        for i, (sid, records) in enumerate(sorted(patients_dict[pid].items()), 1):
            stays.append({
                "stay_id": f"{pid}_{sid}",
                "date": "",
                "visit_number": i,
                "records": records,
            })
        corpus.append({
            "patient_id": pid,
            "stays": stays,
        })

    n_patients = len(corpus)
    n_stays = sum(len(p["stays"]) for p in corpus)
    n_records = sum(len(s["records"]) for p in corpus for s in p["stays"])
    print(f"  → {n_patients} patients, {n_stays} stays, {n_records} records")

    # Save
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(corpus, f, ensure_ascii=False, indent=2)
        print(f"  → Saved to {output_path}")

    return corpus


def preprocess_parquet(parquet_path: str, output_dir: str,
                        patient_col: str = "NIPATIENT",
                        stay_col: str = "NISEJOUR",
                        label_col: str = "LIBELLE",
                        response_col: str = "REPONSE",
                        nda_col: str = "NDA") -> str:
    """
    Full preprocessing pipeline: Parquet → JSON + report.
    Returns path to the output JSON corpus.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    corpus_path = str(out / "corpus.json")

    corpus = parquet_to_corpus(
        parquet_path, patient_col, stay_col, label_col, response_col, nda_col,
        output_path=corpus_path)

    # Generate report
    report = {
        "input": parquet_path,
        "output": corpus_path,
        "n_patients": len(corpus),
        "n_stays": sum(len(p["stays"]) for p in corpus),
        "n_records": sum(len(s["records"]) for p in corpus for s in p["stays"]),
        "patients": [
            {"patient_id": p["patient_id"], "n_stays": len(p["stays"]),
             "n_records": sum(len(s["records"]) for s in p["stays"])}
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
    parser.add_argument("input", help="Input Parquet file")
    parser.add_argument("--output-dir", default="data/preprocessed")
    parser.add_argument("--patient-col", default="NIPATIENT")
    parser.add_argument("--stay-col", default="NISEJOUR")
    parser.add_argument("--label-col", default="LIBELLE")
    parser.add_argument("--response-col", default="REPONSE")
    args = parser.parse_args()

    preprocess_parquet(args.input, args.output_dir,
                        args.patient_col, args.stay_col,
                        args.label_col, args.response_col)
