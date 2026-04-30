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
    """Charge la table NDA → NIP depuis le GT CSV.
    Détecte automatiquement le délimiteur (, ou ;) et le BOM UTF-8.
    """
    nda_to_nip: Dict[str, str] = {}
    with open(gt_path, "r", encoding="utf-8-sig") as f:
        sample = f.read(4096)
        dialect = _csv.Sniffer().sniff(sample, delimiters=",;\t|")
        f.seek(0)
        reader = _csv.DictReader(f, dialect=dialect)
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


def _safe_str(val, default: str = "") -> str:
    """Convertit n'importe quelle valeur en str propre, gère NaN/None/bytes."""
    if val is None:
        return default
    try:
        import pandas as pd
        if pd.isna(val):
            return default
    except (TypeError, ValueError, ImportError):
        pass
    if isinstance(val, bytes):
        try:
            return val.decode("utf-8", errors="replace").strip()
        except Exception:
            return default
    return str(val).strip()


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
    Robuste aux types réels : floats, NaN, bytes, colonnes manquantes,
    NDA dupliqués, valeurs nulles, encodages mixtes.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pip install pandas pyarrow")

    print(f"Loading {parquet_path}...")
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        raise RuntimeError(f"Impossible de lire le parquet : {e}")
    print(f"  {len(df)} rows, columns: {list(df.columns)}")

    # ── Détection flexible des colonnes ──────────────────────────
    col_lower = {c.lower().strip(): c for c in df.columns}

    def _find_col(candidates: list, required: bool = True) -> Optional[str]:
        for c in candidates:
            if c.lower() in col_lower:
                return col_lower[c.lower()]
        if required:
            raise ValueError(f"Aucune colonne parmi {candidates} trouvée. Disponibles : {list(df.columns)}")
        return None

    stay_col  = _find_col([stay_col,  "nisejour", "sejour_id", "visit_id", "encounter_id"])
    label_col = _find_col([label_col, "libelle",  "label",     "item",     "question"])
    resp_col  = _find_col([response_col, "reponse", "response", "valeur", "value", "resultat"])
    nda_col   = _find_col([nda_col,   "nda",       "ndossier",  "dossier"], required=False)

    print(f"  Colonnes → stay={stay_col} label={label_col} response={resp_col} nda={nda_col or 'absent→fallback NISEJOUR'}")

    # ── Jointure GT : NDA → NIP ───────────────────────────────────
    nda_to_nip: Dict[str, str] = {}
    if gt_path and Path(gt_path).exists():
        try:
            nda_to_nip = _load_gt_mapping(gt_path)
            print(f"  [GT] {len(nda_to_nip)} NDA→NIP chargés depuis {gt_path}")
        except Exception as e:
            print(f"  [GT] Erreur chargement GT ({e}) — NIP=UNKNOWN pour tous")
    else:
        print("  [GT] Aucun GT fourni — tous les séjours inclus avec NIP=UNKNOWN")

    # ── Groupement parquet par NDA ────────────────────────────────
    stays_dict: Dict[str, List[Dict]] = defaultdict(list)
    n_skipped_rows = 0

    for _, row in df.iterrows():
        try:
            # NDA : colonne NDA si dispo, sinon NISEJOUR comme fallback
            raw_nda = row[nda_col] if nda_col else row[stay_col]
            nda = _clean_nda(_safe_str(raw_nda))
            if not nda:
                n_skipped_rows += 1
                continue

            label    = _safe_str(row[label_col])
            response = _safe_str(row[resp_col])
            if not label and not response:
                n_skipped_rows += 1
                continue

            stays_dict[nda].append({"LIBELLE": label, "REPONSE": response})

        except Exception as e:
            n_skipped_rows += 1
            continue

    if n_skipped_rows:
        print(f"  [Parse] {n_skipped_rows} lignes ignorées (NDA vide, valeurs nulles, erreurs)")

    # ── Résolution NIP via GT ─────────────────────────────────────
    patients_tmp: Dict[str, List[str]] = defaultdict(list)
    skipped_stays = 0
    for nda in stays_dict:
        nip = nda_to_nip.get(nda)
        if nip is None:
            if nda_to_nip:
                skipped_stays += 1
                continue
            nip = "UNKNOWN"
        patients_tmp[nip].append(nda)

    if skipped_stays:
        print(f"  [GT] {skipped_stays} séjours ignorés (NDA absent du GT — hors cohorte)")

    if not patients_tmp:
        raise RuntimeError(
            "Corpus vide après jointure GT. "
            "Vérifiez que les NDA du parquet correspondent à ceux du GT CSV."
        )

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