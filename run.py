#!/usr/bin/env python3
"""
run.py — PhenoRAG Pipeline Runner (Version Finale Optimisée)

Usage:
    # JSON corpus
    python run.py --arch B2 --corpus data/test_corpus/ehr_ioa_test_50p.json
    # Parquet EHR (auto-converts to JSON)
    python run.py --arch B2 --corpus data/real/ehr.parquet
    # Skip extraction
    python run.py --arch B2 --skip-extraction
    # Without KB
    python run.py --arch B1 --no-kb
    # Without inter-stay RAG
    python run.py --arch B1 --no-inter-stay-rag
    # KB only, no inter-stay RAG (Axe D2)
    python run.py --arch B1 --no-inter-stay-rag
    # Custom prompts (Axe C — few-shot / CoT)
    python run.py --arch B1 --prompt-acr-scoring agent2/acr_scoring_fewshot \
                             --prompt-ra-relatedness agent2/ra_relatedness_fewshot
    # Custom aggregation strategy (Axe E)
    python run.py --arch B1 --gt-patient-agg majority
    python run.py --arch B1 --gt-patient-agg confirmed
    # Limit patients processed
    python run.py --arch B1 --n-patients 10
    # Stay-level ground truth CSV
    python run.py --arch B1 --gt-stay-csv data/gt_stay.csv --gt-patient-agg any_positive
    # Custom checkpoint interval (minutes)
    python run.py --arch B2 --checkpoint-interval 10
    # Preprocess only
    python run.py --only preprocess --corpus data/real/ehr.parquet
    # EDA only
    python run.py --only eda --corpus data/real/ehr.parquet
    # Evaluate only
    python run.py --only eval --run-dir runs/b2
"""

import argparse
import json
import re
import time
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Any, Optional

# Importations conditionnelles pour l'EDA
try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")
    HAS_EDA_LIBS = True
except ImportError:
    HAS_EDA_LIBS = False

from phenorag.agents.orchestrator import Orchestrator
from phenorag.agents.configs import (
    Agent1Config, Agent2Config, Agent3Config, PipelineConfig,
)
from phenorag.utils.llm_client import LLMClient
from phenorag.utils.prompt_loader import PromptLoader

# Configuration par défaut
OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:3b-instruct"
B4_MODELS = ["qwen7b:latest", "mistral_7b:latest", "llama3_1_8b_gguf:latest"]
CORPUS_PATH = "data//ehr_ioa_test_50p.json"
KB_PATH = "data/kb_pr_phenotype.json"
GROUND_TRUTH_PATH = "data/ground_truth_50p.json"
PROMPTS_DIR = "prompts"

# ════════════════════════════════════════════════════════════════
# PATTERNS POLYARTHRITE RHUMATOÏDE (PR)
# ════════════════════════════════════════════════════════════════

RA_PATTERNS = {
    'disease_names': [
        r'\bpolyarthrite\s+rhumatoïde\b', r'\barthrite\s+rhumatoïde\b',
        r'\bPR\b(?!\s*(artérielle|interval))', r'\bRA\b(?!\s*artérielle)',
    ],
    'serology': [
        r'\bRF\b', r'\bfacteur\s+rhumatoïde\b', r'\banti[- ]?CCP\b', r'\bACPA\b',
    ],
    'inflammatory_markers': [
        r'\bCRP\b', r'\bprotéine\s+C\s+réactive\b', r'\bVS\b', r'\bESR\b',
    ],
    'dmards': [
        r'\bm[eé]thotrexate\b', r'\bMTX\b', r'\bsulfasalazine\b',
        r'\bleflunomide\b', r'\bhydroxychloroquine\b', r'\bplaquenil\b',
    ],
    'biologics': [
        r'\badalimumab\b', r'\bhumira\b', r'\betanercept\b', r'\benbrel\b',
        r'\binfliximab\b', r'\btocilizumab\b', r'\brituximab\b', r'\babatacept\b',
    ],
    'jak_inhibitors': [
        r'\btofacitinib\b', r'\bbaricitinib\b', r'\bupadacitinib\b',
    ],
    'joints': [
        r'\bMCP\b', r'\bPIP\b', r'\bMTP\b', r'\bpoignet\b',
        r'\bsynovite\b', r'\bsynovitis\b', r'\braideur\b',
    ],
}

# ════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════

def find_col_name(df_columns, candidates):
    """Détecte dynamiquement le nom d'une colonne parmi des synonymes."""
    df_cols_lower = {c.lower(): c for c in df_columns}
    for candidate in candidates:
        if candidate.lower() in df_cols_lower:
            return df_cols_lower[candidate.lower()]
    return None

# ════════════════════════════════════════════════════════════════
# EDA (ANALYSE EXPLORATOIRE)
# ════════════════════════════════════════════════════════════════

def run_eda(corpus_path: str, output_dir: str, gt_path: Optional[str] = None):
    """EDA Complète : Détection auto, Métriques console et Visualisations."""
    if not HAS_EDA_LIBS:
        print("Erreur : pandas et matplotlib sont requis.")
        return

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*75}")
    print("EDA — ANALYSE EXPLORATOIRE DES DONNÉES")
    print(f"{'='*75}")

    ext = Path(corpus_path).suffix.lower()
    df_flat = None
    patient_stats_list = []

    col_candidates = {
        'patient': ['NIPATIENT', 'patient_id', 'id_patient', 'PATIENT', 'SUBJECT_ID'],
        'stay': ['NISEJOUR', 'stay_id', 'id_sejour', 'SEJOUR', 'VISIT_ID'],
        'label': ['LIBELLE', 'label', 'type', 'category', 'item'],
        'response': ['REPONSE', 'response', 'valeur', 'value', 'texte', 'RESULTAT']
    }

    # ── 1. Chargement et Normalisation ──
    if ext in (".parquet", ".pq"):
        df_flat = pd.read_parquet(corpus_path)
        print(f"  Format détecté : Parquet")
        p_col = find_col_name(df_flat.columns, col_candidates['patient'])
        s_col = find_col_name(df_flat.columns, col_candidates['stay'])
        l_col = find_col_name(df_flat.columns, col_candidates['label'])
        r_col = find_col_name(df_flat.columns, col_candidates['response'])

        if gt_path and Path(gt_path).exists():
            with open(gt_path, "r", encoding="utf-8") as f:
                gt_data = json.load(f)
            stays_counts = df_flat.groupby(p_col)[s_col].nunique().to_dict() if (p_col and s_col) else {}
            for pid, info in gt_data.items():
                if isinstance(info, dict):
                    patient_stats_list.append({
                        "patient_id": pid, "n_stays": info.get("n_stays", stays_counts.get(pid, 0)),
                        "phenotype": info.get("phenotype", "N/A"), "profile": info.get("profile", "N/A")
                    })
    else:
        with open(corpus_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            corpus = json.loads(content) if content.startswith('[') else [json.loads(l) for l in content.splitlines() if l.strip()]

        rows = []
        for p in corpus:
            p_id = p.get("patient_id", "Unknown")
            gt = p.get("ground_truth", {})
            patient_stats_list.append({
                "patient_id": p_id, "n_stays": len(p.get("stays", [])),
                "phenotype": gt.get("phenotype", "N/A"), "profile": gt.get("profile", "N/A")
            })
            for s in p.get("stays", []):
                for r in s.get("records", []):
                    rows.append({
                        "NIPATIENT": p_id, "NISEJOUR": s.get("stay_id", "Unknown"),
                        "LIBELLE": r.get("LIBELLE", ""), "REPONSE": str(r.get("REPONSE", ""))
                    })
        df_flat = pd.DataFrame(rows)
        p_col, s_col, l_col, r_col = "NIPATIENT", "NISEJOUR", "LIBELLE", "REPONSE"
        print(f"  Format détecté : JSON")

    # ── 2. Affichage des Métriques Console ──
    print(f"  Lignes : {len(df_flat):,}")
    print(f"  Colonnes identifiées : Patient={p_col}, Séjour={s_col}, Label={l_col}, Réponse={r_col}")

    print(f"\n{'─'*75}")
    print(f"  {'Colonne':<25} {'Type':<12} {'Unique':<10} {'Missing%'}")
    print(f"  {'─'*70}")
    for col in df_flat.columns:
        nu = df_flat[col].nunique()
        pm = df_flat[col].isna().sum() / len(df_flat) * 100
        print(f"  {col:<25} {str(df_flat[col].dtype):<12} {nu:<10} {pm:<8.1f}")

    if p_col: print(f"\n  Nombre de Patients : {df_flat[p_col].nunique()}")
    if s_col: print(f"  Nombre de Séjours : {df_flat[s_col].nunique()}")

    comp = (1 - df_flat.isnull().sum().sum() / (len(df_flat) * len(df_flat.columns))) * 100
    print(f"  Complétude globale : {comp:.1f}%")

    text_sample = " ".join(df_flat[r_col].fillna("").astype(str).head(500))
    print(f"\n  Terminologie PR (échantillon 500 records) :")
    term_counts = {}
    for cat, patterns in RA_PATTERNS.items():
        n = sum(len(re.findall(p, text_sample, re.I)) for p in patterns)
        term_counts[cat] = n
        if n > 0: print(f"    {cat:<25} {n:>5}")

    # ── 3. Génération des Visualisations ──
    fig, ax = plt.subplots(figsize=(10, 5))
    df_flat.isnull().sum().sort_values().plot(kind='barh', ax=ax, color='salmon')
    ax.set_title("Valeurs manquantes")
    fig.savefig(out / "eda_missing.png")

    if l_col:
        fig, ax = plt.subplots(figsize=(10, 6))
        df_flat[l_col].value_counts().head(15).sort_values().plot(kind='barh', ax=ax, color='skyblue')
        ax.set_title(f"Top 15 Labels ({l_col})")
        fig.savefig(out / "eda_labels_diversity.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    pd.Series(term_counts).sort_values().plot(kind='barh', ax=ax, color='steelblue')
    ax.set_title("Distribution des termes PR")
    fig.savefig(out / "eda_ra_terms.png")

    if patient_stats_list:
        df_p = pd.DataFrame(patient_stats_list)
        fig, ax = plt.subplots(figsize=(8, 4))
        df_p["n_stays"].value_counts().sort_index().plot(kind='bar', ax=ax, color='teal')
        ax.set_title("Nombre de séjours par patient")
        fig.savefig(out / "eda_stays_dist.png")

        if "profile" in df_p.columns and df_p["profile"].nunique() > 1:
            fig, ax = plt.subplots(figsize=(8, 8))
            df_p["profile"].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax, cmap='Pastel1')
            ax.set_title("Répartition des profils cliniques")
            fig.savefig(out / "eda_profiles_pie.png")

    plt.close('all')
    print(f"\n  Visualisations enregistrées dans : {out}")
    print(f"  - eda_missing.png, eda_labels_diversity.png, eda_ra_terms.png")
    if patient_stats_list: print(f"  - eda_stays_dist.png, eda_profiles_pie.png")
    print(f"{'='*75}")

# ════════════════════════════════════════════════════════════════
# EVALUATION
# ════════════════════════════════════════════════════════════════

def _normalize_id(pid) -> str:
    """Normalise un patient_id en string sans décimale (10001.0 → '10001')."""
    try:
        return str(int(float(str(pid).strip())))
    except (ValueError, TypeError):
        return str(pid).strip()


def _load_gt(gt_path: str, gt_patient_agg: str = "any_positive") -> Optional[Dict[str, Any]]:
    """
    Charge le GT (CSV ou JSON).

    CSV attendu : id, NIP, NDA, PR
      PR: T=RA+  F=RA-  D=douteux (exclu du calcul stay-level, compté RA- au patient-level)

    Retourne un dict avec deux espaces de clés :
      "10001"        → label patient agrégé  (any_positive | majority | all_positive | confirmed)
      "stay:NDA..."  → label séjour          (T→RA+  F→RA-  D→None=ignoré)
    """
    import csv as _csv

    if not Path(gt_path).exists():
        print(f"  [EVAL] Ground truth introuvable : {gt_path} — évaluation ignorée.")
        return None

    gt_ext = Path(gt_path).suffix.lower()

    if gt_ext == ".csv":
        # ── Lecture ──────────────────────────────────────────────
        rows = []
        with open(gt_path, "r", encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            cols = [c.strip() for c in (reader.fieldnames or [])]
            col_map = {c.lower(): c for c in cols}

            pid_col   = col_map.get("nip") or col_map.get("patient_id") or col_map.get("id")
            stay_col  = col_map.get("nda") or col_map.get("stay_id") or col_map.get("nisejour")
            label_col = col_map.get("pr")  or col_map.get("phenotype") or col_map.get("label")

            if not pid_col or not label_col:
                print(f"  [EVAL] Colonnes GT non reconnues. Trouvées : {cols}")
                return None

            for row in reader:
                pid   = _normalize_id(row.get(pid_col, ""))
                raw   = str(row.get(label_col, "")).strip().upper()
                sid   = str(row.get(stay_col, "")).strip() if stay_col else None
                rows.append((pid, sid, raw))

        # ── Agrégation patient-level ──────────────────────────────
        from collections import defaultdict
        stay_labels: Dict[str, list] = defaultdict(list)  # pid → [T/F/D ...]
        gt: Dict[str, Any] = {}

        for pid, sid, raw in rows:
            # Stay-level : T→RA+  F→RA-  D→ignoré (douteux)
            if sid:
                if raw == "T":
                    gt[f"stay:{sid}"] = "RA+"
                elif raw == "F":
                    gt[f"stay:{sid}"] = "RA-"
                # D : pas de clé stay → ignoré dans stay-level eval
            stay_labels[pid].append(raw)

        for pid, labels in stay_labels.items():
            definitive = [l for l in labels if l in ("T", "F")]
            n_pos = labels.count("T")
            n_def = len(definitive)
            n_pos_def = definitive.count("T")

            if gt_patient_agg == "any_positive":
                patient_label = "RA+" if n_pos >= 1 else "RA-"
            elif gt_patient_agg == "majority":
                patient_label = "RA+" if n_pos_def > n_def / 2 else "RA-"
            elif gt_patient_agg == "all_positive":
                patient_label = "RA+" if n_pos_def == n_def and n_def > 0 else "RA-"
            elif gt_patient_agg == "confirmed":
                # confirmed : au moins 2 séjours T sans aucun F
                patient_label = "RA+" if n_pos_def >= 2 and definitive.count("F") == 0 else "RA-"
            else:
                patient_label = "RA+" if n_pos >= 1 else "RA-"

            gt[pid] = patient_label

        print(f"  [GT] {len(stay_labels)} patients | "
              f"{sum(1 for k in gt if not k.startswith('stay:') and gt[k]=='RA+')} RA+ | "
              f"{sum(1 for k in gt if k.startswith('stay:'))} séjours définis "
              f"(agg={gt_patient_agg})")
        return gt

    # ── JSON ─────────────────────────────────────────────────────
    with open(gt_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        print(f"  [EVAL] Ground truth vide : {gt_path} — évaluation ignorée.")
        return None
    raw = json.loads(content)
    return {_normalize_id(k): v for k, v in raw.items()}


def _compute_metrics(pairs: List[tuple], label: str) -> Dict[str, Any]:
    """Calcule TP/FP/TN/FN/F1 depuis une liste de (pred, true)."""
    tp = fp = tn = fn = 0
    errors = []
    for pred, true, pid in pairs:
        if pred == "RA+" and true == "RA+": tp += 1
        elif pred == "RA+" and true == "RA-":
            fp += 1; errors.append({"id": pid, "pred": pred, "true": true, "type": "FP"})
        elif pred == "RA-" and true == "RA-": tn += 1
        elif pred == "RA-" and true == "RA+":
            fn += 1; errors.append({"id": pid, "pred": pred, "true": true, "type": "FN"})
    total = tp + fp + tn + fn
    acc  = (tp + tn) / total if total else 0
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec  = tp / (tp + fn) if (tp + fn) else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    print(f"  [{label}] TP={tp} FP={fp} TN={tn} FN={fn} | "
          f"Acc={acc:.1%} Prec={prec:.1%} Rec={rec:.1%} F1={f1:.1%} "
          f"(n={total} matchés)")
    return {"accuracy": round(acc, 4), "precision": round(prec, 4),
            "recall": round(rec, 4), "f1": round(f1, 4),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn, "errors": errors}


def evaluate(run_dir: str, gt_path: str, gt_patient_agg: str = "any_positive"):
    """Évaluation patient-level ET séjour-level si le GT contient les séjours."""
    decisions_path = Path(run_dir) / "decisions.json"
    assess_path    = Path(run_dir) / "assessments.jsonl"
    metrics_path   = Path(run_dir) / "metrics.json"

    if not decisions_path.exists():
        print(f"  [EVAL] Fichier introuvable : {decisions_path}")
        return

    with open(decisions_path, "r", encoding="utf-8") as f:
        decisions = json.load(f)

    # Récupère la stratégie d'agrégation depuis metrics existant ou défaut
    gt = _load_gt(gt_path, gt_patient_agg=gt_patient_agg)
    if gt is None:
        return

    print(f"\n{'─'*60}")
    print(f"EVALUATION — {Path(run_dir).name}")
    print(f"{'─'*60}")

    # ── 1. Patient-level ──────────────────────────────────────────
    patient_pairs = []
    for d in decisions:
        pid = _normalize_id(d.get("patient_id", ""))
        pred = d.get("final_label", "")
        true_val = gt.get(pid)
        if true_val is None:
            continue
        true_label = true_val.get("phenotype") if isinstance(true_val, dict) else str(true_val)
        patient_pairs.append((pred, true_label, pid))

    patient_metrics = _compute_metrics(patient_pairs, "Patient-level")

    # ── 2. Séjour-level (si assessments.jsonl disponible et GT séjour) ──
    stay_metrics = None
    has_stay_gt = any(k.startswith("stay:") for k in gt)
    if assess_path.exists() and has_stay_gt:
        with open(assess_path, "r", encoding="utf-8") as f:
            assessments = [json.loads(l) for l in f]

        stay_pairs = []
        for a in assessments:
            sid = str(a.get("stay_id", "")).strip().replace("NDA", "")
            true_label = gt.get(f"stay:{sid}")
            if true_label is None:
                continue
            pred = a.get("final_stay_label", "")
            stay_pairs.append((pred, true_label, sid))
        if stay_pairs:
            stay_metrics = _compute_metrics(stay_pairs, "Séjour-level ")
        else:
            print("  [Séjour-level] Aucun séjour matché dans le GT.")
    elif not has_stay_gt:
        print("  [Séjour-level] GT ne contient pas de colonne séjour — ignoré.")

    metrics = {"patient_level": patient_metrics}
    if stay_metrics:
        metrics["stay_level"] = stay_metrics

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"  Métriques enregistrées → {metrics_path}")

# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="PhenoRAG Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--arch", default="B1", choices=["B1", "B2", "B3", "B4"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--skip-extraction", action="store_true")
    parser.add_argument("--only", choices=["eval", "preprocess", "eda"])
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--corpus", default=CORPUS_PATH)
    parser.add_argument("--format", default="auto", choices=["auto", "json", "parquet"])
    parser.add_argument("--gt", default=GROUND_TRUTH_PATH)
    parser.add_argument("--no-kb", action="store_true")
    parser.add_argument("--kb", default=KB_PATH)
    parser.add_argument("--ollama-url", default=OLLAMA_URL)
    parser.add_argument("--prompts-dir", default=PROMPTS_DIR)
    # ── GT séjour-level + sélection patients + checkpoint ──────────
    parser.add_argument("--gt-stay-csv", default=None,
                        help="CSV ground truth séjour-level (id,NIP,NDA,PR)")
    parser.add_argument("--n-patients", type=int, default=None,
                        help="Nombre de patients à traiter (N premiers du fichier)")
    parser.add_argument("--checkpoint-interval", type=float, default=5.0,
                        help="Intervalle de checkpoint en minutes (défaut: 5)")
    # PATCH: ajout de "confirmed" dans les choix d'agrégation (EXP-E3)
    parser.add_argument("--gt-patient-agg", default="any_positive",
                        choices=["any_positive", "majority", "all_positive", "confirmed"],
                        help="Agrégation GT séjour→patient")
    # PATCH: flag inter-stay RAG (référencé dans plan.yaml axe D)
    parser.add_argument("--no-inter-stay-rag", action="store_true",
                        help="Désactive le RAG inter-séjours")
    # PATCH: surcharge des fichiers de prompts (axe C du plan.yaml)
    parser.add_argument("--prompt-acr-scoring", default=None,
                        help="Chemin relatif du prompt ACR scoring (ex: agent2/acr_scoring_fewshot)")
    parser.add_argument("--prompt-ra-relatedness", default=None,
                        help="Chemin relatif du prompt RA relatedness (ex: agent2/ra_relatedness_fewshot)")
    args = parser.parse_args()

    run_dir = args.run_dir or f"runs/{args.arch.lower()}"
    Path(run_dir).mkdir(parents=True, exist_ok=True)

    if args.only == "eda":
        run_eda(args.corpus, str(Path(run_dir) / "eda"), args.gt)
        return

    # Détection format
    input_format = args.format
    if input_format == "auto":
        input_format = "parquet" if args.corpus.lower().endswith(".parquet") else "json"

    # Preprocess
    corpus_path = args.corpus
    if input_format == "parquet":
        from phenorag.preprocess import preprocess_parquet
        preprocess_dir = Path(run_dir) / "preprocessed"
        corpus_path = preprocess_parquet(args.corpus, str(preprocess_dir), gt_path=args.gt)
    if args.only == "preprocess": return

    if args.only == "eval":
        evaluate(run_dir, args.gt, gt_patient_agg=args.gt_patient_agg)
        return

    # Exécution Pipeline
    use_kb = not args.no_kb and Path(args.kb).exists()
    use_inter_stay_rag = not args.no_inter_stay_rag

    print(f"\nLancement PhenoRAG — Arch: {args.arch} | Modèle: {args.model}")
    print(f"  KB={use_kb} | InterStayRAG={use_inter_stay_rag} | Agg={args.gt_patient_agg}")

    llm = LLMClient(args.ollama_url)
    prompts = PromptLoader(args.prompts_dir)

    # Surcharge des prompts si spécifiés (axe C)
    if args.prompt_acr_scoring:
        prompts.override("acr_scoring", args.prompt_acr_scoring)
    if args.prompt_ra_relatedness:
        prompts.override("ra_relatedness", args.prompt_ra_relatedness)

    pipeline_cfg = PipelineConfig(
        architecture=args.arch,
        b4_models=B4_MODELS,
        input_format=input_format,
        use_inter_stay_rag=use_inter_stay_rag,
        gt_patient_agg=args.gt_patient_agg,
    )
    a1_cfg = Agent1Config(model=args.model, use_kb_guidance=use_kb, input_format=input_format)
    a2_cfg = Agent2Config(model=args.model)
    a3_cfg = Agent3Config(model=args.model, use_critic=(args.arch in ("B2", "B3")))

    orch = Orchestrator(
        llm=llm, prompts=prompts, pipeline_cfg=pipeline_cfg,
        a1_cfg=a1_cfg, a2_cfg=a2_cfg, a3_cfg=a3_cfg,
        kb_path=args.kb if use_kb else None,
        n_patients=args.n_patients,
        checkpoint_interval=args.checkpoint_interval,
        gt_stay_csv=args.gt_stay_csv,
    )

    orch.run(corpus_path, run_dir, skip_extraction=args.skip_extraction)
    evaluate(run_dir, args.gt, gt_patient_agg=args.gt_patient_agg)

if __name__ == "__main__":
    main()