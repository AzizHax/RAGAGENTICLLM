#!/usr/bin/env python3
"""
eda_noise_sources.py
====================

EDA ciblée sur un parquet EHR brut pour anticiper les sources de bruit
qui peuvent piéger le pipeline PhenoRAG (faux positifs / faux négatifs).

Sections du rapport :
  1. Vue d'ensemble : patients, séjours, records, complétude
  2. Distribution TYPE_QUEST (IDE vs MEDECIN) — quel volume utile ?
  3. Top LIBELLE — items dominants, signal/bruit
  4. Lexique PR : recherche pondérée des termes PR
  5. Faux signaux lexicaux : "articulation", "rhumato", "chronique"
     présents hors-PR (items soignants, traumato, arthrose…)
  6. Quantification labs : RF/anti-CCP/CRP — combien de records, formats
  7. Profils par patient : longueur séjours, dispersion records
  8. Recommandations actionnables

Usage
-----
    python eda_noise_sources.py --parquet data/real/cohorte.parquet \
                                --gt data/real/ground_truth_stays.csv \
                                --output reports/eda_noise.html

GT optionnel : si fourni, on stratifie l'analyse PR+ / PR- / PR=D.
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import re
from collections import Counter, defaultdict
from html import escape as html_escape
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL = True
except ImportError:
    _MPL = False


# ════════════════════════════════════════════════════════════════
# LEXIQUE PR
# ════════════════════════════════════════════════════════════════

# Termes très spécifiques (signal fort, peu de faux positifs lexicaux)
PR_STRONG_TERMS = {
    "polyarthrite": r"polyarthrite\s+rhumato[ïi]de|^PR\b|polyarthrite",
    "anti-CCP": r"anti[- ]?CCP|ACPA",
    "facteur_rhumatoide": r"facteur\s+rhumato[ïi]de|\bRF\b",
    "methotrexate": r"m[eé]thotrexate|\bMTX\b",
    "anti-TNF": r"anti[- ]?TNF|adalimumab|etanercept|infliximab|certolizumab|golimumab",
    "biotherapie_pr": r"tocilizumab|rituximab|abatacept",
    "JAK_inhib": r"tofacitinib|baricitinib|upadacitinib|filgotinib",
}

# Termes ambigus (peuvent apparaître hors PR)
PR_AMBIGUOUS_TERMS = {
    "articulaire": r"articulaire|articulation",
    "rhumato": r"rhumato|rhumatologie",
    "synovite": r"synovit",
    "douleur_articulaire": r"douleur.{0,30}articul|articul.{0,30}douleur",
    "raideur": r"raideur|d[eé]rouill",
    "chronique": r"chronique",
    "inflammation": r"inflamm",
    "tumefaction": r"tumefact|gonfle",
}

# Termes de bruit (apparaissent dans soins infirmiers, AVQ, traumato)
NOISE_CONTEXTS = {
    "AVQ_soin": r"\beviter\b|\bse\s+mouvoir\b|\bcommuniquer\b|\bassurer\b.{0,20}confort|\beliminer\b|\bboire\b",
    "traumato": r"chute|traumat|fracture|entorse|luxation",
    "arthrose": r"arthrose|gonarthrose|coxarthrose",
    "infection": r"infection|sepsis|pneumonie|cystite",
    "constantes": r"\bPAS\b|\bPAD\b|fr[eé]quence\s+cardia|saturation|temp[eé]rature",
}


# ════════════════════════════════════════════════════════════════
# UTILITAIRES PLOT
# ════════════════════════════════════════════════════════════════

def _fig_to_b64(fig) -> str:
    """Convertit une figure matplotlib en data-URI PNG base64."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _bar_plot(labels: List[str], values: List[float], title: str,
              xlabel: str = "", ylabel: str = "", color: str = "steelblue",
              horizontal: bool = True, figsize=(10, 5)) -> Optional[str]:
    if not _MPL or not labels:
        return None
    fig, ax = plt.subplots(figsize=figsize)
    if horizontal:
        ax.barh(labels[::-1], values[::-1], color=color)
        ax.set_xlabel(ylabel or "Count")
    else:
        ax.bar(labels, values, color=color)
        ax.set_ylabel(ylabel or "Count")
        plt.xticks(rotation=45, ha="right")
    ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    fig.tight_layout()
    return _fig_to_b64(fig)


def _stacked_bar(categories: List[str], stacks: Dict[str, List[float]],
                  title: str, colors: Dict[str, str] = None) -> Optional[str]:
    if not _MPL or not categories:
        return None
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = [0] * len(categories)
    colors = colors or {}
    for label, vals in stacks.items():
        ax.bar(categories, vals, bottom=bottom, label=label,
               color=colors.get(label))
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_title(title)
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    return _fig_to_b64(fig)


# ════════════════════════════════════════════════════════════════
# CHARGEMENT GT (optionnel)
# ════════════════════════════════════════════════════════════════

def load_gt_optional(gt_path: Optional[str]) -> Dict[str, str]:
    """Retourne {NIP: 'T'|'F'|'D'} agrégé patient (any_positive)."""
    if not gt_path or not Path(gt_path).exists():
        return {}
    nip_to_pr: Dict[str, List[str]] = defaultdict(list)
    with open(gt_path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            nip = (row.get("NIP") or "").strip()
            pr = (row.get("PR") or "").strip().upper().strip('"')
            if nip and pr in ("T", "F", "D"):
                nip_to_pr[nip].append(pr)
    out = {}
    for nip, prs in nip_to_pr.items():
        if "T" in prs:
            out[nip] = "T"
        elif "D" in prs:
            out[nip] = "D"
        else:
            out[nip] = "F"
    return out


# ════════════════════════════════════════════════════════════════
# SECTIONS D'ANALYSE
# ════════════════════════════════════════════════════════════════

def section_overview(df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "n_records": len(df),
        "n_patients": df["NIPATIENT"].nunique() if "NIPATIENT" in df else 0,
        "n_stays": df["NISEJOUR"].nunique() if "NISEJOUR" in df else 0,
        "n_libelle_unique": df["LIBELLE"].nunique() if "LIBELLE" in df else 0,
        "missing_libelle_pct": round(100 * df["LIBELLE"].isna().mean(), 2)
                                if "LIBELLE" in df else 0,
        "missing_reponse_pct": round(100 * df["REPONSE"].isna().mean(), 2)
                                if "REPONSE" in df else 0,
    }


def section_typequest(df: pd.DataFrame) -> Dict[str, Any]:
    if "TYPE_QUEST" not in df.columns:
        return {}
    counts = df["TYPE_QUEST"].fillna("(missing)").value_counts().to_dict()
    total = sum(counts.values())
    return {
        "counts": counts,
        "ratios": {k: round(100*v/total, 2) for k, v in counts.items()},
        "plot": _bar_plot(
            list(counts.keys()), list(counts.values()),
            "Distribution TYPE_QUEST", horizontal=False,
            color="darkorange", figsize=(8, 4)),
    }


def section_top_libelles(df: pd.DataFrame, top_n: int = 25) -> Dict[str, Any]:
    if "LIBELLE" not in df.columns:
        return {}
    top = df["LIBELLE"].fillna("(missing)").value_counts().head(top_n)
    return {
        "top_n": top_n,
        "items": list(top.items()),
        "plot": _bar_plot(
            [str(k)[:60] for k in top.index],
            list(top.values),
            f"Top {top_n} LIBELLE",
            color="steelblue", figsize=(10, 8)),
    }


def section_pr_lexicon(df: pd.DataFrame, gt: Dict[str, str]) -> Dict[str, Any]:
    """Recherche des termes PR dans LIBELLE+REPONSE et stratification GT."""
    text = (df["LIBELLE"].fillna("") + " " + df["REPONSE"].fillna("")).str.lower()

    results = {"strong": {}, "ambiguous": {}}

    for category, terms in (("strong", PR_STRONG_TERMS),
                              ("ambiguous", PR_AMBIGUOUS_TERMS)):
        for name, pattern in terms.items():
            mask = text.str.contains(pattern, regex=True, na=False)
            n_records = int(mask.sum())
            n_patients = df.loc[mask, "NIPATIENT"].nunique() if n_records else 0
            n_stays = df.loc[mask, "NISEJOUR"].nunique() if n_records else 0

            # Stratification par GT si disponible
            stratification = {}
            if gt and n_records:
                # Convert NIPATIENT to str for match
                touched_nips = df.loc[mask, "NIPATIENT"].astype(str).str.strip().unique()
                for nip in touched_nips:
                    pr = gt.get(nip)
                    if pr:
                        stratification.setdefault(pr, set()).add(nip)
                stratification = {k: len(v) for k, v in stratification.items()}

            results[category][name] = {
                "n_records": n_records,
                "n_patients": n_patients,
                "n_stays": n_stays,
                "stratification": stratification,
            }

    # Plot synthèse
    plot = None
    if _MPL:
        all_terms = list(results["strong"].keys()) + list(results["ambiguous"].keys())
        all_n = [results["strong"].get(t, results["ambiguous"].get(t, {})).get("n_records", 0)
                 for t in all_terms]
        colors = ["#2c7a3e"] * len(results["strong"]) + ["#c9842b"] * len(results["ambiguous"])
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(all_terms[::-1], all_n[::-1], color=colors[::-1])
        ax.set_title("Termes PR dans LIBELLE+REPONSE\n(vert = forts, orange = ambigus)")
        ax.set_xlabel("Nombre de records")
        fig.tight_layout()
        plot = _fig_to_b64(fig)

    return {"results": results, "plot": plot}


def section_noise_contexts(df: pd.DataFrame, gt: Dict[str, str]) -> Dict[str, Any]:
    """Détecte les contextes de bruit qui peuvent piéger l'extraction."""
    text = (df["LIBELLE"].fillna("") + " " + df["REPONSE"].fillna("")).str.lower()

    # Termes ambigus PR
    pr_ambiguous_mask = pd.Series(False, index=df.index)
    for pat in PR_AMBIGUOUS_TERMS.values():
        pr_ambiguous_mask |= text.str.contains(pat, regex=True, na=False)

    # Pour chaque contexte de bruit : combien de records contiennent
    # à la fois un terme ambigu PR ET un signal de bruit ?
    results = {}
    for noise_name, noise_pat in NOISE_CONTEXTS.items():
        noise_mask = text.str.contains(noise_pat, regex=True, na=False)
        confused = pr_ambiguous_mask & noise_mask
        n = int(confused.sum())
        if n == 0:
            continue
        examples = df.loc[confused, ["LIBELLE", "REPONSE"]].head(5)
        results[noise_name] = {
            "n_confused_records": n,
            "examples": [
                f"{str(r['LIBELLE'])[:60]}: {str(r['REPONSE'])[:80]}"
                for _, r in examples.iterrows()
            ],
        }

    # Plot
    plot = None
    if _MPL and results:
        names = list(results.keys())
        vals = [results[k]["n_confused_records"] for k in names]
        plot = _bar_plot(
            names, vals,
            "Records avec confusion PR-ambigu × bruit",
            color="firebrick", figsize=(10, 4))

    return {"results": results, "plot": plot}


def section_lab_quantification(df: pd.DataFrame) -> Dict[str, Any]:
    """Quantifie la présence de RF, anti-CCP, CRP avec valeurs numériques."""
    text = (df["LIBELLE"].fillna("") + " " + df["REPONSE"].fillna("")).str.lower()

    NUMERIC = re.compile(r"\d+(?:[.,]\d+)?")
    results = {}
    for name, pat in [
        ("RF", r"\bRF\b|facteur\s+rhumato"),
        ("anti-CCP", r"anti[- ]?CCP|ACPA"),
        ("CRP", r"\bCRP\b"),
        ("VS_ESR", r"\bVS\b|ESR"),
    ]:
        mask = text.str.contains(pat, regex=True, na=False)
        n_records = int(mask.sum())
        if n_records == 0:
            results[name] = {"n_records": 0, "n_with_value": 0, "with_value_pct": 0,
                              "with_norm_pct": 0, "examples": []}
            continue

        rep_text = df.loc[mask, "REPONSE"].fillna("").astype(str)
        with_value = rep_text.apply(lambda s: bool(NUMERIC.search(s))).sum()
        with_norm = rep_text.str.contains(r"N\s*<", regex=True, case=False).sum()

        # Échantillons
        examples = []
        for _, r in df.loc[mask].head(5).iterrows():
            examples.append(f"{str(r['LIBELLE'])[:50]}: {str(r['REPONSE'])[:80]}")

        results[name] = {
            "n_records": n_records,
            "n_with_value": int(with_value),
            "with_value_pct": round(100 * with_value / n_records, 1),
            "with_norm_pct": round(100 * with_norm / n_records, 1),
            "examples": examples,
        }

    # Plot
    plot = None
    if _MPL and any(v["n_records"] for v in results.values()):
        names = list(results.keys())
        n_total = [results[n]["n_records"] for n in names]
        n_value = [results[n]["n_with_value"] for n in names]
        x = range(len(names))
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(x, n_total, label="Total mentions", color="lightgray")
        ax.bar(x, n_value, label="Avec valeur numérique", color="seagreen")
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.legend()
        ax.set_title("Mentions de labs : présence de valeurs numériques")
        ax.set_ylabel("Nombre de records")
        fig.tight_layout()
        plot = _fig_to_b64(fig)

    return {"results": results, "plot": plot}


def section_patient_profiles(df: pd.DataFrame, gt: Dict[str, str]) -> Dict[str, Any]:
    """Distribution séjours/patient et records/séjour, stratifié par GT."""
    if "NIPATIENT" not in df or "NISEJOUR" not in df:
        return {}

    stays_per_pat = df.groupby("NIPATIENT")["NISEJOUR"].nunique()
    records_per_stay = df.groupby("NISEJOUR").size()

    stats = {
        "stays_per_patient": {
            "min": int(stays_per_pat.min()),
            "max": int(stays_per_pat.max()),
            "mean": round(stays_per_pat.mean(), 1),
            "median": int(stays_per_pat.median()),
        },
        "records_per_stay": {
            "min": int(records_per_stay.min()),
            "max": int(records_per_stay.max()),
            "mean": round(records_per_stay.mean(), 1),
            "median": int(records_per_stay.median()),
        },
    }

    # Stratification par GT
    if gt:
        gt_strata: Dict[str, List[int]] = {"T": [], "F": [], "D": []}
        for nip, n_stays in stays_per_pat.items():
            pr = gt.get(str(nip).strip())
            if pr in gt_strata:
                gt_strata[pr].append(int(n_stays))
        stats["stays_by_gt"] = {
            k: {"n_patients": len(v),
                "mean_stays": round(sum(v)/len(v), 1) if v else 0,
                "max_stays": max(v) if v else 0}
            for k, v in gt_strata.items()
        }

    # Plot
    plot = None
    if _MPL:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].hist(stays_per_pat.values, bins=range(1, max(stays_per_pat.max()+2, 10)),
                     color="steelblue", edgecolor="black")
        axes[0].set_title("Séjours par patient")
        axes[0].set_xlabel("Nombre de séjours")
        axes[0].set_ylabel("Nombre de patients")
        axes[1].hist(records_per_stay.values, bins=30,
                     color="darkorange", edgecolor="black")
        axes[1].set_title("Records par séjour")
        axes[1].set_xlabel("Nombre de records")
        axes[1].set_ylabel("Nombre de séjours")
        fig.tight_layout()
        plot = _fig_to_b64(fig)

    return {"stats": stats, "plot": plot}


def section_recommendations(overview: Dict, typequest: Dict, top_lib: Dict,
                              pr_lex: Dict, noise: Dict, labs: Dict) -> List[str]:
    """Génère des recommandations actionnables."""
    recs = []

    # Volume IDE/MEDECIN
    if typequest.get("ratios"):
        ratios = typequest["ratios"]
        ide_pct = ratios.get("IDE", 0)
        if ide_pct > 80:
            recs.append(
                f"⚠ <b>Bruit IDE massif ({ide_pct:.0f}%)</b> : la majorité des records "
                "sont des constantes infirmières (FC, T°, PAS…). Considère filtrer "
                "<code>TYPE_QUEST=='MEDECIN'</code> avant le pre-filtrage par ancres "
                "pour réduire de ~80% le volume sans perdre de signal PR.")

    # Top LIBELLE non-cliniques
    if top_lib.get("items"):
        non_clinical_top = []
        for libelle, count in top_lib["items"][:10]:
            lib_l = str(libelle).lower()
            if any(p in lib_l for p in ["constant", "fr.quence", "temp.rature",
                                          "saturation", "pas", "pad", "eviter",
                                          "mouvoir", "communiquer", "confort",
                                          "allergie", "diagramme"]):
                non_clinical_top.append(libelle)
        if non_clinical_top:
            recs.append(
                f"⚠ <b>Top {len(non_clinical_top)} LIBELLE non-cliniques</b> dominent : "
                f"{', '.join(str(x)[:30] for x in non_clinical_top[:5])}. "
                "Vérifie que ton pre-filtrage par ancres exclut bien ces libellés.")

    # Termes ambigus dans contextes de bruit
    if noise.get("results"):
        problematic = sorted(noise["results"].items(),
                              key=lambda x: x[1]["n_confused_records"],
                              reverse=True)
        if problematic:
            top_noise = problematic[0]
            recs.append(
                f"⚠ <b>Confusion lexicale</b> : <code>{top_noise[0]}</code> "
                f"({top_noise[1]['n_confused_records']} records contiennent à la fois "
                "un terme PR ambigu ET un contexte de bruit). Source potentielle de FP. "
                "Exemples : "
                f"{', '.join(top_noise[1]['examples'][:2])}")

    # Quantification labs
    if labs.get("results"):
        for lab_name, info in labs["results"].items():
            if info["n_records"] > 0 and info["with_value_pct"] < 50:
                recs.append(
                    f"⚠ <b>Lab '{lab_name}'</b> : seulement "
                    f"{info['with_value_pct']:.0f}% des mentions ont une valeur "
                    f"numérique extractible ({info['n_with_value']}/{info['n_records']}). "
                    "Le LLM va galérer à inférer la polarity. Renforce le regex fallback "
                    "côté Agent 1.")
            if info["n_records"] > 0 and info["with_norm_pct"] > 30:
                recs.append(
                    f"✓ <b>Lab '{lab_name}'</b> : {info['with_norm_pct']:.0f}% des "
                    "mentions contiennent la norme 'N<X'. Ton patch <code>_infer_lab_polarity</code> "
                    "qui utilise NORM_LT devrait bien fonctionner.")

    # Stratification PR
    if pr_lex.get("results", {}).get("strong"):
        rf = pr_lex["results"]["strong"].get("facteur_rhumatoide", {})
        ccp = pr_lex["results"]["strong"].get("anti-CCP", {})
        if rf.get("n_records", 0) > 0 and ccp.get("n_records", 0) > 0:
            ratio = ccp["n_records"] / rf["n_records"]
            if ratio < 0.3:
                recs.append(
                    f"⚠ <b>Anti-CCP sous-mesuré</b> : ratio anti-CCP/RF = {ratio:.2f}. "
                    "Beaucoup de patients ont RF sans anti-CCP. La sérologie ACR-EULAR sera "
                    "moins discriminante. Prévois que des FP RF-borderline-isolés "
                    "apparaîtront — ton <code>strict_any_positive</code> sera utile.")

    # Volumétrie globale
    n_libelle = overview.get("n_libelle_unique", 0)
    if n_libelle > 1000:
        recs.append(
            f"⚠ <b>Polysémie élevée</b> : {n_libelle} LIBELLE distincts. Un même concept "
            "peut être nommé plusieurs façons (ex : 'RF', 'Facteur rhumatoïde', 'Latex RF'). "
            "Vérifie que tes patterns regex couvrent les variantes courantes.")

    if not recs:
        recs.append("✓ Aucun signal de bruit majeur détecté à ce stade.")

    return recs


# ════════════════════════════════════════════════════════════════
# RENDU HTML
# ════════════════════════════════════════════════════════════════

CSS = """
body { font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif;
       max-width: 1100px; margin: 24px auto; padding: 0 20px;
       color: #222; line-height: 1.5; }
h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 8px; }
h2 { color: #34495e; margin-top: 36px; border-bottom: 1px solid #ddd;
     padding-bottom: 6px; }
h3 { color: #555; margin-top: 22px; }
table { border-collapse: collapse; margin: 12px 0; }
th, td { padding: 6px 12px; border: 1px solid #ccc; text-align: left;
         font-size: 14px; }
th { background: #ecf0f1; }
tr:nth-child(even) { background: #fafbfc; }
.kv { display: inline-block; padding: 2px 6px; background: #ecf0f1;
      border-radius: 3px; font-family: monospace; font-size: 13px; }
.warning { background: #fff3cd; border-left: 4px solid #f39c12;
           padding: 12px; margin: 8px 0; border-radius: 3px; }
.success { background: #d4edda; border-left: 4px solid #27ae60;
           padding: 12px; margin: 8px 0; border-radius: 3px; }
.recos li { margin: 8px 0; }
img.plot { max-width: 100%; height: auto; display: block; margin: 12px 0;
           border: 1px solid #ddd; border-radius: 4px; }
code { background: #f4f6f8; padding: 1px 5px; border-radius: 3px;
       font-size: 13px; }
.example { font-family: monospace; font-size: 12px; color: #555;
           background: #f8f9fa; padding: 4px 8px; margin: 2px 0;
           border-left: 2px solid #bbb; }
.gt-T { color: #27ae60; font-weight: bold; }
.gt-F { color: #c0392b; }
.gt-D { color: #f39c12; }
"""


def _table(headers: List[str], rows: List[List[Any]]) -> str:
    h = "".join(f"<th>{html_escape(str(x))}</th>" for x in headers)
    body = ""
    for row in rows:
        body += "<tr>" + "".join(f"<td>{html_escape(str(c))}</td>" for c in row) + "</tr>"
    return f"<table><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table>"


def _img(b64: Optional[str], alt: str = "") -> str:
    if not b64:
        return "<p><em>(graphique non disponible)</em></p>"
    return f'<img class="plot" src="data:image/png;base64,{b64}" alt="{html_escape(alt)}">'


def _strat_html(strat: Dict[str, int]) -> str:
    if not strat:
        return ""
    parts = []
    for k in ("T", "F", "D"):
        if k in strat:
            parts.append(f'<span class="gt-{k}">{k}={strat[k]}</span>')
    return " | ".join(parts)


def render_html(parquet_path: str, gt_path: Optional[str],
                 overview: Dict, typequest: Dict, top_lib: Dict,
                 pr_lex: Dict, noise: Dict, labs: Dict, profiles: Dict,
                 recommendations: List[str]) -> str:
    parts = [f"<style>{CSS}</style>",
             "<h1>EDA — Sources de bruit pour pipeline PhenoRAG</h1>",
             f"<p><b>Source :</b> <code>{html_escape(parquet_path)}</code></p>"]
    if gt_path:
        parts.append(f"<p><b>GT :</b> <code>{html_escape(gt_path)}</code></p>")

    # 1. Overview
    parts.append("<h2>1. Vue d'ensemble</h2>")
    parts.append(_table(
        ["Métrique", "Valeur"],
        [["Records", f"{overview['n_records']:,}"],
         ["Patients uniques", overview["n_patients"]],
         ["Séjours uniques", overview["n_stays"]],
         ["LIBELLE distincts", overview["n_libelle_unique"]],
         ["LIBELLE manquants", f"{overview['missing_libelle_pct']}%"],
         ["REPONSE manquantes", f"{overview['missing_reponse_pct']}%"]]))

    # 2. TYPE_QUEST
    if typequest:
        parts.append("<h2>2. Distribution TYPE_QUEST</h2>")
        rows = [[k, f"{v:,}", f"{typequest['ratios'][k]}%"]
                for k, v in typequest["counts"].items()]
        parts.append(_table(["Type", "Records", "%"], rows))
        parts.append(_img(typequest.get("plot"), "TYPE_QUEST"))

    # 3. Top LIBELLE
    if top_lib:
        parts.append(f"<h2>3. Top {top_lib['top_n']} LIBELLE</h2>")
        parts.append(_img(top_lib.get("plot"), "Top LIBELLE"))

    # 4. Lexique PR
    if pr_lex.get("results"):
        parts.append("<h2>4. Lexique PR dans le corpus</h2>")
        parts.append(_img(pr_lex.get("plot"), "Lexique PR"))
        for cat_name, cat_label in (("strong", "Termes forts (haute spécificité)"),
                                     ("ambiguous", "Termes ambigus")):
            parts.append(f"<h3>{cat_label}</h3>")
            rows = []
            for term, info in pr_lex["results"][cat_name].items():
                rows.append([term, info["n_records"], info["n_patients"],
                             info["n_stays"], _strat_html(info["stratification"])])
            parts.append(_table(
                ["Terme", "N records", "N patients", "N séjours", "GT"],
                rows))

    # 5. Confusions lexicales
    if noise.get("results"):
        parts.append("<h2>5. Confusions lexicales (PR ambigu × bruit)</h2>")
        parts.append("<p>Records contenant à la fois un terme PR ambigu "
                     "(ex : 'articulaire') ET un contexte non-PR "
                     "(soin AVQ, traumato, arthrose…). Source classique de FP.</p>")
        parts.append(_img(noise.get("plot"), "Confusions"))
        for noise_name, info in noise["results"].items():
            parts.append(f"<h3>Contexte : <code>{noise_name}</code> "
                         f"({info['n_confused_records']} records)</h3>")
            for ex in info["examples"]:
                parts.append(f'<div class="example">{html_escape(ex)}</div>')

    # 6. Quantification labs
    if labs.get("results"):
        parts.append("<h2>6. Quantification des labs (RF, anti-CCP, CRP, VS)</h2>")
        parts.append(_img(labs.get("plot"), "Labs"))
        rows = []
        for name, info in labs["results"].items():
            rows.append([name, info["n_records"], info["n_with_value"],
                         f"{info['with_value_pct']}%", f"{info['with_norm_pct']}%"])
        parts.append(_table(
            ["Lab", "N records", "Avec valeur", "% avec valeur", "% avec norme N<"],
            rows))
        for name, info in labs["results"].items():
            if info.get("examples"):
                parts.append(f"<h3>Exemples — {name}</h3>")
                for ex in info["examples"]:
                    parts.append(f'<div class="example">{html_escape(ex)}</div>')

    # 7. Profils patient
    if profiles:
        parts.append("<h2>7. Profils par patient</h2>")
        s = profiles["stats"]
        parts.append(_table(
            ["Métrique", "Min", "Médiane", "Moyenne", "Max"],
            [["Séjours/patient",
              s["stays_per_patient"]["min"], s["stays_per_patient"]["median"],
              s["stays_per_patient"]["mean"], s["stays_per_patient"]["max"]],
             ["Records/séjour",
              s["records_per_stay"]["min"], s["records_per_stay"]["median"],
              s["records_per_stay"]["mean"], s["records_per_stay"]["max"]]]))
        if "stays_by_gt" in s:
            parts.append("<h3>Séjours par GT</h3>")
            rows = [[k, v["n_patients"], v["mean_stays"], v["max_stays"]]
                    for k, v in s["stays_by_gt"].items()]
            parts.append(_table(["GT", "N patients", "Moy séjours", "Max séjours"], rows))
        parts.append(_img(profiles.get("plot"), "Profils"))

    # 8. Recommandations
    parts.append("<h2>8. Recommandations actionnables</h2>")
    if recommendations:
        parts.append('<ul class="recos">')
        for rec in recommendations:
            cls = "warning" if rec.startswith("⚠") else "success"
            parts.append(f'<li><div class="{cls}">{rec}</div></li>')
        parts.append("</ul>")

    return "\n".join(parts)


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="EDA noise sources pour PhenoRAG")
    ap.add_argument("--parquet", required=True, help="Chemin vers le parquet EHR")
    ap.add_argument("--gt", default=None, help="GT CSV (optionnel)")
    ap.add_argument("--output", default="reports/eda_noise.html")
    ap.add_argument("--top-libelle", type=int, default=25,
                    help="Nombre de top LIBELLE à afficher (défaut: 25)")
    args = ap.parse_args()

    print(f"\n  Chargement {args.parquet}...")
    df = pd.read_parquet(args.parquet)

    # Normalisation NIPATIENT en str pour join GT
    if "NIPATIENT" in df.columns:
        df["NIPATIENT"] = df["NIPATIENT"].astype(str).str.replace(r"\.0$", "", regex=True)

    print(f"    {len(df):,} records, {df['NIPATIENT'].nunique() if 'NIPATIENT' in df else '?'} patients")

    gt = load_gt_optional(args.gt)
    if gt:
        print(f"  GT chargé: {len(gt)} patients ({Counter(gt.values())})")

    print("\n  Analyse en cours...")
    overview = section_overview(df)
    print("    [1/7] Vue d'ensemble")
    typequest = section_typequest(df)
    print("    [2/7] TYPE_QUEST")
    top_lib = section_top_libelles(df, top_n=args.top_libelle)
    print("    [3/7] Top LIBELLE")
    pr_lex = section_pr_lexicon(df, gt)
    print("    [4/7] Lexique PR")
    noise = section_noise_contexts(df, gt)
    print("    [5/7] Confusions lexicales")
    labs = section_lab_quantification(df)
    print("    [6/7] Quantification labs")
    profiles = section_patient_profiles(df, gt)
    print("    [7/7] Profils patient")

    recs = section_recommendations(overview, typequest, top_lib, pr_lex, noise, labs)
    print(f"  → {len(recs)} recommandations générées")

    html = render_html(args.parquet, args.gt, overview, typequest, top_lib,
                        pr_lex, noise, labs, profiles, recs)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"\n  → {out}  ({out.stat().st_size // 1024} KB)")
    print(f"\n  Ouvre le fichier dans un navigateur pour voir le rapport.")


if __name__ == "__main__":
    main()
