#!/usr/bin/env python3
"""
generate_test_corpus.py

Generates a synthetic IOA corpus for testing PhenoRAG pipeline.
- 50 patients, ~150 stays (3 stays/patient avg)
- 30 RA+ / 20 RA- with ground truth
- Format: IOA LIBELLE/REPONSE pairs
- Realistic variability: missing data, partial evidence, borderline cases

Profiles:
  RA+ Classic (10):     RF+, anti-CCP+, DMARD, joints, CRP elevated
  RA+ Seronegative (5): RF-, CCP-, but DMARD + joints + CRP + PR confirmed
  RA+ Incomplete (10):  Some evidence missing, DMARD present, partial labs
  RA+ Borderline (5):   Score ~5-6, ambiguous
  RA- Clean (10):       No RA evidence at all (trauma, infection, etc.)
  RA- Mimics (5):       Some RA-like features but NOT RA (arthrose, lupus)
  RA- With noise (5):   Elevated CRP from infection, no RA
"""

import json
import random
from pathlib import Path
from datetime import datetime, timedelta

random.seed(42)

# ════════════════════════════════════════════════════════════════
# TEMPLATES for IOA records per profile
# ════════════════════════════════════════════════════════════════

def _date(base_year=2024, month_range=(1, 12)):
    lo = max(1, min(12, month_range[0]))
    hi = max(lo, min(12, month_range[1]))
    m = random.randint(lo, hi)
    d = random.randint(1, 28)
    return f"{base_year}-{m:02d}-{d:02d}T08:00:00"


def _ra_classic_stay(stay_id: str, visit_num: int, is_first: bool) -> dict:
    """RA+ classic: seropositive, DMARD, joints, inflammation."""
    rf_val = random.choice([45, 67, 87, 120, 230])
    ccp_val = random.choice([35, 55, 78, 150])
    crp_val = random.choice([15, 22, 34, 48, 65])
    mtx_dose = random.choice(["10 mg/semaine", "15 mg/semaine", "20 mg/semaine"])

    records = [
        {"LIBELLE": "Motif de consultation", "REPONSE": random.choice([
            "Suivi polyarthrite rhumatoide", "Controle PR sous traitement",
            "Consultation rhumatologie PR", "Poussee inflammatoire PR"])},
        {"LIBELLE": "Antecedents", "REPONSE": "Polyarthrite rhumatoide diagnostiquee depuis 2019"},
        {"LIBELLE": "Douleur articulaire", "REPONSE": random.choice([
            "Douleurs MCP et poignets bilateraux", "Raideur matinale > 1h, MCP 2-3-4 gonfles",
            "Douleur PIP et MTP, symetrique", "Synovites MCP bilaterales"])},
        {"LIBELLE": "Examen clinique", "REPONSE": random.choice([
            "Synovites MCP 2,3,4 bilaterales, poignets gonfles",
            "Tumefaction PIP 2,3 main droite, MCP 2,3,4 bilateraux",
            "Articulations douloureuses: 8 petites articulations"])},
        {"LIBELLE": "RF (Facteur Rhumatoide)", "REPONSE": f"Positif ({rf_val} UI/mL)"},
        {"LIBELLE": "Anti-CCP (ACPA)", "REPONSE": f"Positif ({ccp_val} U/mL)"},
        {"LIBELLE": "CRP", "REPONSE": f"{crp_val} mg/L"},
        {"LIBELLE": "VS", "REPONSE": f"{random.randint(25, 55)} mm/h"},
        {"LIBELLE": "Traitement en cours", "REPONSE": f"Methotrexate {mtx_dose}"},
    ]

    if random.random() > 0.5:
        records.append({"LIBELLE": "Duree des symptomes", "REPONSE": "Depuis plus de 6 mois"})
    if random.random() > 0.6:
        records.append({"LIBELLE": "Imagerie", "REPONSE": "Echographie: synovites actives grade 2-3"})

    return {"stay_id": stay_id, "visit_number": visit_num,
            "date": _date(2024, (1 + visit_num * 3, min(12, 3 + visit_num * 3))),
            "records": records}


def _ra_seroneg_stay(stay_id: str, visit_num: int, is_first: bool) -> dict:
    """RA+ seronegative: RF-, CCP-, but clinical + DMARD."""
    crp_val = random.choice([12, 18, 25, 32])
    records = [
        {"LIBELLE": "Motif", "REPONSE": "Suivi polyarthrite rhumatoide seronegative"},
        {"LIBELLE": "Antecedents", "REPONSE": "PR seronegative depuis 2020, sous MTX"},
        {"LIBELLE": "Douleur", "REPONSE": "Raideur matinale 45min, douleur poignets bilateraux"},
        {"LIBELLE": "Examen", "REPONSE": "Synovites poignets et MCP 2,3 bilateraux"},
        {"LIBELLE": "RF", "REPONSE": "Negatif"},
        {"LIBELLE": "Anti-CCP", "REPONSE": "Negatif"},
        {"LIBELLE": "CRP", "REPONSE": f"{crp_val} mg/L"},
        {"LIBELLE": "Traitement", "REPONSE": "Methotrexate 15 mg/semaine + acide folique"},
    ]
    if random.random() > 0.5:
        records.append({"LIBELLE": "Evolution", "REPONSE": "Suivi depuis 3 ans, maladie chronique"})
    return {"stay_id": stay_id, "visit_number": visit_num,
            "date": _date(2024, (1 + visit_num * 3, min(12, 3 + visit_num * 3))),
            "records": records}


def _ra_incomplete_stay(stay_id: str, visit_num: int, is_first: bool) -> dict:
    """RA+ with incomplete data: some dimensions missing."""
    records = [
        {"LIBELLE": "Motif", "REPONSE": random.choice([
            "Controle MTX", "Renouvellement ordonnance PR",
            "Consultation suivi rhumatologie"])},
        {"LIBELLE": "Traitement en cours", "REPONSE": random.choice([
            "Methotrexate 15 mg/semaine", "MTX 20mg + adalimumab",
            "Leflunomide 20mg/j"])},
    ]
    # Randomly add some but not all dimensions
    if random.random() > 0.4:
        records.append({"LIBELLE": "CRP", "REPONSE": f"{random.choice([8, 12, 18, 25])} mg/L"})
    if random.random() > 0.6:
        records.append({"LIBELLE": "Douleur", "REPONSE": "Douleur moderee poignets"})
    if random.random() > 0.7:
        records.append({"LIBELLE": "RF", "REPONSE": f"Positif ({random.randint(30, 100)} UI/mL)"})
    # No joints, no duration in many cases -> triggers feedback loop
    if random.random() > 0.8:
        records.append({"LIBELLE": "Diagnostic", "REPONSE": "Polyarthrite rhumatoide connue"})

    return {"stay_id": stay_id, "visit_number": visit_num,
            "date": _date(2024, (1 + visit_num * 3, min(12, 3 + visit_num * 3))),
            "records": records}


def _ra_borderline_stay(stay_id: str, visit_num: int, is_first: bool) -> dict:
    """RA+ borderline: score ~5-6, ambiguous."""
    records = [
        {"LIBELLE": "Motif", "REPONSE": "Bilan arthralgies inflammatoires"},
        {"LIBELLE": "Examen", "REPONSE": random.choice([
            "Douleur articulaire diffuse, pas de synovite franche",
            "Raideur matinale 30min, sensibilite MCP"])},
        {"LIBELLE": "CRP", "REPONSE": f"{random.choice([8, 11, 14])} mg/L"},
    ]
    if random.random() > 0.4:
        records.append({"LIBELLE": "RF", "REPONSE": f"Faiblement positif ({random.randint(22, 40)} UI/mL)"})
    if random.random() > 0.5:
        records.append({"LIBELLE": "Anti-CCP", "REPONSE": "Negatif"})
    if random.random() > 0.6:
        records.append({"LIBELLE": "Evolution", "REPONSE": "Symptomes depuis 2 mois"})

    return {"stay_id": stay_id, "visit_number": visit_num,
            "date": _date(2024, (1 + visit_num * 3, min(12, 3 + visit_num * 3))),
            "records": records}


def _ra_neg_clean_stay(stay_id: str, visit_num: int, is_first: bool) -> dict:
    """RA- clean: no RA evidence, other pathology."""
    motifs = [
        ("Chute traumatisme", "Traumatisme cheville droite suite a chute"),
        ("Infection urinaire", "Fievre 38.5, brulures mictionnelles"),
        ("Lombalgie mecanique", "Douleur lombaire mecanique, pas de syndrome inflammatoire"),
        ("Bilan preoperatoire", "Bilan avant chirurgie programmee"),
        ("Controle diabete", "Suivi diabete type 2, HbA1c a controler"),
        ("Pneumonie", "Toux productive, fievre, foyer pulmonaire droit"),
    ]
    motif, detail = random.choice(motifs)
    records = [
        {"LIBELLE": "Motif", "REPONSE": motif},
        {"LIBELLE": "Description", "REPONSE": detail},
        {"LIBELLE": "Examen clinique", "REPONSE": "Pas d'atteinte articulaire inflammatoire"},
    ]
    if random.random() > 0.5:
        records.append({"LIBELLE": "CRP", "REPONSE": f"{random.choice([2, 3, 5])} mg/L"})
    if random.random() > 0.7:
        records.append({"LIBELLE": "Biologie", "REPONSE": "Bilan biologique normal"})

    return {"stay_id": stay_id, "visit_number": visit_num,
            "date": _date(2024, (1 + visit_num * 3, min(12, 3 + visit_num * 3))),
            "records": records}


def _ra_neg_mimic_stay(stay_id: str, visit_num: int, is_first: bool) -> dict:
    """RA- mimic: looks like RA but isn't (arthrose, lupus, etc.)."""
    scenarios = [
        {"motif": "Arthrose digitale", "examen": "Nodules Heberden, pas de synovite",
         "diag": "Arthrose erosive des mains, pas de PR"},
        {"motif": "Lupus erythemateux", "examen": "Arthralgies diffuses, rash malaire",
         "diag": "Lupus systemique, anti-DNA positif"},
        {"motif": "Goutte polyarticulaire", "examen": "Gonflement MTP1 bilateral",
         "diag": "Goutte polyarticulaire, uricemie 95 mg/L"},
    ]
    s = random.choice(scenarios)
    records = [
        {"LIBELLE": "Motif", "REPONSE": s["motif"]},
        {"LIBELLE": "Examen", "REPONSE": s["examen"]},
        {"LIBELLE": "Diagnostic", "REPONSE": s["diag"]},
        {"LIBELLE": "RF", "REPONSE": "Negatif"},
        {"LIBELLE": "Anti-CCP", "REPONSE": "Negatif"},
    ]
    if random.random() > 0.5:
        records.append({"LIBELLE": "CRP", "REPONSE": f"{random.choice([5, 8, 12, 25])} mg/L"})

    return {"stay_id": stay_id, "visit_number": visit_num,
            "date": _date(2024, (1 + visit_num * 3, min(12, 3 + visit_num * 3))),
            "records": records}


def _ra_neg_noise_stay(stay_id: str, visit_num: int, is_first: bool) -> dict:
    """RA- with noise: elevated CRP from infection, not RA."""
    records = [
        {"LIBELLE": "Motif", "REPONSE": random.choice([
            "Fievre et arthralgies", "Bilan inflammatoire",
            "Douleurs articulaires post-infectieuses"])},
        {"LIBELLE": "CRP", "REPONSE": f"{random.choice([35, 55, 80, 120])} mg/L"},
        {"LIBELLE": "Diagnostic", "REPONSE": random.choice([
            "Arthrite reactionnelle post-infectieuse",
            "Syndrome grippal avec arthralgies",
            "Infection bacterienne, arthralgies secondaires"])},
        {"LIBELLE": "RF", "REPONSE": "Negatif"},
        {"LIBELLE": "Conclusion", "REPONSE": "Pas de polyarthrite rhumatoide"},
    ]
    return {"stay_id": stay_id, "visit_number": visit_num,
            "date": _date(2024, (1 + visit_num * 3, min(12, 3 + visit_num * 3))),
            "records": records}


# ════════════════════════════════════════════════════════════════
# MIXED STAY for RA+ patients (non-RA stay)
# ════════════════════════════════════════════════════════════════

def _non_ra_stay_for_ra_patient(stay_id: str, visit_num: int) -> dict:
    """A non-RA stay for an RA+ patient (trauma, infection, etc.)."""
    records = [
        {"LIBELLE": "Motif", "REPONSE": random.choice([
            "Chute avec fracture poignet", "Grippe saisonniere",
            "Bilan hepatique (surveillance MTX)", "Vaccination"])},
        {"LIBELLE": "Examen", "REPONSE": "Examen general, pas de poussee articulaire"},
    ]
    return {"stay_id": stay_id, "visit_number": visit_num,
            "date": _date(2024, (1 + visit_num * 3, min(12, 3 + visit_num * 3))),
            "records": records}


# ════════════════════════════════════════════════════════════════
# CORPUS GENERATOR
# ════════════════════════════════════════════════════════════════

def generate_corpus():
    patients = []
    patient_id_counter = 1

    profiles = [
        # (count, phenotype, stay_generator, label)
        (10, "RA+_classic",      _ra_classic_stay,      "RA+"),
        (5,  "RA+_seroneg",      _ra_seroneg_stay,      "RA+"),
        (10, "RA+_incomplete",   _ra_incomplete_stay,    "RA+"),
        (5,  "RA+_borderline",   _ra_borderline_stay,    "RA+"),
        (10, "RA-_clean",        _ra_neg_clean_stay,     "RA-"),
        (5,  "RA-_mimic",        _ra_neg_mimic_stay,     "RA-"),
        (5,  "RA-_noise",        _ra_neg_noise_stay,     "RA-"),
    ]

    total_stays = 0

    for count, profile, gen_fn, label in profiles:
        for _ in range(count):
            pid = f"P{patient_id_counter:03d}"
            patient_id_counter += 1

            # Number of stays: 2-4 per patient
            n_stays = random.choice([2, 3, 3, 4])
            stays = []

            for v in range(n_stays):
                sid = f"{pid}_S{v+1:02d}"

                # For RA+ patients: sometimes include a non-RA stay
                if label == "RA+" and v > 0 and random.random() < 0.2:
                    stay = _non_ra_stay_for_ra_patient(sid, v + 1)
                else:
                    stay = gen_fn(sid, v + 1, is_first=(v == 0))

                stays.append(stay)
                total_stays += 1

            patients.append({
                "patient_id": pid,
                "stays": stays,
                "ground_truth": {
                    "phenotype": label,
                    "profile": profile,
                    "n_stays": len(stays),
                }
            })

    # Shuffle patients
    random.shuffle(patients)

    return patients, total_stays


def main():
    output_dir = Path("data/test_corpus")
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus, total_stays = generate_corpus()

    # Save corpus
    corpus_path = output_dir / "ehr_ioa_test_50p.json"
    with open(corpus_path, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)

    # Save ground truth separately (for evaluation)
    gt = {p["patient_id"]: p["ground_truth"]["phenotype"] for p in corpus}
    gt_path = output_dir / "ground_truth_50p.json"
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(gt, f, indent=2, ensure_ascii=False)

    # Stats
    ra_pos = sum(1 for p in corpus if p["ground_truth"]["phenotype"] == "RA+")
    ra_neg = len(corpus) - ra_pos
    profiles = defaultdict(int)
    for p in corpus:
        profiles[p["ground_truth"]["profile"]] += 1

    print("=" * 60)
    print("TEST CORPUS GENERATED")
    print("=" * 60)
    print(f"Patients:    {len(corpus)}")
    print(f"Stays:       {total_stays}")
    print(f"RA+:         {ra_pos} ({100*ra_pos/len(corpus):.0f}%)")
    print(f"RA-:         {ra_neg} ({100*ra_neg/len(corpus):.0f}%)")
    print(f"\nProfiles:")
    for prof, cnt in sorted(profiles.items()):
        print(f"  {prof:<20s} {cnt}")
    print(f"\nCorpus:       {corpus_path}")
    print(f"Ground truth: {gt_path}")
    print("=" * 60)


# Need defaultdict for stats
from collections import defaultdict

if __name__ == "__main__":
    main()
