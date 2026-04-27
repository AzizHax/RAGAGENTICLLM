#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Générateur EHR IOA avec CHRONOLOGIE et ÉVOLUTION TEMPORELLE

Nouveautés:
- Dates réelles pour chaque séjour (espacés de semaines/mois)
- Évolution cohérente entre séjours (CRP baisse, symptômes évoluent)
- Références temporelles ("depuis dernier contrôle", "aggravation depuis M2")
- Compatible RAG inter-séjours (retrieval historique patient)

USAGE:
  python generate_ehr_ioa_chronology.py --patients 50 --model qwen2.5:3b-instruct --output ehr_chrono/
"""

import json
import random
import argparse
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import requests

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# =============================================================================
# QUESTIONNAIRES IOA
# =============================================================================

QUESTIONNAIRE_BI = [
    "Nom du patient",
    "Date de naissance",
    "Sexe",
    "Date consultation",
    "Antécédents médicaux",
    "Traitement habituel",
    "Allergies",
    "Motif de consultation",
    "Date début symptômes",
    "Douleurs actuelles",
    "Localisation douleurs",
    "Intensité douleur (0-10)",
]

QUESTIONNAIRE_BM_S = [
    "Date consultation",
    "Température",
    "Tension artérielle",
    "Poids",
    "Douleurs articulaires",
    "Raideur matinale",
    "Durée raideur matinale",
    "Articulations gonflées",
    "Articulations douloureuses",
    "Évolution depuis dernière visite",  # ← NOUVEAU
    "Traitement suivi",
    "Observance traitement",
    "Effets secondaires",
    "Biologie récente",
    "CRP",
    "VS",
    "Facteur rhumatoïde",
    "Anti-CCP",
    "Diagnostic retenu",
    "Traitement prescrit",
    "Date prochain contrôle",
]


# =============================================================================
# OLLAMA CLIENT (qwen2.5:3b-instruct)
# =============================================================================

class OllamaClient:
    def __init__(self, model: str = "qwen2.5:3b-instruct", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.calls = 0
        self.total_time = 0.0
        self._check_model()
    
    def _check_model(self):
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = [m["name"] for m in response.json().get("models", [])]
                if self.model in models:
                    print(f"✓ Modèle {self.model} disponible")
                else:
                    print(f"⚠️  Modèle {self.model} non trouvé. Disponibles: {', '.join(models)}")
        except Exception as e:
            print(f"⚠️  Erreur Ollama: {e}")
    
    def generate_response(self, question: str, clinical_context: Dict[str, Any]) -> str:
        """Génère réponse avec contexte temporel"""
        
        # Contexte temporel pour RAG
        temporal_context = ""
        if clinical_context.get("previous_stay"):
            prev = clinical_context["previous_stay"]
            temporal_context = f"""
Dernière visite (il y a {clinical_context.get('days_since_last', '?')} jours):
- CRP: {prev.get('crp', 'NC')} mg/L
- Traitement: {prev.get('treatment', 'NC')}
- Symptômes: {prev.get('symptoms', 'NC')}
"""
        
        prompt = f"""Tu es un infirmier IOA remplissant un questionnaire.

Contexte patient:
- Âge: {clinical_context.get('age')} ans
- Diagnostic: {clinical_context.get('diagnosis', 'à déterminer')}
- Visite n°{clinical_context.get('visit_number', 1)}
{temporal_context}

Question: {question}

Réponds de façon concise et cohérente avec l'historique (max 20 mots).
Si évolution temporelle pertinente, la mentionner.

Réponse:"""

        start = time.time()
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 60,
                    }
                },
                timeout=40
            )
            
            if response.status_code == 200:
                text = response.json().get("response", "").strip()
                self.calls += 1
                self.total_time += time.time() - start
                return text
            else:
                return "[Erreur]"
                
        except Exception as e:
            print(f"⚠️  Erreur LLM: {e}")
            return "[Erreur]"


# =============================================================================
# GÉNÉRATEUR AVEC ÉVOLUTION TEMPORELLE
# =============================================================================

@dataclass
class ClinicalState:
    """État clinique à un instant T (pour évolution temporelle)"""
    date: datetime
    crp: float
    vs: float
    pain_score: int  # 0-10
    stiffness_duration: int  # minutes
    swollen_joints: int
    treatment: str
    symptoms: str


@dataclass
class PatientProfile:
    patient_id: str
    phenotype: str
    age: int
    gender: str
    date_first_symptoms: datetime
    date_diagnosis: Optional[datetime]


@dataclass
class IOAStay:
    stay_id: str
    patient_id: str
    date: datetime
    visit_number: int
    visit_type: str
    questionnaire_type: str
    records: List[Dict[str, str]]
    clinical_state: ClinicalState  # État clinique pour RAG


class TemporalIOAGenerator:
    """Générateur avec chronologie et évolution"""
    
    def __init__(self, use_llm: bool = True, llm_model: str = "qwen2.5:3b-instruct", seed: int = 42):
        self.use_llm = use_llm
        self.seed = seed
        random.seed(seed)
        
        if use_llm:
            self.llm = OllamaClient(model=llm_model)
        else:
            self.llm = None
        
        self.stats = {
            "patients": 0,
            "stays": 0,
            "records": 0,
            "llm_calls": 0,
        }
    
    def generate_patient_profile(self, patient_id: str, phenotype: str) -> PatientProfile:
        """Génère profil avec dates réalistes"""
        
        age = random.randint(35, 70) if phenotype.startswith("RA") else random.randint(50, 80)
        gender = random.choice(["M", "F"])
        
        # Date premiers symptômes (6-24 mois avant aujourd'hui)
        months_ago = random.randint(6, 24)
        date_first_symptoms = datetime.now() - timedelta(days=months_ago * 30)
        
        # Date diagnostic (2-6 mois après symptômes pour RA)
        if phenotype.startswith("RA"):
            date_diagnosis = date_first_symptoms + timedelta(days=random.randint(60, 180))
        else:
            date_diagnosis = None
        
        return PatientProfile(
            patient_id=patient_id,
            phenotype=phenotype,
            age=age,
            gender=gender,
            date_first_symptoms=date_first_symptoms,
            date_diagnosis=date_diagnosis
        )
    
    def generate_initial_state(self, phenotype: str) -> ClinicalState:
        """État clinique initial"""
        
        if phenotype == "RA_progressive":
            return ClinicalState(
                date=datetime.now(),
                crp=random.uniform(15, 35),
                vs=random.uniform(25, 50),
                pain_score=random.randint(5, 8),
                stiffness_duration=random.randint(30, 90),
                swollen_joints=random.randint(2, 6),
                treatment="AINS",
                symptoms="douleurs MCPs bilatérales"
            )
        elif phenotype == "RA_stable":
            return ClinicalState(
                date=datetime.now(),
                crp=random.uniform(3, 8),
                vs=random.uniform(8, 20),
                pain_score=random.randint(1, 3),
                stiffness_duration=random.randint(10, 30),
                swollen_joints=0,
                treatment="Méthotrexate 20mg/sem",
                symptoms="rémission clinique"
            )
        else:  # mimicker
            return ClinicalState(
                date=datetime.now(),
                crp=random.uniform(5, 15),
                vs=random.uniform(10, 25),
                pain_score=random.randint(3, 6),
                stiffness_duration=0,
                swollen_joints=0,
                treatment="Paracétamol",
                symptoms="douleurs mécaniques genoux"
            )
    
    def evolve_clinical_state(
        self,
        previous: ClinicalState,
        phenotype: str,
        visit_type: str,
        days_elapsed: int
    ) -> ClinicalState:
        """Fait évoluer l'état clinique de façon cohérente"""
        
        new_date = previous.date + timedelta(days=days_elapsed)
        
        if phenotype == "RA_progressive":
            if visit_type == "diagnosis":
                # Diagnostic confirmé → début traitement de fond
                return ClinicalState(
                    date=new_date,
                    crp=previous.crp * random.uniform(0.9, 1.1),  # Stable/légère hausse
                    vs=previous.vs * random.uniform(0.95, 1.15),
                    pain_score=previous.pain_score,
                    stiffness_duration=previous.stiffness_duration + random.randint(-10, 20),
                    swollen_joints=previous.swollen_joints + random.randint(0, 2),
                    treatment="Méthotrexate 15mg/sem initié",
                    symptoms="synovite MCPs + PIPs persistante"
                )
            elif visit_type == "improvement":
                # Amélioration sous traitement
                return ClinicalState(
                    date=new_date,
                    crp=previous.crp * random.uniform(0.4, 0.7),  # Baisse CRP
                    vs=previous.vs * random.uniform(0.5, 0.8),
                    pain_score=max(1, previous.pain_score - random.randint(2, 4)),
                    stiffness_duration=max(10, int(previous.stiffness_duration * 0.5)),
                    swollen_joints=max(0, previous.swollen_joints - random.randint(1, 3)),
                    treatment="Méthotrexate 20mg/sem",
                    symptoms="amélioration nette sous DMARD"
                )
            else:  # followup stable
                return ClinicalState(
                    date=new_date,
                    crp=previous.crp * random.uniform(0.9, 1.1),
                    vs=previous.vs * random.uniform(0.9, 1.1),
                    pain_score=previous.pain_score + random.randint(-1, 1),
                    stiffness_duration=previous.stiffness_duration + random.randint(-15, 15),
                    swollen_joints=previous.swollen_joints,
                    treatment=previous.treatment,
                    symptoms="état stable"
                )
        
        elif phenotype == "RA_stable":
            # Patient stable, petites variations
            return ClinicalState(
                date=new_date,
                crp=previous.crp * random.uniform(0.8, 1.2),
                vs=previous.vs * random.uniform(0.9, 1.1),
                pain_score=max(0, min(10, previous.pain_score + random.randint(-1, 1))),
                stiffness_duration=max(0, previous.stiffness_duration + random.randint(-10, 10)),
                swollen_joints=0,
                treatment=previous.treatment,
                symptoms="rémission maintenue" if previous.pain_score <= 2 else "activité faible"
            )
        
        else:  # mimicker
            # Légère amélioration symptomatique
            return ClinicalState(
                date=new_date,
                crp=previous.crp * random.uniform(0.7, 1.0),
                vs=previous.vs * random.uniform(0.8, 1.0),
                pain_score=max(2, previous.pain_score - random.randint(1, 2)),
                stiffness_duration=0,
                swollen_joints=0,
                treatment="Antalgiques si besoin",
                symptoms="amélioration mécanique"
            )
    
    def generate_response_deterministic(
        self,
        question: str,
        clinical_state: ClinicalState,
        profile: PatientProfile,
        visit_number: int,
        previous_state: Optional[ClinicalState] = None
    ) -> str:
        """Génère réponse déterministe avec cohérence temporelle"""
        
        q = question.lower()
        
        # Date consultation
        if "date consultation" in q:
            return clinical_state.date.strftime("%d/%m/%Y")
        
        # Date naissance
        if "date de naissance" in q or "date naissance" in q:
            birth_date = datetime.now() - timedelta(days=profile.age * 365)
            return birth_date.strftime("%d/%m/%Y")
        
        # Sexe
        if "sexe" in q:
            return "Masculin" if profile.gender == "M" else "Féminin"
        
        # Antécédents
        if "antécédent" in q:
            if profile.phenotype.startswith("RA"):
                if profile.date_diagnosis:
                    years = (datetime.now() - profile.date_diagnosis).days // 365
                    return f"Polyarthrite rhumatoïde diagnostiquée il y a {years} ans"
                else:
                    return "Douleurs articulaires récentes à explorer"
            else:
                return "Arthrose, HTA"
        
        # Traitement habituel
        if "traitement" in q and "habituel" in q:
            return clinical_state.treatment
        
        # Motif consultation
        if "motif" in q:
            if visit_number == 1:
                return "Douleurs articulaires débutantes"
            else:
                return "Contrôle traitement de fond" if "stable" in clinical_state.symptoms else "Réévaluation symptômes"
        
        # Date début symptômes
        if "date début" in q:
            return profile.date_first_symptoms.strftime("%d/%m/%Y")
        
        # Douleurs
        if "douleur" in q and "articulaire" in q:
            if clinical_state.pain_score >= 5:
                return f"Oui, modérées à intenses ({clinical_state.pain_score}/10)"
            elif clinical_state.pain_score >= 3:
                return f"Oui, légères ({clinical_state.pain_score}/10)"
            else:
                return "Non ou minimes"
        
        # Intensité douleur
        if "intensité" in q:
            return f"{clinical_state.pain_score}/10"
        
        # Raideur matinale
        if "raideur" in q and "matinale" in q and "durée" not in q:
            return "Oui" if clinical_state.stiffness_duration > 15 else "Non"
        
        # Durée raideur
        if "durée" in q and "raideur" in q:
            if clinical_state.stiffness_duration == 0:
                return "Aucune"
            else:
                return f"{clinical_state.stiffness_duration} minutes"
        
        # Articulations gonflées
        if "gonfl" in q:
            if clinical_state.swollen_joints > 0:
                return f"Oui ({clinical_state.swollen_joints} articulations: MCPs, PIPs)"
            else:
                return "Non"
        
        # Évolution depuis dernière visite ← NOUVEAU
        if "évolution" in q and "dernière" in q:
            if not previous_state:
                return "Première consultation"
            
            days = (clinical_state.date - previous_state.date).days
            
            # Comparer états
            if clinical_state.crp < previous_state.crp * 0.7:
                return f"Amélioration nette (CRP {previous_state.crp:.1f} → {clinical_state.crp:.1f} mg/L)"
            elif clinical_state.crp > previous_state.crp * 1.3:
                return f"Aggravation biologique (CRP en hausse)"
            else:
                return f"Stable depuis {days} jours"
        
        # CRP
        if q == "crp":
            return f"{clinical_state.crp:.1f} mg/L"
        
        # VS
        if q == "vs":
            return f"{clinical_state.vs:.0f} mm/h"
        
        # Facteur rhumatoïde
        if "facteur" in q or "rf" in q:
            if profile.phenotype in ["RA_progressive", "RA_stable"]:
                return f"Positif ({random.randint(70, 150)} UI/mL)"
            else:
                return "Négatif"
        
        # Anti-CCP
        if "anti-ccp" in q or "anti ccp" in q:
            if profile.phenotype in ["RA_progressive", "RA_stable"]:
                return f"Positif ({random.randint(50, 120)} U/mL)"
            else:
                return "Négatif"
        
        # Diagnostic
        if "diagnostic" in q and "retenu" in q:
            if profile.phenotype.startswith("RA"):
                return "Polyarthrite rhumatoïde séropositive"
            elif "mimicker" in profile.phenotype:
                return "Arthrose avec poussée inflammatoire"
            else:
                return "Arthralgie mécanique"
        
        # Traitement prescrit
        if "traitement" in q and "prescrit" in q:
            return clinical_state.treatment
        
        # Date prochain contrôle
        if "prochain contrôle" in q:
            next_date = clinical_state.date + timedelta(days=random.randint(60, 120))
            return next_date.strftime("%d/%m/%Y")
        
        # Observance
        if "observance" in q:
            return random.choice(["Bonne", "Correcte", "Irrégulière"])
        
        # Température
        if "température" in q:
            return f"{round(random.uniform(36.5, 37.2), 1)}°C"
        
        # Tension
        if "tension" in q:
            return f"{random.randint(110, 140)}/{random.randint(70, 90)} mmHg"
        
        # Default
        return random.choice(["Non", "Aucun", "Normal", "NC"])
    
    def generate_stay(
        self,
        profile: PatientProfile,
        stay_number: int,
        clinical_state: ClinicalState,
        previous_state: Optional[ClinicalState] = None
    ) -> IOAStay:
        """Génère séjour avec évolution temporelle"""
        
        stay_id = f"{profile.patient_id}_S{stay_number:03d}"
        
        # Type questionnaire
        questionnaire_type = "BI" if stay_number == 1 else random.choice(["BI", "BM_S"])
        
        # Questions
        if questionnaire_type == "BI":
            questions = QUESTIONNAIRE_BI
        else:
            questions = QUESTIONNAIRE_BM_S
        
        # Contexte pour LLM
        days_since_last = (clinical_state.date - previous_state.date).days if previous_state else None
        
        clinical_context = {
            "age": profile.age,
            "diagnosis": "Polyarthrite rhumatoïde" if profile.phenotype.startswith("RA") else "Arthrose",
            "visit_number": stay_number,
            "previous_stay": {
                "crp": f"{previous_state.crp:.1f}" if previous_state else None,
                "treatment": previous_state.treatment if previous_state else None,
                "symptoms": previous_state.symptoms if previous_state else None,
            } if previous_state else None,
            "days_since_last": days_since_last
        }
        
        # Générer réponses
        records = []
        for question in questions:
            
            # Déterministe d'abord
            response = self.generate_response_deterministic(
                question, clinical_state, profile, stay_number, previous_state
            )
            
            # LLM pour certaines questions (si activé)
            if self.use_llm and self.llm and random.random() < 0.3:  # 30% LLM
                try:
                    llm_response = self.llm.generate_response(question, clinical_context)
                    if len(llm_response) > 5 and "[Erreur]" not in llm_response:
                        response = llm_response
                        self.stats["llm_calls"] += 1
                except:
                    pass  # Garder réponse déterministe
            
            records.append({
                "LIBELLE": question,
                "REPONSE": response
            })
            
            self.stats["records"] += 1
        
        visit_type = "initial" if stay_number == 1 else "followup"
        
        return IOAStay(
            stay_id=stay_id,
            patient_id=profile.patient_id,
            date=clinical_state.date,
            visit_number=stay_number,
            visit_type=visit_type,
            questionnaire_type=questionnaire_type,
            records=records,
            clinical_state=clinical_state
        )
    
    def generate_patient(
        self,
        patient_id: str,
        phenotype: str,
        n_stays: int = 4
    ) -> Dict[str, Any]:
        """Génère patient avec trajectoire chronologique complète"""
        
        profile = self.generate_patient_profile(patient_id, phenotype)
        
        # État clinique initial
        current_state = self.generate_initial_state(phenotype)
        current_state.date = profile.date_first_symptoms + timedelta(days=random.randint(7, 30))
        
        stays = []
        previous_state = None
        
        for i in range(n_stays):
            # Générer séjour
            stay = self.generate_stay(profile, i + 1, current_state, previous_state)
            stays.append(stay)
            self.stats["stays"] += 1
            
            # Préparer état suivant
            if i < n_stays - 1:
                # Délai entre séjours
                if phenotype == "RA_progressive":
                    days_elapsed = random.randint(30, 90)  # 1-3 mois
                    visit_type = "diagnosis" if i == 1 else "improvement" if i == 2 else "followup"
                elif phenotype == "RA_stable":
                    days_elapsed = random.randint(60, 120)  # 2-4 mois
                    visit_type = "followup"
                else:
                    days_elapsed = random.randint(45, 90)
                    visit_type = "followup"
                
                previous_state = current_state
                current_state = self.evolve_clinical_state(
                    previous_state, phenotype, visit_type, days_elapsed
                )
        
        self.stats["patients"] += 1
        
        return {
            "patient_id": patient_id,
            "profile": asdict(profile),
            "ground_truth": {
                "final_label": "RA+" if phenotype.startswith("RA") else "RA−",
                "phenotype": phenotype
            },
            "stays": [asdict(s) for s in stays],
            "trajectory_summary": {
                "n_stays": len(stays),
                "date_first": stays[0].date.isoformat(),
                "date_last": stays[-1].date.isoformat(),
                "duration_days": (stays[-1].date - stays[0].date).days,
                "crp_evolution": [s.clinical_state.crp for s in stays],
                "pain_evolution": [s.clinical_state.pain_score for s in stays],
            }
        }


# =============================================================================
# EXPORT
# =============================================================================

def export_to_parquet(patients: List[Dict], output_dir: Path):
    """Export Parquet avec colonnes temporelles"""
    
    if not HAS_PANDAS:
        print("⚠️  pandas non disponible")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_records = []
    for patient in patients:
        for stay in patient["stays"]:
            for record in stay["records"]:
                all_records.append({
                    "NIPATIENT": patient["patient_id"],
                    "NISEJOUR": stay["stay_id"],
                    "DATE_SEJOUR": stay["date"],
                    "VISIT_NUMBER": stay["visit_number"],
                    "TYPE_QUESTIONNAIRE": stay["questionnaire_type"],
                    "LIBELLE": record["LIBELLE"],
                    "REPONSE": record["REPONSE"],
                    "CRP_SNAPSHOT": stay["clinical_state"]["crp"],
                    "PAIN_SCORE": stay["clinical_state"]["pain_score"],
                    "PHENOTYPE": patient["ground_truth"]["phenotype"],
                    "LABEL": patient["ground_truth"]["final_label"],
                })
    
    df = pd.DataFrame(all_records)
    df["DATE_SEJOUR"] = pd.to_datetime(df["DATE_SEJOUR"])
    
    parquet_file = output_dir / "ehr_ioa_chronology.parquet"
    df.to_parquet(parquet_file, index=False)
    
    print(f"\n✓ Export Parquet: {parquet_file}")
    print(f"  Records: {len(all_records)}")
    print(f"  Période: {df['DATE_SEJOUR'].min()} → {df['DATE_SEJOUR'].max()}")


def export_to_json(patients: List[Dict], output_dir: Path):
    """Export JSON avec trajectoires"""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    json_file = output_dir / "ehr_ioa_chronology.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(patients, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"✓ Export JSON: {json_file}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Générateur EHR IOA avec chronologie")
    parser.add_argument("--patients", type=int, default=100, help="Nombre patients")
    parser.add_argument("--model", type=str, default="qwen2.5:3b-instruct", help="Modèle Ollama")
    parser.add_argument("--no-llm", action="store_true", help="Mode déterministe")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--output", type=str, default="ehr_chrono", help="Output")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("GÉNÉRATEUR EHR IOA AVEC CHRONOLOGIE")
    print("=" * 80)
    print(f"Patients: {args.patients}")
    print(f"LLM: {'Non' if args.no_llm else args.model}")
    print(f"Seed: {args.seed}\n")
    
    generator = TemporalIOAGenerator(
        use_llm=not args.no_llm,
        llm_model=args.model,
        seed=args.seed
    )
    
    distribution = {
        "RA_progressive": int(args.patients * 0.4),
        "RA_stable": int(args.patients * 0.2),
        "mimicker_arthrose": int(args.patients * 0.2),
        "other": int(args.patients * 0.2),
    }
    distribution["RA_progressive"] += args.patients - sum(distribution.values())
    
    patients = []
    patient_counter = 1
    
    start = time.time()
    
    for phenotype, count in distribution.items():
        print(f"\nGénération {phenotype} ({count} patients)...")
        
        for i in range(count):
            patient_id = f"P{patient_counter:03d}"
            n_stays = random.randint(3, 6)
            
            patient = generator.generate_patient(patient_id, phenotype, n_stays)
            patients.append(patient)
            
            patient_counter += 1
            
            if patient_counter % 10 == 0:
                print(f"  {patient_counter-1}/{args.patients} patients...")
    
    elapsed = time.time() - start
    
    output_dir = Path(args.output)
    export_to_json(patients, output_dir)
    export_to_parquet(patients, output_dir)
    
    print("\n" + "=" * 80)
    print("STATISTIQUES")
    print("=" * 80)
    print(f"Patients: {generator.stats['patients']}")
    print(f"Séjours: {generator.stats['stays']}")
    print(f"Records: {generator.stats['records']}")
    print(f"Appels LLM: {generator.stats['llm_calls']}")
    print(f"Temps: {elapsed:.1f}s ({elapsed/args.patients:.1f}s/patient)")
    
    print(f"\n✓ Corpus dans {output_dir}/")
    print("\nTEST RAG INTER-SÉJOURS:")
    print("  python -c \"import pandas as pd; df=pd.read_parquet('ehr_chrono/ehr_ioa_chronology.parquet'); print(df[df['NIPATIENT']=='P001'].sort_values('DATE_SEJOUR')[['DATE_SEJOUR','LIBELLE','REPONSE','CRP_SNAPSHOT']].head(20))\"")


if __name__ == "__main__":
    main()
