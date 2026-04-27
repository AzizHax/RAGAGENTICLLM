# PhenoRAG

**Phénotypage automatique de la Polyarthrite Rhumatoïde par une architecture Multi-Agents LLM + RAG**

PhenoRAG est un système multi-agents qui identifie automatiquement les patients atteints de polyarthrite rhumatoïde (PR) à partir de dossiers patients informatisés (EHR), en combinant des modèles de langage locaux (LLM), la génération augmentée par récupération (RAG) et un classifieur bayésien naïf multi-classes.

---

## Architecture

```
Dossier Patient (Parquet/JSON)
        │
   ┌────▼────┐
   │ Agent 1 │  Extraction hybride (LLM + Regex + RAG-KB)
   │Extracteur│  → disease_mentions, labs, drugs
   └────┬────┘
        │
   ┌────▼────┐
   │ Agent 2 │  Score ACR/EULAR 2010 (0-10)
   │Raisonneur│  + Classifieur Bayésien Naïf (5 classes)
   └────┬────┘
        │
   ┌────▼────┐
   │ Agent 3 │  Agrégation patient + Bayesian update temporel
   │Agrégateur│  → RA+ / RA- + P(PR+) + tendance
   └─────────┘
```

### 4 architectures disponibles

| Architecture | Description |
|---|---|
| **B1** Sequential | Pipeline linéaire Agent 1 → Agent 2 → Agent 3 |
| **B2** Hierarchical | Boucle de rétroaction Agent 1 ↔ Agent 2 + Critic LLM |
| **B3** Adaptive | Routage par complexité (Fast / Standard / Full + Critic) |
| **B4** Consensus | 3 modèles LLM votent indépendamment sur le scoring |

### Couche probabiliste

Au-delà du label binaire (RA+/RA-), le système produit une distribution sur 5 classes cliniques :

| Classe | Description |
|---|---|
| PR_ABSENT | Pas de PR |
| PR_LATENT | Suspectée, critères insuffisants |
| PR_REMISSION | PR connue sous traitement, faible activité |
| PR_MODERATE | PR active modérée (DAS28 3.2-5.1) |
| PR_SEVERE | PR active sévère (DAS28 > 5.1) |

---

## Installation

### Prérequis

- Python 3.10+
- [Ollama](https://ollama.ai/) avec au moins un modèle installé

### Setup

```bash
git clone https://github.com/votre-repo/phenorag.git
cd phenorag

pip install -r requirements.txt

# Installer un modèle LLM local
ollama pull qwen2.5:3b-instruct
```

### Structure du projet

```
ProjetPR/
├── run.py                          # Point d'entrée principal
├── requirements.txt
├── phenorag/
│   ├── __init__.py
│   ├── preprocess.py               # Parquet → JSON
│   ├── monitor_server.py           # Dashboard live WebSocket
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── agent1.py               # Extraction (LLM + Regex + RAG)
│   │   ├── agent2.py               # Scoring ACR/EULAR + Probabiliste
│   │   ├── agent3.py               # Agrégation patient + Bayesian
│   │   ├── orchestrator.py         # LangGraph B1/B2/B3/B4
│   │   ├── probabilistic.py        # Classifieur Bayésien Naïf
│   │   ├── configs.py              # Configurations
│   │   └── interfaces.py           # Dataclasses partagées
│   └── utils/
│       ├── __init__.py
│       ├── llm_client.py           # Client Ollama
│       └── prompt_loader.py        # Chargeur de prompts
├── prompts/
│   ├── agent1/
│   │   ├── system.txt
│   │   └── extraction_lite.txt
│   ├── agent2/
│   │   ├── ra_relatedness.txt
│   │   └── acr_scoring.txt
│   └── agent3/
│       └── critic.txt
├── data/
│   ├── kb_pr_phenotype.json        # Base de connaissances (optionnel)
│   └── test_corpus/
│       ├── ehr_ioa_test_50p.json   # Corpus de test synthétique
│       └── ground_truth_50p.json   # Annotations gold standard
├── visualize_probabilistic.py      # Plots pour le mémoire
└── phenorag_monitor.html           # Dashboard live (navigateur)
```

---

## Utilisation

### Commandes de base

```bash
# Pipeline séquentiel sur corpus JSON
python run.py --arch B1

# Pipeline hiérarchique avec feedback loop
python run.py --arch B2

# Réutiliser l'extraction existante (skip Agent 1)
python run.py --arch B2 --skip-extraction

# Pipeline sur données Parquet (auto-conversion)
python run.py --arch B2 --corpus data/real/ehr.parquet

# Sans base de connaissances
python run.py --arch B1 --no-kb

# Évaluer un run existant
python run.py --only eval --run-dir runs/b2

# Préprocessing seul (Parquet → JSON)
python run.py --only preprocess --corpus data/real/ehr.parquet
```

### Options complètes

```
--arch {B1,B2,B3,B4}     Architecture multi-agents
--model MODEL             Modèle Ollama (défaut: qwen2.5:3b-instruct)
--corpus PATH             Chemin du corpus (.json, .jsonl, .parquet)
--format {auto,json,parquet}  Format d'entrée (auto-détecté par extension)
--no-kb                   Désactiver la base de connaissances
--skip-extraction         Réutiliser facts.jsonl existant
--only {eval,preprocess}  Exécuter uniquement l'évaluation ou le préprocessing
--run-dir DIR             Dossier de sortie
--gt PATH                 Ground truth pour l'évaluation
--ollama-url URL          URL du serveur Ollama
--patient-col COL         Colonne patient dans le Parquet (défaut: NIPATIENT)
--stay-col COL            Colonne séjour dans le Parquet (défaut: NISEJOUR)
```

### Format des données d'entrée

**JSON** (format par défaut) :
```json
[
  {
    "patient_id": "P001",
    "stays": [
      {
        "stay_id": "P001_S01",
        "date": "2024-01-15",
        "visit_number": 1,
        "records": [
          {"LIBELLE": "Motif", "REPONSE": "Douleurs articulaires"},
          {"LIBELLE": "Biologie", "REPONSE": "CRP: 32 mg/L"}
        ]
      }
    ]
  }
]
```

**Parquet** (format hospitalier) :
| NIPATIENT | NISEJOUR | LIBELLE | REPONSE |
|---|---|---|---|
| PAT001 | SEJ001 | Motif | Douleurs articulaires |
| PAT001 | SEJ001 | Biologie | CRP: 32 mg/L |

Le Parquet est automatiquement converti en JSON par `phenorag/preprocess.py`.

---

## Monitoring live

Visualiser les agents en temps réel pendant l'exécution :

```bash
# Terminal 1 : lancer le serveur monitor
python -m phenorag.monitor_server --arch B2 --corpus data/test.json --no-kb

# Terminal 2 : ouvrir le dashboard dans le navigateur
# → Ouvrir phenorag_monitor.html
# → Connecter à ws://localhost:8765
```

---

## Visualisation des résultats

```bash
# Générer les plots pour le mémoire
python visualize_probabilistic.py \
    --run-dir runs/b2 \
    --gt data/test_corpus/ground_truth_50p.json

# Comparer les 4 architectures
python visualize_probabilistic.py \
    --run-dir runs/b2 \
    --compare runs/b1 runs/b2 runs/b3 runs/b4
```

Plots générés :
- Distribution des 5 classes PR
- Histogramme P(PR+) par label réel
- Courbe de calibration (Brier Score)
- Confusion matrices rule-based vs probabiliste
- Trajectoires temporelles par patient
- Tableau métriques à différents seuils

---

## Résultats

### Performances (corpus synthétique, 50 patients)

| Méthode | F1 | Precision | Recall | FP | FN |
|---|---|---|---|---|---|
| Rule-based | 0.915 | 0.931 | 0.900 | 2 | 3 |
| **Probabiliste (P≥0.6)** | **0.929** | **1.000** | **0.867** | **0** | **4** |

### Métriques clés

- **Brier Score** : 0.0786 (calibration satisfaisante)
- **Séparation bimodale** : 96% des patients à P(PR+) > 0.8 ou P(PR+) < 0.3
- **0 faux positifs** au seuil P ≥ 0.6
- **Détection PR en rémission** : patients sous DMARD correctement identifiés comme PR+

---

## Modèle probabiliste

Le classifieur Bayésien Naïf utilise 8 variables d'entrée :
- 4 dimensions ACR/EULAR (articulations, sérologie, inflammation, durée)
- 4 signaux cliniques (DMARD, biologique, mention PR, exclusion PR)

Les vraisemblances sont calibrées sur la littérature :
- Nishimura et al. 2007 (sensibilité/spécificité RF et anti-CCP)
- Motta et al. 2023 (RF isotypes, méta-analyse)
- Orr et al. 2018 (CRP/ESR et activité de la maladie)
- Greenmyer et al. 2020 (seuils DAS28-CRP)

Documentation complète : [`prior_documentation.md`](prior_documentation.md)

---

## Configuration serveur

### Windows (développement)

```bash
ollama pull qwen2.5:3b-instruct
python run.py --arch B1 --model qwen2.5:3b-instruct
```

### Debian (production)

```bash
# Ollama sur port 11436
ollama pull qwen7b:latest
ollama pull mistral_7b:latest
ollama pull llama3_1_8b_gguf:latest

python run.py --arch B4 --ollama-url http://localhost:11436 --model qwen7b:latest
```

---

## Citation

```bibtex
@mastersthesis{phenorag2026,
    title={Phénotypage automatique de la Polyarthrite Rhumatoïde
           par une architecture Multi-Agents LLM + RAG},
    author={Aziz Ben Ammar, Thibaut Fabacher},
    year={2026},
    school={Hôpitaux universitaires de Strasbourg}
}
```

## Licence

Ce projet est développé dans le cadre d'un mémoire de recherche.
