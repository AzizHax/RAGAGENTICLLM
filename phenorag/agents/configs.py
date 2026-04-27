"""
phenorag/agents/configs.py
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class Agent1Config:
    model: str = "qwen7b:latest"
    temperature: float = 0.1
    timeout_s: int = 180
    max_retries: int = 3
    max_workers: int = 1

    use_kb_guidance: bool = True
    use_inter_stay_rag: bool = True
    use_prefilter: bool = True
    use_regex_fallback: bool = True

    inter_stay_top_k: int = 2
    max_records_in_prompt: int = 20

    prompt_extraction: str = "agent1/extraction_lite"
    prompt_system: str = "agent1/system"

    kb_query: str = "rheumatoid arthritis RF anti-CCP methotrexate"
    query_focus: Optional[List[str]] = None

    # Parquet preprocessing
    input_format: str = "json"  # "json" | "parquet"
    parquet_patient_col: str = "NIPATIENT"
    parquet_stay_col: str = "NISEJOUR"
    parquet_label_col: str = "LIBELLE"
    parquet_response_col: str = "REPONSE"
    parquet_nda_col: str = "NDA"


@dataclass
class Agent2Config:
    model: str = "qwen7b:latest"
    temperature: float = 0.1
    timeout_s: int = 180

    acr_threshold: int = 5
    rf_uln: float = 20.0
    ccp_uln: float = 10.0
    crp_threshold: float = 10.0
    esr_threshold: float = 20.0

    prompt_ra_relatedness: str = "agent2/ra_relatedness"
    prompt_acr_scoring: str = "agent2/acr_scoring"
    variant_name: str = "baseline"


@dataclass
class Agent3Config:
    model: str = "qwen7b:latest"
    temperature: float = 0.1
    timeout_s: int = 180

    # PATCH: synchronisé avec --gt-patient-agg (any_positive | majority | all_positive | confirmed)
    aggregation_strategy: str = "any_positive"

    use_critic: bool = True
    critic_only_on_uncertain: bool = True
    critic_confidence_threshold: float = 0.80
    allow_override: bool = True

    prompt_critic: str = "agent3/critic"


@dataclass
class PipelineConfig:
    architecture: str = "B1"
    b3_fast_max_records: int = 3
    b3_complex_min_missing: int = 3
    b4_models: Optional[List[str]] = None
    b4_vote_strategy: str = "majority"
    input_format: str = "json"  # "json" | "parquet"
    preprocess_output: Optional[str] = None
    # PATCH: flags RAG (axe D du plan.yaml)
    use_inter_stay_rag: bool = True
    # PATCH: stratégie d'agrégation patient-level (axe E) — transmise à Agent3
    gt_patient_agg: str = "any_positive"
