#!/usr/bin/env python3
"""
Script de génération de cas médical via OpenWebUI API
Utilise l'endpoint OpenWebUI pour générer du contenu avec le modèle GPT-OSS
"""

import requests
import json
from datetime import datetime
import argparse
import os
import sys
from dotenv import load_dotenv

# Charger les variables d'environnement depuis .env
load_dotenv()

# Configuration de l'API OpenWebUI depuis .env
OPENWEBUI_BASE_URL = os.getenv("OPENWEBUI_BASE_URL", "https://med.ia.unistra.fr")
MODEL_NAME = os.getenv("MODEL_NAME", "GPT-OSS")
DEFAULT_API_KEY = os.getenv("OPENWEBUI_API_KEY", "")
DEFAULT_OUTPUT_DIR = os.getenv("OUTPUT_DIR", "generated_cases")
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
DEFAULT_MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2000"))

# Prompt personnalisé pour générer des cas médicaux
DEFAULT_PROMPT = os.getenv("DEFAULT_PROMPT", """Génère un cas clinique médical détaillé comprenant :
- Informations du patient (âge, sexe, antécédents)
- Motif de consultation
- Symptômes et signes cliniques
- Examens complémentaires
- Diagnostic différentiel
- Diagnostic final
- Traitement proposé

Le cas doit être réaliste et pédagogique.""")


def generate_case(api_key: str, prompt: str = DEFAULT_PROMPT, case_number: int = 1, 
                  temperature: float = DEFAULT_TEMPERATURE, max_tokens: int = DEFAULT_MAX_TOKENS) -> dict:
    """
    Génère un cas médical en appelant l'API OpenWebUI
    
    Args:
        api_key: Clé API pour l'authentification
        prompt: Prompt personnalisé pour la génération
        case_number: Numéro du cas (pour les métadonnées)
        temperature: Température de génération
        max_tokens: Nombre maximum de tokens
    
    Returns:
        dict: Réponse complète avec métadonnées
    """
    
    # Headers pour l'authentification
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Corps de la requête (format compatible OpenAI/OpenWebUI)
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False
    }
    
    # Liste des endpoints possibles pour OpenWebUI
    # OpenWebUI peut utiliser différents chemins selon la version
    possible_endpoints = [
        f"{OPENWEBUI_BASE_URL}/api/chat/completions",  # Endpoint standard OpenWebUI
        f"{OPENWEBUI_BASE_URL}/api/chat",              # Endpoint alternatif
        f"{OPENWEBUI_BASE_URL}/api/v1/chat/completions"  # Style OpenAI
    ]
    
    last_error = None
    
    for url in possible_endpoints:
        try:
            print(f"📡 Test de l'endpoint: {url}")
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            
            # Si succès, on utilise cet endpoint
            if response.status_code == 200:
                print(f"✅ Endpoint fonctionnel: {url}")
                api_response = response.json()
                
                # Extraction du texte généré
                generated_text = api_response.get("choices", [{}])[0].get("message", {}).get("content", "")
                
                # Construction de la réponse complète avec métadonnées
                result = {
                    "metadata": {
                        "case_number": case_number,
                        "timestamp": datetime.now().isoformat(),
                        "model": MODEL_NAME,
                        "endpoint": url,
                        "base_url": OPENWEBUI_BASE_URL,
                        "prompt": prompt,
                        "generation_params": {
                            "temperature": temperature,
                            "max_tokens": max_tokens
                        }
                    },
                    "api_response": {
                        "id": api_response.get("id"),
                        "created": api_response.get("created"),
                        "model": api_response.get("model"),
                        "usage": api_response.get("usage", {})
                    },
                    "generated_content": generated_text
                }
                
                print(f"✅ Cas généré avec succès!")
                return result
            else:
                print(f"⚠️  Code {response.status_code} pour {url}")
                last_error = response
                
        except requests.exceptions.RequestException as e:
            print(f"⚠️  Erreur pour {url}: {e}")
            last_error = e
            continue
    
    # Si aucun endpoint ne fonctionne
    print(f"\n❌ Aucun endpoint ne fonctionne!")
    if last_error:
        if isinstance(last_error, requests.Response):
            print(f"Dernier code HTTP: {last_error.status_code}")
            print(f"Réponse: {last_error.text[:500]}")
        else:
            print(f"Dernière erreur: {last_error}")
    
    print("\n💡 Conseils de dépannage:")
    print("1. Vérifiez que votre clé API est valide")
    print("2. Vérifiez que le modèle 'GPT-OSS' existe dans votre instance")
    print("3. Consultez la documentation de votre OpenWebUI pour l'endpoint exact")
    
    sys.exit(1)


def save_to_json(data: dict, output_dir: str = "generated_cases") -> str:
    """
    Sauvegarde les données dans un fichier JSON
    
    Args:
        data: Dictionnaire contenant les données à sauvegarder
        output_dir: Répertoire de destination
    
    Returns:
        str: Chemin du fichier créé
    """
    # Créer le répertoire si nécessaire
    os.makedirs(output_dir, exist_ok=True)
    
    # Nom du fichier avec timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    case_num = data["metadata"]["case_number"]
    filename = f"case_{case_num:04d}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Sauvegarder avec indentation pour lisibilité
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Cas sauvegardé dans: {filepath}")
    return filepath


def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(
        description="Génère un cas médical via l'API OpenWebUI"
    )
    parser.add_argument(
        "--api-key",
        default=DEFAULT_API_KEY,
        help="Clé API OpenWebUI (ou définir OPENWEBUI_API_KEY dans .env)"
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt personnalisé pour la génération"
    )
    parser.add_argument(
        "--case-number",
        type=int,
        default=1,
        help="Numéro du cas (pour tracking)"
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Répertoire de sortie pour les fichiers JSON"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Température de génération (0.0-2.0)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Nombre maximum de tokens"
    )
    
    args = parser.parse_args()
    
    # Vérifier que la clé API est fournie
    if not args.api_key:
        print("❌ Erreur: Clé API requise!")
        print("\nDéfinissez-la dans .env ou utilisez --api-key")
        print("\nExemple .env:")
        print("OPENWEBUI_API_KEY=sk-votre-cle-ici")
        sys.exit(1)
    
    print("=" * 60)
    print("🏥 Générateur de cas médical - OpenWebUI")
    print("=" * 60)
    print(f"Modèle: {MODEL_NAME}")
    print(f"Base URL: {OPENWEBUI_BASE_URL}")
    print(f"Cas numéro: {args.case_number}")
    print(f"Température: {args.temperature}")
    print(f"Max tokens: {args.max_tokens}")
    print("-" * 60)
    
    # Générer le cas
    case_data = generate_case(
        api_key=args.api_key,
        prompt=args.prompt,
        case_number=args.case_number,
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )
    
    # Sauvegarder dans un fichier JSON
    filepath = save_to_json(case_data, args.output_dir)
    
    print("-" * 60)
    print(f"✨ Génération terminée!")
    print(f"📄 Fichier: {filepath}")
    print("=" * 60)


if __name__ == "__main__":
    main()