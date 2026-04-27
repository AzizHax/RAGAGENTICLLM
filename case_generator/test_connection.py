#!/usr/bin/env python3
"""
Script de test de connexion à l'API OpenWebUI
Vérifie que l'endpoint et la clé API fonctionnent correctement
"""

import requests
import sys
import getpass
import os
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Configuration depuis .env
DEFAULT_BASE_URL = os.getenv("OPENWEBUI_BASE_URL", "https://med.ia.unistra.fr")
DEFAULT_MODEL = os.getenv("MODEL_NAME", "GPT-OSS")
DEFAULT_API_KEY = os.getenv("OPENWEBUI_API_KEY", "")


def test_connection(api_key: str, base_url: str = DEFAULT_BASE_URL, model: str = DEFAULT_MODEL):
    """
    Teste la connexion à l'API OpenWebUI
    
    Args:
        api_key: Clé API
        base_url: URL de base de l'API
        model: Nom du modèle
    
    Returns:
        bool: True si connexion réussie
    """
    print("=" * 60)
    print("🧪 TEST DE CONNEXION - OpenWebUI API")
    print("=" * 60)
    print(f"Base URL: {base_url}")
    print(f"Modèle: {model}")
    print("-" * 60)
    
    # Test 1: Vérifier l'endpoint de base
    print("\n[1/3] Test de l'endpoint de base...")
    try:
        response = requests.get(base_url, timeout=10)
        print(f"✅ Endpoint accessible (Status: {response.status_code})")
    except requests.exceptions.RequestException as e:
        print(f"❌ Impossible d'accéder à l'endpoint: {e}")
        return False
    
    # Test 2: Tester l'authentification avec différents endpoints
    print("\n[2/3] Test d'authentification sur différents endpoints...")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Payload de test
    test_payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Test de connexion"}],
        "max_tokens": 10,
        "stream": False
    }
    
    # Liste des endpoints possibles
    endpoints = [
        f"{base_url}/api/chat/completions",
        f"{base_url}/api/chat",
        f"{base_url}/api/v1/chat/completions"
    ]
    
    successful_endpoint = None
    
    for url in endpoints:
        try:
            print(f"\n   Essai: {url}")
            response = requests.post(url, headers=headers, json=test_payload, timeout=30)
            
            if response.status_code == 200:
                print(f"   ✅ Endpoint fonctionnel!")
                successful_endpoint = url
                data = response.json()
                break
            else:
                print(f"   ⚠️  Status {response.status_code}: {response.text[:100]}")
                
        except requests.exceptions.RequestException as e:
            print(f"   ⚠️  Erreur: {str(e)[:100]}")
            continue
    
    if not successful_endpoint:
        print("\n❌ Aucun endpoint ne fonctionne!")
        return False
    
    # Test 3: Vérifier la réponse
    print("\n[3/3] Test de génération...")
    try:
        if "choices" in data and len(data["choices"]) > 0:
            content = data["choices"][0].get("message", {}).get("content", "")
            print("✅ Génération réussie!")
            print(f"\n📝 Réponse test: {content[:100]}...")
            
            # Afficher les informations de l'API
            if "usage" in data:
                print(f"\n📊 Utilisation:")
                print(f"   - Tokens prompt: {data['usage'].get('prompt_tokens', 'N/A')}")
                print(f"   - Tokens completion: {data['usage'].get('completion_tokens', 'N/A')}")
                print(f"   - Tokens total: {data['usage'].get('total_tokens', 'N/A')}")
            
            # Sauvegarder l'endpoint fonctionnel
            print(f"\n💡 Endpoint à utiliser: {successful_endpoint}")
            
            return True
        else:
            print("⚠️  Réponse inattendue de l'API")
            print(f"Réponse: {data}")
            return False
            
    except Exception as e:
        print(f"❌ Erreur lors de l'analyse de la réponse: {e}")
        return False


def main():
    """Fonction principale"""
    print("\n🔍 Vérification de la configuration OpenWebUI\n")
    
    # Vérifier si .env existe
    if os.path.exists(".env"):
        print("✅ Fichier .env détecté")
        print(f"   Base URL: {DEFAULT_BASE_URL}")
        print(f"   Modèle: {DEFAULT_MODEL}")
        if DEFAULT_API_KEY:
            print(f"   API Key: {'*' * 20}{DEFAULT_API_KEY[-4:]}")
    else:
        print("⚠️  Aucun fichier .env détecté")
        print("   Vous pouvez créer un .env à partir de .env.example\n")
    
    # Demander la clé API
    if DEFAULT_API_KEY:
        print("\n🔑 Clé API détectée dans .env")
        use_env = input("Utiliser cette clé pour le test ? (o/n, défaut: o): ").lower()
        if use_env != 'n':
            api_key = DEFAULT_API_KEY
        else:
            api_key = getpass.getpass("Entrez votre clé API OpenWebUI: ")
    else:
        api_key = getpass.getpass("\n🔑 Entrez votre clé API OpenWebUI: ")
    
    if not api_key:
        print("❌ Clé API requise!")
        sys.exit(1)
    
    # Tester la connexion
    success = test_connection(api_key, DEFAULT_BASE_URL, DEFAULT_MODEL)
    
    # Résultat final
    print("\n" + "=" * 60)
    if success:
        print("✅ TOUS LES TESTS RÉUSSIS!")
        print("=" * 60)
        print("\n✨ Votre configuration est prête!")
        print("   Vous pouvez maintenant utiliser:")
        print("   - python generate_case.py")
        print("   - python batch_generator.py")
        print("\n💡 Conseil: Créez un fichier .env pour ne pas avoir à")
        print("   saisir la clé API à chaque fois")
    else:
        print("❌ TESTS ÉCHOUÉS")
        print("=" * 60)
        print("\n⚠️  Vérifiez votre configuration:")
        print("   1. La clé API est valide")
        print("   2. L'URL de base est correcte dans .env")
        print(f"      OPENWEBUI_BASE_URL={DEFAULT_BASE_URL}")
        print(f"   3. Le modèle '{DEFAULT_MODEL}' existe dans votre instance")
        print("   4. Vous avez accès réseau à l'endpoint")
        sys.exit(1)
    
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Test annulé")
        sys.exit(1)