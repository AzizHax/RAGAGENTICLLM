#!/usr/bin/env python3
"""
Script batch pour générer plusieurs cas médicaux
Lance le générateur de cas en boucle selon le nombre demandé
"""

import subprocess
import sys
import os
import getpass
from datetime import datetime
import time
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Variables de configuration
DEFAULT_API_KEY = os.getenv("OPENWEBUI_API_KEY", "")
DEFAULT_OUTPUT_DIR = os.getenv("OUTPUT_DIR", "generated_cases")
DEFAULT_PROMPT = os.getenv("DEFAULT_PROMPT", "")


def get_api_key() -> str:
    """
    Demande la clé API à l'utilisateur ou utilise celle du .env
    Fallback sur input() si getpass échoue selon l'environnement.
    """
    if DEFAULT_API_KEY:
        print("🔑 Clé API détectée dans .env")
        use_env = input("Utiliser cette clé ? (o/n, défaut: o): ").strip().lower()
        if use_env != 'n':
            return DEFAULT_API_KEY

    print("\n🔑 Authentification OpenWebUI")
    print("-" * 60)

    try:
        api_key = getpass.getpass("Entrez votre clé API OpenWebUI (saisie masquée): ").strip()
    except (Exception, KeyboardInterrupt):
        # Certains environnements cassent getpass -> fallback
        print("\n⚠️  Saisie masquée indisponible ici. Utilisation de la saisie visible.")
        api_key = input("Entrez votre clé API OpenWebUI: ").strip()

    if not api_key:
        print("❌ Clé API requise!")
        sys.exit(1)

    return api_key



def get_number_of_cases() -> int:
    """
    Demande le nombre de cas à générer
    
    Returns:
        int: Nombre de cas à générer
    """
    while True:
        try:
            print("\n📊 Configuration de la génération")
            print("-" * 60)
            num_cases = int(input("Combien de cas voulez-vous générer ? "))
            
            if num_cases < 1:
                print("⚠️  Le nombre doit être au moins 1")
                continue
            
            if num_cases > 100:
                confirm = input(f"⚠️  Vous allez générer {num_cases} cas. Continuer ? (o/n): ")
                if confirm.lower() != 'o':
                    continue
            
            return num_cases
            
        except ValueError:
            print("⚠️  Veuillez entrer un nombre valide")
        except KeyboardInterrupt:
            print("\n\n❌ Opération annulée")
            sys.exit(0)


def get_custom_prompt() -> str:
    """
    Demande si l'utilisateur veut un prompt personnalisé
    
    Returns:
        str: Prompt personnalisé ou chaîne vide pour utiliser le défaut du .env
    """
    print("\n✏️  Prompt de génération")
    print("-" * 60)
    
    if DEFAULT_PROMPT:
        print("Un prompt par défaut est défini dans .env")
        print("Voulez-vous le modifier pour cette session ?")
        modify = input("o/n (défaut: n): ").lower()
        if modify != 'o':
            return ""
    
    print("\nVoulez-vous utiliser un prompt personnalisé ?")
    print("(Appuyez sur Entrée pour utiliser le prompt par défaut)")
    
    use_custom = input("o/n (défaut: n): ").lower()
    
    if use_custom == 'o':
        print("\nEntrez votre prompt (terminez avec une ligne vide):")
        lines = []
        while True:
            line = input()
            if not line:
                break
            lines.append(line)
        
        return "\n".join(lines)
    
    return ""


def run_generator(api_key: str, case_number: int, prompt: str = "", output_dir: str = None) -> bool:
    """
    Lance le script de génération pour un cas
    
    Args:
        api_key: Clé API
        case_number: Numéro du cas
        prompt: Prompt personnalisé (optionnel)
        output_dir: Répertoire de sortie
    
    Returns:
        bool: True si succès, False sinon
    """
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    
    cmd = [
        sys.executable,  # Utilise le même interpréteur Python
        "generate_case.py",
        "--case-number", str(case_number),
        "--output-dir", output_dir
    ]
    
    # Ajouter l'API key seulement si elle n'est pas déjà dans .env
    if api_key and api_key != DEFAULT_API_KEY:
        cmd.extend(["--api-key", api_key])
    
    if prompt:
        cmd.extend(["--prompt", prompt])
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Erreur lors de la génération du cas {case_number}")
        return False
    except FileNotFoundError:
        print("\n❌ Erreur: Le fichier 'generate_case.py' est introuvable!")
        print("Assurez-vous qu'il est dans le même répertoire que ce script.")
        return False


def main():
    """Fonction principale"""
    print("=" * 60)
    print("🏥 GÉNÉRATEUR BATCH DE CAS MÉDICAUX")
    print("   OpenWebUI - GPT-OSS")
    print("=" * 60)
    
    # Récupérer les paramètres
    api_key = get_api_key()
    num_cases = get_number_of_cases()
    custom_prompt = get_custom_prompt()
    
    # Demander le répertoire de sortie
    print("\n📁 Répertoire de sortie")
    print("-" * 60)
    output_dir = input("Répertoire de sortie (défaut: generated_cases): ").strip()
    if not output_dir:
        output_dir = "generated_cases"
    
    # Confirmation
    print("\n" + "=" * 60)
    print("📋 RÉCAPITULATIF")
    print("=" * 60)
    print(f"Nombre de cas à générer: {num_cases}")
    print(f"Répertoire de sortie: {output_dir}")
    print(f"Prompt personnalisé: {'Oui' if custom_prompt else 'Non (défaut)'}")
    print("=" * 60)
    
    confirm = input("\n▶️  Démarrer la génération ? (o/n): ")
    if confirm.lower() != 'o':
        print("❌ Opération annulée")
        sys.exit(0)
    
    # Génération des cas
    print("\n" + "=" * 60)
    print("🚀 DÉBUT DE LA GÉNÉRATION")
    print("=" * 60)
    
    start_time = datetime.now()
    successful = 0
    failed = 0
    
    for i in range(1, num_cases + 1):
        print(f"\n{'=' * 60}")
        print(f"📝 Génération du cas {i}/{num_cases}")
        print(f"{'=' * 60}")
        
        if run_generator(api_key, i, custom_prompt, output_dir):
            successful += 1
        else:
            failed += 1
            retry = input(f"\n⚠️  Réessayer le cas {i} ? (o/n): ")
            if retry.lower() == 'o':
                if run_generator(api_key, i, custom_prompt, output_dir):
                    successful += 1
                    failed -= 1
        
        # Pause entre les requêtes pour éviter la surcharge
        if i < num_cases:
            print("\n⏳ Pause de 2 secondes avant le prochain cas...")
            time.sleep(2)
    
    # Rapport final
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "=" * 60)
    print("✅ GÉNÉRATION TERMINÉE")
    print("=" * 60)
    print(f"Total de cas demandés: {num_cases}")
    print(f"✅ Réussis: {successful}")
    print(f"❌ Échoués: {failed}")
    print(f"⏱️  Durée totale: {duration}")
    print(f"📁 Fichiers sauvegardés dans: {output_dir}/")
    print("=" * 60)
    
    # Afficher les statistiques
    if successful > 0:
        avg_time = duration.total_seconds() / successful
        print(f"\n📊 Temps moyen par cas: {avg_time:.2f} secondes")
    
    print("\n✨ Processus terminé!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Processus interrompu par l'utilisateur")
        sys.exit(1)