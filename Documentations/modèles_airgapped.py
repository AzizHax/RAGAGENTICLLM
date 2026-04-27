# 📦 Déploiement hors-ligne (air-gapped) de modèles Ollama  
**Debian 12 – VMware – Source Windows connectée**

---

## 1. Objectif

Cette documentation décrit une méthode **complète et reproductible** pour :

- installer et utiliser Ollama sur une machine **Debian 12 sans accès Internet**
- transférer des modèles LLM depuis une machine connectée
- contourner les limitations des supports amovibles
- garantir un fonctionnement stable d’Ollama en tant que service systemd
- gérer l’ajout de nouveaux modèles dans le temps

---

## 2. Architecture générale

| Élément | Description |
|------|------------|
| Machine source | Windows (connectée à Internet) |
| Machine cible | Debian 12 (VMware, air-gapped) |
| Hyperviseur | VMware (USB 3.0) |
| Support de transfert | Clé USB formatée en **exFAT** |
| Stockage modèles | Support externe (`/mnt/usb`) |
| Service | `ollama.service` (systemd) |

---

## 3. Contraintes techniques identifiées

- Les modèles Ollama contiennent des fichiers **supérieurs à 4 Go**
- FAT32 est incompatible → **exFAT requis**
- exFAT **ne gère pas** les permissions Unix (`chown`)
- Les services systemd **ne peuvent pas accéder** aux montages utilisateurs (`/media/<user>`)
- La partition système Debian par défaut est souvent trop petite pour stocker plusieurs modèles

---

## 4. Préparation côté Windows (machine connectée)

### 4.1 Installation et téléchargement des modèles

Exemples de modèles téléchargés :

```powershell
ollama pull llama3.1:8b
ollama pull qwen2.5:7b-instruct
ollama pull qwen2.5:3b-instruct
ollama pull mistral:7b-instruct
