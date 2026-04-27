import pandas as pd
import json

# Tes données JSON (extraites de ton message)
data = [
    # ... (le contenu de ton fichier ehr_ioa_test_50p.json) ...
]

# Si tu as le fichier localement, tu peux aussi faire :
# with open('ehr_ioa_test_50p.json', 'r', encoding='utf-8') as f:
#     data = json.load(f)

rows = []

# On aplatit la structure nichée : Patient -> Stay -> Record
for patient in data:
    patient_id = patient.get("patient_id")
    for stay in patient.get("stays", []):
        stay_id = stay.get("stay_id")
        for record in stay.get("records", []):
            rows.append({
                "NIPATIENT": patient_id,  # Nom de colonne par défaut dans ton run.py
                "NISEJOUR": stay_id,      # Nom de colonne par défaut dans ton run.py
                "LIBELLE": record.get("LIBELLE"),
                "REPONSE": record.get("REPONSE")
            })

# Création du DataFrame
df = pd.DataFrame(rows)

# Sauvegarde en format Parquet
# Note : nécessite l'installation de 'pyarrow' ou 'fastparquet'
df.to_parquet('ehr.parquet', engine='pyarrow', index=False)

print(f"Fichier 'ehr.parquet' généré avec {len(df)} lignes.")