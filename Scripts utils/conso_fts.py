import sqlite3

DB = "umls.sqlite"

con = sqlite3.connect(DB)
cur = con.cursor()

# 1) créer la table FTS5 (si pas déjà)
cur.execute("""
CREATE VIRTUAL TABLE IF NOT EXISTS conso_fts
USING fts5(str, content='conso', content_rowid='rowid');
""")

# 2) remplir l'index depuis conso
cur.execute("INSERT INTO conso_fts(rowid, str) SELECT rowid, str FROM conso;")

con.commit()
con.close()

print("✅ conso_fts created and populated")
