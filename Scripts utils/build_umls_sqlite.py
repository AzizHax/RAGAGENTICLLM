import argparse, sqlite3, csv, os
from tqdm import tqdm

def build_db(mrconso_path: str, mrsty_path: str | None, out_db: str,
             languages=("ENG", "FRE"),
             sab_allow=None,
             keep_suppressed=False):
    if os.path.exists(out_db):
        os.remove(out_db)

    con = sqlite3.connect(out_db)
    cur = con.cursor()

    # Main table
    cur.execute("""
    CREATE TABLE conso (
        cui TEXT,
        lat TEXT,
        sab TEXT,
        tty TEXT,
        ispref INTEGER,
        str TEXT
    );
    """)

    # Optional semantic types
    cur.execute("""
    CREATE TABLE sty (
        cui TEXT,
        tui TEXT,
        sty TEXT
    );
    """)

    # FTS index for fast lookup
    # We keep fields duplicated for search; join on rowid
    cur.execute("""
    CREATE VIRTUAL TABLE conso_fts USING fts5(
        str,
        content='conso',
        content_rowid='rowid',
        tokenize = 'unicode61 remove_diacritics 2'
    );
    """)

    con.commit()

    # Load MRCONSO.RRF
    # Columns (relevant): CUI(0) LAT(1) TS(2) LUI(3) STT(4) SUI(5) ISPREF(6) AUI(7) SAUI(8) SCUI(9)
    #                    SDUI(10) SAB(11) TTY(12) CODE(13) STR(14) SRL(15) SUPPRESS(16) CVF(17)
    allowed_langs = set(languages)

    sab_allow_set = set(sab_allow) if sab_allow else None
    import csv
    import sys
    try:
        csv.field_size_limit(10_000_000)
    except OverflowError:
        csv.field_size_limit(1_000_000)


    inserted = 0
    with open(mrconso_path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f, delimiter="|")
        for row in tqdm(reader, desc="Loading MRCONSO"):
            if len(row) < 15:
                continue
            cui = row[0]
            lat = row[1]
            ispref = 1 if row[6] == "Y" else 0
            sab = row[11]
            tty = row[12]
            s = row[14]
            suppress = row[16] if len(row) > 16 else "N"

            if lat not in allowed_langs:
                continue
            if sab_allow_set is not None and sab not in sab_allow_set:
                continue
            if (not keep_suppressed) and suppress and suppress != "N":
                continue
            if not s:
                continue

            cur.execute("INSERT INTO conso(cui, lat, sab, tty, ispref, str) VALUES (?,?,?,?,?,?)",
                        (cui, lat, sab, tty, ispref, s))
            inserted += 1

    con.commit()

    # Build FTS index
    cur.execute("INSERT INTO conso_fts(conso_fts) VALUES('rebuild');")
    con.commit()

    # Load MRSTY if provided
    if mrsty_path and os.path.exists(mrsty_path):
        with open(mrsty_path, "r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.reader(f, delimiter="|")
            for row in tqdm(reader, desc="Loading MRSTY"):
                if len(row) < 4:
                    continue
                cui, tui, sty = row[0], row[1], row[3]
                cur.execute("INSERT INTO sty(cui, tui, sty) VALUES (?,?,?)", (cui, tui, sty))
        con.commit()

    # Helpful indexes
    cur.execute("CREATE INDEX idx_conso_cui ON conso(cui);")
    cur.execute("CREATE INDEX idx_conso_sab ON conso(sab);")
    con.commit()
    con.close()

    print(f"✅ Built {out_db} with {inserted:,} MRCONSO rows (langs={languages}, sab_allow={sab_allow})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mrconso", required=True, help="Path to MRCONSO.RRF")
    ap.add_argument("--mrsty", default=None, help="Path to MRSTY.RRF (optional)")
    ap.add_argument("--out", required=True, help="Output SQLite DB path, e.g. umls.sqlite")
    ap.add_argument("--langs", default="ENG,FRE", help="Comma-separated languages, e.g. ENG or ENG,FRE")
    ap.add_argument("--sab", default=None, help="Comma-separated SAB allowlist (optional), e.g. SNOMEDCT_US,MSH")
    ap.add_argument("--keep_suppressed", action="store_true", help="Keep suppressed strings")
    args = ap.parse_args()

    langs = tuple([x.strip() for x in args.langs.split(",") if x.strip()])
    sab = [x.strip() for x in args.sab.split(",")] if args.sab else None

    build_db(args.mrconso, args.mrsty, args.out, languages=langs, sab_allow=sab,
             keep_suppressed=args.keep_suppressed)