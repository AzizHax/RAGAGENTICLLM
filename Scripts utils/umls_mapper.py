import sqlite3
import re
from typing import List, Dict, Any, Optional, Iterable, Tuple
from rapidfuzz import fuzz

def _normalize(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def _tokenize(s: str) -> List[str]:
    s = _normalize(s)
    return [t for t in re.split(r"[^a-z0-9]+", s) if t]

class UMLSMapper:
    """
    Requires SQLite with:
      - conso(rowid, cui, lat, sab, tty, ispref, str)
      - conso_fts (FTS5) over conso.str
      - sty(cui, sty) (optional but present in your DB)
    """
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.con = sqlite3.connect(db_path)
        self.con.row_factory = sqlite3.Row
        self._validate_schema()

    def close(self):
        self.con.close()

    # ---------- schema / helpers ----------
    def _table_exists(self, name: str) -> bool:
        row = self.con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
            (name,),
        ).fetchone()
        return row is not None

    def _validate_schema(self):
        # Hard requirements
        if not self._table_exists("conso"):
            raise RuntimeError(
                f"UMLS DB schema mismatch: missing table 'conso'. db_path={self.db_path}"
            )
        # FTS is optional but expected in your case
        self.has_fts = self._table_exists("conso_fts")
        self.has_sty = self._table_exists("sty")

    def _fetch_rows_fts(self, qn: str, topk_fts: int) -> List[sqlite3.Row]:
        # Try phrase match first, then token query
        phrase = f"\"{qn}\""
        rows = self.con.execute(
            """
            SELECT c.rowid, c.cui, c.lat, c.sab, c.tty, c.ispref, c.str
            FROM conso_fts f
            JOIN conso c ON c.rowid = f.rowid
            WHERE conso_fts MATCH ?
            LIMIT ?;
            """,
            (phrase, topk_fts),
        ).fetchall()

        if rows:
            return rows

        toks = _tokenize(qn)
        if not toks:
            return []
        token_query = " ".join(toks)
        rows = self.con.execute(
            """
            SELECT c.rowid, c.cui, c.lat, c.sab, c.tty, c.ispref, c.str
            FROM conso_fts f
            JOIN conso c ON c.rowid = f.rowid
            WHERE conso_fts MATCH ?
            LIMIT ?;
            """,
            (token_query, topk_fts),
        ).fetchall()
        return rows

    def _fetch_rows_like(self, qn: str, topk_fts: int) -> List[sqlite3.Row]:
        # Fallback when FTS missing (slower)
        toks = _tokenize(qn)
        if not toks:
            return []
        # Build AND of LIKE
        where = " AND ".join(["lower(c.str) LIKE ?"] * len(toks))
        params = [f"%{t}%" for t in toks] + [topk_fts]
        q = f"""
        SELECT c.rowid, c.cui, c.lat, c.sab, c.tty, c.ispref, c.str
        FROM conso c
        WHERE {where}
        LIMIT ?;
        """
        return self.con.execute(q, params).fetchall()

    def _get_sem_types(self, cuis: Iterable[str]) -> Dict[str, List[str]]:
        if not self.has_sty:
            return {}
        cuis = list({c for c in cuis if c})
        if not cuis:
            return {}
        qmarks = ",".join(["?"] * len(cuis))
        rows = self.con.execute(
            f"SELECT cui, sty FROM sty WHERE cui IN ({qmarks})",
            cuis
        ).fetchall()
        out: Dict[str, List[str]] = {}
        for r in rows:
            out.setdefault(r["cui"], []).append(r["sty"])
        return out

    # ---------- core lookup ----------
    def lookup(
        self,
        query: str,
        topk_fts: int = 50,
        topk: int = 10,
        sab_prefer: Optional[List[str]] = None,
        require_sty: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        qn = _normalize(query)
        if not qn:
            return []

        # Fetch candidates
        if self.has_fts:
            rows = self._fetch_rows_fts(qn, topk_fts)
        else:
            rows = self._fetch_rows_like(qn, topk_fts)

        if not rows:
            return []

        # Semantic types (optional filter)
        sty_map: Dict[str, List[str]] = {}
        if require_sty:
            sty_map = self._get_sem_types(r["cui"] for r in rows)

        sab_prefer_set = set(sab_prefer) if sab_prefer else None

        scored: List[Dict[str, Any]] = []
        for r in rows:
            cand = r["str"]
            score = float(fuzz.token_set_ratio(qn, _normalize(cand)))  # 0..100

            # small boosts
            if int(r["ispref"]) == 1:
                score += 3.0
            if sab_prefer_set and r["sab"] in sab_prefer_set:
                score += 2.0

            if require_sty:
                stys = sty_map.get(r["cui"], [])
                ok = any(
                    any(req.lower() in sty.lower() for req in require_sty)
                    for sty in stys
                )
                if not ok:
                    continue

            scored.append({
                "cui": r["cui"],
                "string": r["str"],
                "lat": r["lat"],
                "sab": r["sab"],
                "tty": r["tty"],
                "ispref": bool(r["ispref"]),
                "score": score,
                "semantic_types": sty_map.get(r["cui"], []),
            })

        if not scored:
            return []

        # Deduplicate by CUI keep best
        best_by_cui: Dict[str, Dict[str, Any]] = {}
        for item in scored:
            cui = item["cui"]
            if cui not in best_by_cui or item["score"] > best_by_cui[cui]["score"]:
                best_by_cui[cui] = item

        out = sorted(best_by_cui.values(), key=lambda x: x["score"], reverse=True)[:topk]
        return out

    # ---------- high-level mapping with acronym handling ----------
    _ACRONYM_EXPANSIONS = {
        "PR": ["polyarthrite rhumatoïde", "rheumatoid arthritis", "PR polyarthrite rhumatoïde"],
        "RF": ["rheumatoid factor", "facteur rhumatoïde", "RF rheumatoid factor"],
        "ACPA": ["anti-citrullinated protein antibodies", "anticorps anti-citrullinés", "anti-CCP", "ACPA anti-CCP"],
        "ANTI-CCP": ["anti-CCP", "anti-CCP antibody", "anticorps anti-CCP"],
        "MTX": ["methotrexate", "méthotrexate", "MTX methotrexate"],
    }

    _CATEGORY_CFG = {
        "disease": dict(sab_prefer=["SNOMEDCT_US", "MSH", "MSHFRE"]),
        "lab":     dict(sab_prefer=["SNOMEDCT_US", "MSH", "MSHFRE"]),
        "drug":    dict(sab_prefer=["RXNORM", "SNOMEDCT_US", "MSH"]),
    }

    def map(
        self,
        mention: str,
        category: str,
        topk: int = 5,
    ) -> Dict[str, Any]:
        """
        Returns:
          {
            mention, category,
            queries_tried, query_used,
            status: ok/none,
            best: {...} | None,
            top: [...]
          }
        """
        m = (mention or "").strip()
        cat = (category or "").strip().lower()

        cfg = self._CATEGORY_CFG.get(cat, {})
        sab_prefer = cfg.get("sab_prefer")

        # Build query list
        queries_tried: List[str] = []
        key = m.upper()
        if key in self._ACRONYM_EXPANSIONS:
            queries = self._ACRONYM_EXPANSIONS[key]
        else:
            queries = [m]

        best_hit = None
        used = None
        best_list = None

        for q in queries:
            queries_tried.append(q)
            hits = self.lookup(q, topk=topk, sab_prefer=sab_prefer)
            if hits:
                best_hit = hits[0]
                best_list = hits
                used = q
                break

        return {
            "mention": m,
            "category": cat,
            "queries_tried": queries_tried,
            "query_used": used,
            "status": "ok" if best_hit else "none",
            "best": best_hit,
            "top": best_list or [],
        }
