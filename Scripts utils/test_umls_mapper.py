from umls_mapper import UMLSMapper

m = UMLSMapper("umls.sqlite")

tests = [
    ("polyarthrite rhumatoïde", None, "disease"),
    ("PR", "diagnostic de polyarthrite rhumatoïde (PR) suspectée", "disease"),
    ("RF", "facteur rhumatoïde (RF) positif", "lab"),
    ("ACPA", "anticorps anti-citrullinés (ACPA) positifs", "lab"),
    ("MTX", "traitement par MTX en cours", "drug"),
]

for mention, ctx, cat in tests:
    out = m.map(mention=mention, context=ctx, expected_category=cat, topk=5)
    print("\nMENTION:", mention, "| CAT:", cat)
    print("queries_tried:", out["queries_tried"])
    print("query_used:", out["query_used"])
    print("status:", out["status"])
    if out["best_hit"]:
        b = out["best_hit"]
        print("best:", b["cui"], "|", b["string"], "|", b["sab"], "| conf=", b["confidence"])

m.close()
