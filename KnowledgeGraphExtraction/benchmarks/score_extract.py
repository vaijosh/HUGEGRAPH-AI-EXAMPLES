"""Score each model's insurance extraction against facts from the checked Q&A file (verified against the texts).

Usage: score_extract.py [RESULTS_DIR] [-v]   (default RESULTS_DIR: benchmarks/results/insurance/extract)
The facts below are specific to DOJ_fraud_samples/insurance; other datasets need their own list.
"""
import glob
import json
import re
import sys
from collections import Counter
from pathlib import Path


def norm(s):
    s = str(s).lower().replace("’", "'").replace("‘", "'")
    return re.sub(r"[^a-z0-9' ]+", " ", s).split()


def name_match(want, have):
    have_tokens = set(norm(have))
    return all(t in have_tokens for t in norm(want))


def num_match(want, have):
    try:
        return abs(float(have) - want) <= max(1.0, want * 0.001)
    except (TypeError, ValueError):
        return False


def text_match(want, have):
    return have is not None and " ".join(norm(want)) in " ".join(norm(have))


# (source, edge label, target, {edge prop: expected}); target None = any target; symmetric labels match either way
E = [
    ("Brendan Lawler", "owns", "BL Insurance Brokerage", {}),
    ("Lisa Lawler", "owns", "BL Insurance Brokerage", {}),
    ("Brendan Lawler", "spouse_of", "Lisa Lawler", {}),
    ("Brendan Lawler", "sentenced_by", "Myong J. Joun", {"sentence_months": 8}),
    ("Lisa Lawler", "sentenced_by", "Myong J. Joun", {"sentence_months": 8}),
    ("James Keating", "owns", "Kodiak Asset Recovery", {}),
    ("James Keating", "owns", "Surety Risk Solutions", {}),
    ("James Keating", "owns", "American Construction Industrial", {}),
    ("James Keating", "employed_by", "U.S. Fire Insurance", {}),
    ("James Keating", "employed_by", "Allied World", {}),
    ("James Keating", "defrauded", "Allied World", {"loss_amount": 1446491.95}),
    ("James Keating", "sentenced_by", "Victor A. Bolden", {"restitution_amount": 1226603.97}),
    ("Tonja Van Roy", "defrauded", "AFCO Credit", {"loss_amount": 3700000, "charge": "wire fraud"}),
    ("Tonja Van Roy", "investigated_by", "Homeland Security Investigations", {}),
    ("Tonja Van Roy", "investigated_by", "California Department of Insurance", {}),
    ("Tonja Van Roy", "sentenced_by", "Stephen V. Wilson", {"sentence_months": 50, "restitution_amount": 1880237}),
    ("Tonja Van Roy", "owns", "Pegasus Insurance", {}),
    ("Heleonel Gonzalez", "defrauded", "Peach State Federal Credit Union", {"loss_amount": 6000000}),
    ("Heleonel Gonzalez", "owns", "Sharp Insurance Agency", {}),
    ("Lisette Patricio", "conspired_with", "Arturo Morales", {}),
    ("Lisette Patricio", "conspired_with", "Yarelis Felipe", {}),
    ("Lisette Patricio", "conspired_with", "Darely Carballeira", {}),
    ("Jaron Earl Taylor", "conspired_with", "Davion Percy", {}),
    ("Jaron Earl Taylor", "conspired_with", "Candace Tyler", {}),
    ("Jaron Earl Taylor", "conspired_with", "Conrad D'Haiti", {}),
    ("Jaron Earl Taylor", "conspired_with", "Michael Anthony Owen", {}),
    ("Louis Belletieri", "prosecuted_by", "John J. Boscia", {}),
    ("Louis Belletieri", "prosecuted_by", "Rebecca J. Kulik", {}),
    ("Louis Belletieri", "sentenced_by", None, {"sentence_months": 54, "restitution_amount": 6000000}),
    ("Davion Percy", "prosecuted_by", "Joseph Baldwin", {}),
    ("Jaron Earl Taylor", "prosecuted_by", "Joseph Baldwin", {}),
    ("Michael Anthony Owen", "sentenced_by", "Lydia Griggsby", {"sentence_months": 6}),
    ("James William Wilson", "sentenced_by", "Deborah K. Chasanow", {"sentence_months": 144, "restitution_amount": 18705520.30}),
    ("Maureen Ann Wilson", "sentenced_by", "Deborah K. Chasanow", {"restitution_amount": 18705520.30}),
    ("James William Wilson", "spouse_of", "Maureen Wilson", {}),
    ("Thomas J. Lemmen", "sentenced_by", None, {"sentence_months": 70, "restitution_amount": 1601505.27}),
    ("Lonnie White", "sentenced_by", None, {"sentence_months": 53, "restitution_amount": 2375861}),
    ("Lonnie White", "spouse_of", "Lisa Ogletree", {}),
    ("Lonnie White", "conspired_with", "Lisa Ogletree", {}),
    ("Christian Decristofaro", "defrauded", None, {"loss_amount": 2196323}),
    ("Christian Decristofaro", "sentenced_by", "Indira Talwani", {"restitution_amount": 2196323}),
    ("Christian Decristofaro", "owns", "NE Premier Home Services", {}),
    ("David L. Mongeau", "owns", "Mongeau Enterprises", {}),
    ("David L. Mongeau", "defrauded", "First National Bank of Syracuse", {}),
]
# (vertex name, property, expected)
V = [
    ("Lisette Patricio", "age", 64),
    ("Maureen Ann Wilson", "age", 77),
    ("Davion Percy", "occupation", "police officer"),
    ("Candace Tyler", "occupation", "police officer"),
    ("Jaron Earl Taylor", "occupation", "police officer"),
    ("Christian Conrad Claus", "occupation", "police officer"),
    ("Michael Anthony Owen", "occupation", "police officer"),
]
SYMMETRIC = {"spouse_of", "conspired_with"}


def score(path):
    data = json.load(open(path))
    vertices, edges, secs, dropped, errors = {}, [], 0.0, 0, 0
    for doc in data.values():
        secs += doc.get("secs", 0)
        errors += "error" in doc
        dropped += doc.get("dropped", 0)
        for v in doc.get("vertices", []):
            vertices.setdefault(v["id"], v)
        edges += doc.get("edges", [])
    vname = {vid: v["properties"].get("name", vid.split(":", 1)[-1]) for vid, v in vertices.items()}

    def endpoints(e):
        return vname.get(e["outV"], e["outV"].split(":", 1)[-1]), vname.get(e["inV"], e["inV"].split(":", 1)[-1])

    found, partial, missing = 0, 0, []
    for src, label, dst, props in E:
        best = -1
        for e in edges:
            if e["label"] != label:
                continue
            a, b = endpoints(e)
            pairs = [(a, b), (b, a)] if label in SYMMETRIC else [(a, b)]
            if not any(name_match(src, x) and (dst is None or name_match(dst, y)) for x, y in pairs):
                continue
            ok = sum(num_match(v, e["properties"].get(k)) if isinstance(v, (int, float))
                     else text_match(v, e["properties"].get(k)) for k, v in props.items())
            best = max(best, ok)
        if best == len(props):
            found += 1
        elif best >= 0:
            partial += 1
            missing.append(f"{src} -{label}-> {dst}: edge found, props {best}/{len(props)}")
        else:
            missing.append(f"{src} -{label}-> {dst}: no edge")
    for name, prop, want in V:
        hit = any(name_match(name, v["properties"].get("name", "")) and
                  (num_match(want, v["properties"].get(prop)) if isinstance(want, (int, float))
                   else text_match(want, v["properties"].get(prop))) for v in vertices.values())
        found += hit
        if not hit:
            missing.append(f"{name}.{prop}={want}: missing")
    names = Counter(" ".join(norm(n)) for n in vname.values())
    return {"facts": found, "partial": partial, "total": len(E) + len(V), "vertices": len(vertices),
            "edges": len(edges), "dropped": dropped, "errors": errors, "secs": round(secs),
            "docs": len(data), "missing": missing}


args = [a for a in sys.argv[1:] if a != "-v"]
results_dir = Path(args[0]) if args else Path(__file__).resolve().parent / "results" / "insurance" / "extract"
files = sorted(glob.glob(str(results_dir / "*.json")))
results = {Path(f).stem: score(f) for f in files}
for name, r in results.items():
    print(f"{name:40} facts {r['facts']}/{r['total']} partial {r['partial']} | v={r['vertices']} e={r['edges']} "
          f"dropped={r['dropped']} errors={r['errors']} docs={r['docs']} time={r['secs']}s")
if "-v" in sys.argv:
    for name, r in results.items():
        print(f"\n== {name}")
        for m in r["missing"]:
            print("  ", m)
