"""Per model: extracted names not found in their source document, partial-name people, and the edge-label mix.

Usage: precision.py [RESULTS_DIR] [DOCS_DIR]   (defaults: benchmarks/results/insurance/extract, the insurance samples)
"""
import json, glob, re, sys
from collections import Counter
from pathlib import Path
HERE = Path(__file__).resolve().parent
RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results" / "insurance" / "extract"
DOCS = str(Path(sys.argv[2]) if len(sys.argv) > 2 else HERE.parent / "DOJ_fraud_samples" / "insurance") + "/"
def norm(s): return re.sub(r"[^a-z0-9 ]+", " ", str(s).lower().replace("’","'").replace("'","")).split()
for f in sorted(glob.glob(str(RESULTS / "*.json"))):
    data = json.load(open(f)); ungrounded=[]; nv=0; labels=Counter(); people=Counter()
    for doc, x in data.items():
        text = " ".join(norm(open(DOCS+doc).read())); toks=set(text.split())
        for v in x.get("vertices", []):
            nv += 1; name = v["properties"].get("name", "")
            if not all(t in toks for t in norm(name)): ungrounded.append(name)
            if v["label"]=="person": people[" ".join(norm(name))]+=1
        labels.update(e["label"] for e in x.get("edges", []))
    # people whose name is a strict token-subset of another person's name (surname-only or partial dupes)
    names=list(people); partial=[a for a in names if len(a.split())>=1 and any(a!=b and set(a.split())<set(b.split()) for b in names)]
    print(f"{Path(f).stem:36} docs={len(data)} vertices={nv} ungrounded={len(ungrounded)} partial-name people={len(partial)}")
    print("   edges:", dict(labels.most_common()))
    print("   ungrounded sample:", ungrounded[:8])
    print("   partial-name sample:", partial[:8])
