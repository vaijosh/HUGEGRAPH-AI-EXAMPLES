"""Extract the insurance .txt files with each model (no graph import) and save vertices/edges per model.

Usage: bench_extract.py OUT_DIR provider:model [provider:model ...]   e.g. ollama:qwen2.5:14b claude:anthropic/claude-sonnet
Set BENCH_DOCS to another folder of .txt files to extract a different dataset (default: the insurance samples).
"""
import glob
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import graphrag_showcase as app  # noqa: E402
import litellm  # noqa: E402

litellm.drop_params = True  # Opus 5.5 only accepts temperature=1; hugegraph-llm sends 0.01
from hugegraph_llm.flows.scheduler import SchedulerSingleton  # noqa: E402

out_dir, specs = Path(sys.argv[1]), sys.argv[2:]
out_dir.mkdir(parents=True, exist_ok=True)
files = sorted(glob.glob(str(Path(os.getenv("BENCH_DOCS", ROOT / "DOJ_fraud_samples" / "insurance")) / "*.txt")))
schema_dict = app.fetch_graph_schema()
schema = json.dumps(schema_dict, ensure_ascii=False)
cfg = app.default_config()

for spec in specs:
    provider, model = spec.split(":", 1)
    defaults = app.provider_defaults(provider)
    app.apply_config(provider, model, defaults["embedding_model"], "", defaults["api_base"], cfg["ollama_host"],
                     cfg["ollama_port"], cfg["graph_url"], cfg["graph_name"], cfg["graph_user"], cfg["graph_pwd"],
                     extract_model=model, text2gql_model=model)
    scheduler = SchedulerSingleton.get_instance()
    result, start = {}, time.time()
    for path in files:
        t = time.time()
        try:
            extracted, raw = app.extract_file(path, schema, "", scheduler)
            drops = app.audit_extraction(raw, schema_dict)
            app.drop_null_properties(extracted)
            app.coerce_property_types(extracted, schema_dict)
            result[Path(path).name] = {"vertices": extracted.get("vertices", []), "edges": extracted.get("edges", []),
                                       "dropped": app.dropped_item_count(drops), "secs": round(time.time() - t, 1)}
        except Exception as exc:  # noqa: BLE001
            result[Path(path).name] = {"error": str(exc), "secs": round(time.time() - t, 1)}
        r = result[Path(path).name]
        print(f"[{spec}] {Path(path).name[:40]} v={len(r.get('vertices', []))} e={len(r.get('edges', []))} "
              f"{r['secs']}s {r.get('error', '')[:80]}", flush=True)
        name = spec.replace("/", "_").replace(":", "_")
        json.dump(result, open(out_dir / f"{name}.json", "w"), indent=1, ensure_ascii=False, default=str)
    print(f"[{spec}] DONE in {round(time.time() - start)}s", flush=True)
