"""Extract a few insurance releases with the schema questions added to the extraction prompt, and print the
prosecutor and fraud links that matter for questions #12, #17, #20 and #21.

Nothing is written to the graph. The schema is read from the live graph, so load the insurance graph first.
Run from KnowledgeGraphExtraction/: ../.venv/bin/python benchmarks/extract_with_questions.py [rounds]
"""
import glob
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import graphrag_showcase as app  # noqa: E402
from hugegraph_llm.flows.scheduler import SchedulerSingleton  # noqa: E402

rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 2

cfg = app.default_config()
defaults = app.provider_defaults("claude")
app.apply_config("claude", "anthropic/claude-sonnet", cfg["embedding_model"], "", defaults["api_base"],
                 cfg["ollama_host"], cfg["ollama_port"], cfg["graph_url"], cfg["graph_name"], cfg["graph_user"],
                 cfg["graph_pwd"], extract_model="anthropic/claude-sonnet", text2gql_model="anthropic/claude-sonnet")
app.install_llm_timeouts()
app.install_prompt_caching()
app.install_split_retry()
app.install_extraction_recorder()
app.install_parallel_extraction()

schema = app.fetch_graph_schema()
if not schema or not schema.get("vertexlabels"):
    sys.exit("The graph has no schema. Load the insurance graph first.")
schema = json.dumps(schema, ensure_ascii=False)
questions_md = (ROOT / "DOJ_fraud_samples/insurance/SCHEMA QUESTIONS.md").read_text(encoding="utf-8")
questions = re.search(r"```text\n(.*?)\n```", questions_md, re.S).group(1)
hints = app.question_hints(questions)  # the wording the app sends with the questions box

files = sorted(glob.glob(str(ROOT / "DOJ_fraud_samples/insurance/04_*.txt")))
files += sorted(glob.glob(str(ROOT / "DOJ_fraud_samples/insurance/06_*.txt")))
scheduler = SchedulerSingleton.get_instance()


def name(vertex_id: str) -> str:
    return vertex_id.split(":", 1)[-1]


def extract(path: str):
    extracted, _ = app.extract_file(path, schema, hints, scheduler)
    return Path(path).name[:45], extracted


for n in range(1, rounds + 1):
    print(f"\nRUN {n}", flush=True)
    with ThreadPoolExecutor(len(files)) as pool:
        results = list(pool.map(extract, files))
    for file_name, extracted in results:
        edges = extracted.get("edges", [])
        if file_name.startswith("04_"):
            fraud = [(name(e["inV"]), e.get("properties")) for e in edges
                     if e["label"].startswith("defraud") and "Decristofaro" in e["outV"]]
            print(f"  {file_name}: Decristofaro defrauded = {fraud or 'none'}")
        else:
            prosecuted = sorted(f"{name(e['outV'])} -> {name(e['inV'])}" for e in edges
                                if e["label"].startswith("prosecut") and "Baldwin" in e["outV"] + e["inV"])
            print(f"  {file_name}: Baldwin prosecuted = {prosecuted or 'none'}")
print("\nDONE")
