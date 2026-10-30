"""Compare Ollama chat/Text2Gremlin models on a '... QUESTIONS to ASK.md' file against the loaded graph."""
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import graphrag_showcase as app  # noqa: E402
import litellm  # noqa: E402

litellm.drop_params = True  # Opus 5.5 only accepts temperature=1
app.install_llm_timeouts()  # as the app does, so a stalled gateway call times out and is retried
app.install_last_gremlin_block()  # as the app does, so a self-corrected query is the one that runs
app.install_text_filter_rewrite()  # as the app does, for servers older than HugeGraph 1.8

qfile, out, models = sys.argv[1], sys.argv[2], sys.argv[3:]
text = open(qfile, encoding="utf-8").read()
text = text.split("## Known gaps")[0]

# Each item: list of questions (follow-ups chain), expected answer after the last one.
items, pending = [], []
section = ""
for m in re.finditer(r"^## ([^\n]+)$|```text\n([^`]+?)\n```|^\*\*A:\*\* ([^\n]+)$", text, re.M):
    if m[1]:
        section, pending = m[1].strip(), []
    elif m[2]:
        pending.append(m[2].strip())
    elif m[3]:
        items.append({"section": section, "questions": pending, "expected": m[3].strip()})
        pending = []
print(f"{len(items)} items", flush=True)

cfg = app.default_config()
results = []
for spec in models:
    provider, model = spec.split(":", 1) if spec.split(":", 1)[0] in app.PROVIDERS else ("ollama", spec)
    defaults = app.provider_defaults(provider)
    app.apply_config(provider, model, cfg["embedding_model"], "", defaults["api_base"], cfg["ollama_host"], cfg["ollama_port"],
                     cfg["graph_url"], cfg["graph_name"], cfg["graph_user"], cfg["graph_pwd"],
                     extract_model=cfg["extract_model"], text2gql_model=model)
    for n, item in enumerate(items, 1):
        history, start = [], time.time()
        try:
            for q in item["questions"]:
                for step in app.ask_with_history(q, history):
                    if isinstance(step[3], list):
                        history = step[3]
            last = history[-1]
            answer, gremlin = last["answer"], last["gremlin"]
        except Exception as exc:  # noqa: BLE001
            answer, gremlin = f"ERROR: {exc}", ""
        secs = round(time.time() - start, 1)
        results.append({"model": model, "n": n, **item, "answer": answer, "gremlin": gremlin, "secs": secs})
        print(f"[{model}] {n}/{len(items)} {secs}s", flush=True)
        json.dump(results, open(out, "w"), indent=1, ensure_ascii=False)
print("DONE", flush=True)
