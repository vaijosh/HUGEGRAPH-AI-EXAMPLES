# Benchmarks

Scripts behind [docs/model-choice.md](../docs/model-choice.md) and the results in
[docs/text-to-graph-findings.md](../docs/text-to-graph-findings.md). Run them from `KnowledgeGraphExtraction/`
with the project venv, which is one level up in `../.venv`. HugeGraph and Ollama must be running, and for Claude
the gateway variables (`ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN`) must be set.

| Script | What it does |
|---|---|
| `bench_models.py` | Asks every question in a `... QUESTIONS to ASK.md` file through the app's question pipeline (Text2Gremlin, partial-match retry, summary, follow-up rewriting), once per model, and saves answer, Gremlin and time. Uses the app's LLM timeout and retry and runs the last query in a reply, as the app does. Grading the answers is done by hand. |
| `bench_extract.py` | Extracts every `.txt` file with each model and saves vertices/edges per model. Nothing is written to the graph. Reads the schema from the live graph. It calls hugegraph-llm's extraction without the app's prompt caching, split retry and parallel extraction, so it is slower than an import in the app. |
| `score_extract.py` | Scores those extractions against 51 facts from the insurance Q&A file. The fact list is insurance-only. |
| `precision.py` | Per model: names not found in their source document, partial-name duplicates, and the edge-label mix. |
| `extract_with_questions.py` | Extracts the 5 insurance releases about Christian Decristofaro and Joseph Baldwin with the insurance `SCHEMA QUESTIONS.md` in the extraction prompt, as the app sends them, and prints the fraud and prosecutor links found. Optional argument: the number of rounds (default 2). Nothing is written to the graph. |

Models are given as `provider:model`, for example `ollama:qwen2.5:14b` or `claude:anthropic/claude-sonnet`.
`bench_models.py` and `bench_extract.py` set `litellm.drop_params = True`, because Claude Opus 5.5 rejects the
`temperature=0.01` that hugegraph-llm sends.

## Running

Question answering runs against the graph currently loaded in HugeGraph, so build that dataset's graph in the app
first (Reset, paste its `SCHEMA QUESTIONS.md`, infer, create, ingest). The graph holds one dataset at a time.

```bash
../.venv/bin/python benchmarks/bench_models.py "DOJ_fraud_samples/banking/BANKING FRAUD QUESTIONS to ASK.md" \
    benchmarks/results/banking_qa.json claude:anthropic/claude-sonnet
../.venv/bin/python benchmarks/bench_models.py "DOJ_fraud_samples/insurance/INSURANCE FRAUD QUESTIONS to ASK.md" \
    benchmarks/results/insurance/qa.json claude:anthropic/claude-sonnet
../.venv/bin/python benchmarks/bench_models.py "WIKI_semiconductor_samples/SEMICONDUCTOR QUESTIONS to ASK.md" \
    benchmarks/results/semiconductor_qa.json claude:anthropic/claude-sonnet
```

A run takes about 5 minutes for 30 questions. The answers file is written after every question, so a stopped run
keeps what it has. hugegraph-llm prints its log to standard output; add `2>&1 | grep -E "items|DONE|Traceback"` to
see only progress.

Extraction and its scoring, for the model comparison:

```bash
../.venv/bin/python benchmarks/bench_extract.py benchmarks/results/insurance/extract claude:anthropic/claude-sonnet ollama:qwen2.5:14b
python3 benchmarks/score_extract.py -v
python3 benchmarks/precision.py
```

For another dataset, set `BENCH_DOCS` for `bench_extract.py`. `score_extract.py` needs its own fact list for a new
dataset.

Running models one per process in parallel is fine for Claude, within the gateway's token rate limit. Run Ollama
models one at a time, since Ollama serves one request at a time unless `OLLAMA_NUM_PARALLEL` is raised.

## Results

All answers were graded by hand against the questions file and the source documents. Questions files were
corrected when a later graph found facts they missed, so older runs are graded against the file as it was then.

### Insurance (`results/insurance/`, 30 questions)

| File | Run | Correct | Partly | Wrong |
|---|---|---|---|---|
| `qa_sonnet.json` | 27 Oct, model comparison on the 26 Oct schema | 30 | 0 | 0 |
| `qa_opus55.json` | 27 Oct, same graph | 30 | 0 | 0 |
| `qa_opus45.json` | 27 Oct, same graph | 23 | 3 | 4 |
| `qa_qwen.json` | 27 Oct, same graph: qwen3-coder 12 correct, qwen2.5:14b 10 correct | | | |
| `qa_sonnet_2026-10-28.json` | Schema re-inferred without questions (stopped after 28 questions) | 9 | 7 | 12 |
| `qa_sonnet_merged.json` | Schema merged from three drafts | 12 | 12 | 6 |
| `qa_sonnet_questions.json` | Questions sent to schema inference | 23 | 7 | 0 |
| `qa_sonnet_questions_extract.json` | Questions also sent to extraction | 25 | 5 | 0 |
| `qa_nets_off.json` | 29 Oct import without the safety nets | 28 | 2 | 0 |

`extract/` holds the 27 Oct extraction per model, scored by `score_extract.py`: Sonnet 50 of 51 facts, Opus 5.5 50,
Opus 4.5 49, qwen2.5:14b 33.

### Banking (`results/`, 30 questions)

| File | Run | Correct | Partly | Wrong |
|---|---|---|---|---|
| `banking_qa_sonnet_questions.json` | 28 Oct, schema inferred with questions | 28 | 1 | 1 |
| `banking_qa_sonnet_rerun.json` | 28 Oct, new import | 28 | 2 | 0 |
| `banking_qa_nets_off.json` | 29 Oct, without the safety nets; the schema reply had no types, so every property was TEXT | 17 | 4 | 9 |
| `banking_qa_nets_off2.json` | 29 Oct, without the safety nets, typed schema | 29 | 1 | 0 |
| `banking_qa_no_indexes.json` | 29 Oct, without property indexes | 20 | 2 | 8 |
| `banking_qa_cluster.json` | 29 Oct, HugeGraph cluster on hstore, with the text filter rewrite | 27 | 2 | 1 |
| `banking_qa_cluster2.json` | 29 Oct, cluster, new schema and import after Reset removed the old schema | 28 | 2 | 0 |

The first `banking_qa_sonnet_questions.json` wrong answer (#27) was a self-corrected query; the app now runs the
last query in a reply. The 8 wrong answers without indexes are all refused queries that start by filtering on a
property. The `banking_qa_cluster.json` wrong answer (#10) is an extraction miss: that import stored no pending
plea status and no charges for Ryan Small and Wayne Bessant.

### Semiconductor (`results/`, 36 questions)

| File | Run | Correct | Partly | Wrong |
|---|---|---|---|---|
| `semiconductor_qa_sonnet_questions.json` | 28 Oct, schema inferred with questions | 28 | 5 | 3 |
| `semiconductor_qa_sonnet_questions2.json` | 28 Oct graph, re-asked after the Avago answers were changed to Broadcom Inc. | 26 | 7 | 3 |
| `semiconductor_qa_nets_off.json` | 29 Oct import without the safety nets | 28 | 7 | 1 |

On the current questions file, which accepts either Avago or Broadcom Inc. as the acquirer and asks about both,
the 29 Oct run scores 32 correct, 4 partly correct and 0 wrong; its two reworded Avago questions (#11 and #35) were
asked again separately.

### Latest per dataset

| Dataset | Correct | Partly | Wrong |
|---|---|---|---|
| Banking | 29 | 1 | 0 |
| Insurance | 28 | 2 | 0 |
| Semiconductor | 32 | 4 | 0 |
