# Why the demo runs on Claude Sonnet

We tested five models on both jobs the app gives a language model: turning documents into a graph, and answering
questions about that graph. Sonnet was as accurate as anything we tried, faster than Opus, and cheaper than Opus.

*Model comparison run 27 Oct 2026 on the insurance fraud dataset (27 DOJ press releases), using the Cloudera AI
gateway for Claude and local Ollama for qwen. Sonnet was then run on all three datasets with the current pipeline on
29 Oct 2026.*

## Decision

**Use Claude Sonnet for chat, extraction and Text2Gremlin.** Keep `nomic-embed-text` on Ollama for embeddings,
because Anthropic has no embedding API.

Sonnet found 50 of 51 checked facts during ingestion and answered all 30 checked questions correctly. Opus 5.5
matched it but was slower and costs more. Opus 4.5 and both local qwen models answered noticeably fewer questions
correctly. With the current pipeline, Sonnet answers 29 of 30 banking, 28 of 30 insurance and 32 of 36
semiconductor questions correctly, with no wrong answers.

## How the models were tested

Both tests used the app's own code paths, so the results reflect what a user of the demo would see.

- **Ingestion.** Each model extracted vertices and edges from all 27 insurance documents using the same inferred
  schema and the app's usual cleanup steps. Nothing was written to the graph. The output was checked against 51
  facts: who owns which company, spouses, conspirators, prosecutors, judges, sentence length in months,
  restitution, loss amounts, charges, investigating agencies, ages and occupations. The facts come from the
  dataset's checked answer file and were spot-checked against the press releases.
- **Question answering.** Each model answered the 30 questions in
  `DOJ_fraud_samples/insurance/INSURANCE FRAUD QUESTIONS to ASK.md` against the same loaded graph. They went
  through the full pipeline: Text2Gremlin, the partial-match retry, the written summary and follow-up rewriting.
  The questions cover single-hop lookups, numeric filters, multi-hop paths, aggregations and follow-up questions.
  Each answer was graded against the expected answer.

## Ingestion accuracy

| Model | Facts found | Vertices / edges | Time, 27 docs | Verdict |
|---|---|---|---|---|
| **Claude Sonnet** | **50 / 51** | 257 / 271 | 325 s | Chosen |
| Claude Opus 5.5 | 50 / 51 | 316 / 332 | 454 s | Equal accuracy |
| Claude Opus 4.5 | 49 / 51 | 288 / 245 | 265 s | Equal accuracy |
| qwen2.5:14b (local) | 33 / 51 | 179 / 276 | 2,934 s | Misses a third |

qwen3-coder extraction was not measured; that run was cancelled. Times come from processing the files one after
another. The app ingests several files at once, so real ingestion is faster for the Claude models.

All three Claude models are close to a perfect score, and all three missed the same fact. No Claude model invented
a name: every extracted name appears in its source document, apart from one inferred "New Orleans Police
Department" from Opus 4.5. qwen2.5:14b missed about a third of the facts and was about 9 times slower than Sonnet.
It also created generic vertices such as "Insurance Company". The graph can only answer what ingestion put into it,
so these misses cannot be recovered at query time.

## Question-answering accuracy

| Model | Correct | Partly correct | Wrong, empty or error | Median per question |
|---|---|---|---|---|
| **Claude Sonnet** | **30 / 30** | 0 | 0 | 9.2 s |
| Claude Opus 5.5 | 30 / 30 | 0 | 0 | 12.2 s |
| Claude Opus 4.5 | 23 / 30 | 3 | 4 | 12.4 s |
| qwen3-coder (local) | 12 / 30 | 4 | 14 | 3.4 s |
| qwen2.5:14b (local) | 10 / 30 | 5 | 15 | 6.7 s |

One-hop lookups worked for every model. The harder questions separated them:

- **Values stored on edges.** Sentence months, restitution and loss amounts are stored on the relationships, not
  on the people. Sonnet and Opus 5.5 read them correctly. Both qwen models read them from the wrong place, so every
  "how many months" or "how much" answer came back empty.
- **Edge direction.** "Which defendants did Joseph Baldwin prosecute?" means following the `prosecuted_by` edge
  backwards. Both qwen models got the direction wrong and found nothing.
- **Aggregations.** The longest sentence, the most common charge and judges with more than one defendant were all
  right with Sonnet. qwen3-coder answered several of these confidently and wrongly: it named the wrong defendant
  for the longest sentence, said "bank fraud" instead of wire fraud, and listed five judges as if they were
  defendants.
- **Opus 4.5** returned empty results for two numeric filters, wrote one Gremlin query that failed to run,
  undercounted the FBI's cases (5 instead of 8) and said James Keating owned one company when he owned three.

## Sonnet on all three datasets

After the comparison, the pipeline changed: the schema is inferred from the questions the graph should answer as
well as the documents, extraction gets the same questions, a schema without types is inferred again, LLM calls have
a timeout and retry, and a self-corrected query is the one that runs. Sonnet was then run on each dataset, with a
schema inferred from that dataset's `SCHEMA QUESTIONS.md`:

| Dataset | Questions | Correct | Partly correct | Wrong |
|---|---|---|---|---|
| Banking fraud (26 DOJ releases) | 30 | 29 | 1 | 0 |
| Insurance fraud (27 DOJ releases) | 30 | 28 | 2 | 0 |
| Semiconductor companies (8 Wikipedia articles) | 36 | 32 | 4 | 0 |

The partly correct answers are facts extraction missed or misread, such as a two-year mandatory minimum stored as a
maximum, or an acquisition year off by one. Between the comparison and these runs, the insurance score ranged from
9 to 30 correct with the same model, depending on the inferred schema: without the questions, each inference gave a
different design. The schema mattered far more than the choice among the Claude models.

## Why not the others

| Model | Verdict | Reason |
|---|---|---|
| Claude Opus 5.5 | Not worth the premium | As accurate as Sonnet on both tests, with slightly richer answers. But it was about 30% slower per question, the slowest at extraction because it reasons before answering, and it costs more per token. It also does not run in the app as-is (see below). |
| Claude Opus 4.5 | Weaker at questions | Extraction was as good as Sonnet's and a little faster. Question answering was clearly worse: 23 of 30 correct, including one confident wrong answer. It is the previous generation and costs more than Sonnet. |
| Claude Haiku 4.5 | Loses relationships | Not part of this benchmark. In an earlier extraction test on the SWIFT sample files it was twice as fast as Sonnet but found only about two thirds of the edges, without reporting anything as dropped. Missing edges are invisible to users and cannot be fixed at query time. |
| qwen2.5:14b, qwen3-coder (local Ollama) | Too inaccurate for the demo | Free to run and fully local, but qwen2.5:14b found 33 of 51 facts during ingestion and the two models answered 10 and 12 of 30 questions. qwen3-coder answers quickly, but its wrong answers sound confident, which is worse in a demo than an empty result. |

**Opus 5.5 known issue:** hugegraph-llm sends `temperature=0.01`, and Opus 5.5 accepts only `temperature=1`, so the
first call fails until the app tells LiteLLM to drop that setting (`litellm.drop_params = True`).

## When a local model still makes sense

Choose local qwen only when documents must not leave the machine or the demo must run offline. In that case, set
expectations that the graph will be less complete and that numeric and multi-hop questions will often fail.
Detailed extraction hints and a Text2Gremlin prompt that explains properties stored on edges and edge direction are
the most promising improvements.

## What this test does not show

- The model comparison covers one dataset, insurance fraud. Banking and semiconductor were run with Sonnet only.
- The comparison predates the current pipeline. The other models have not been run with the questions box, so their
  scores may be higher now; the qwen failures (values read from the wrong place, edges walked backwards) are in
  query writing, which the questions do not change.
- The checked answers were first produced with Claude, which may favour Claude slightly. The qwen failures do not
  depend on this: they were empty results, query errors and wrong people.
- Sonnet and Opus both score close to the maximum, so this test cannot show small differences between them. A
  harder, hand-checked fact list would be needed for that.
- Costs are described relative to each other. We did not check the prices in the gateway contract.

## Recommended settings

With the Cloudera gateway, the app picks up Sonnet from the existing gateway variables. Extraction and
Text2Gremlin fall back to the chat model when their own variables are left blank.

```bash
LLM_PROVIDER=claude
ANTHROPIC_BASE_URL=<gateway URL>
ANTHROPIC_AUTH_TOKEN=<gateway token>
ANTHROPIC_DEFAULT_SONNET_MODEL=claude-sonnet   # chat, extraction, Text2Gremlin
OLLAMA_EMBEDDING_MODEL=nomic-embed-text        # embeddings stay on local Ollama
```

Keep Ollama running with `nomic-embed-text` pulled, even with Claude. Text2Gremlin loads an embedding model on every
question.

---

*Benchmark scripts and raw results are in [`benchmarks/`](../benchmarks/README.md). The model comparison ran
against the insurance graph loaded in HugeGraph on 27 Oct 2026; the Sonnet runs on all three datasets used imports of
29 Oct 2026.*
