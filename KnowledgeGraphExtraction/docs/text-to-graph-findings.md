# HugeGraph Text-to-Graph Findings

What it took to build a knowledge graph automatically from unstructured press releases and articles, answer
questions over it, and where it still goes wrong. October 2026.

- **Graph:** Apache HugeGraph 1.8
- **LLM layer:** hugegraph-llm via LiteLLM
- **Model:** Claude Sonnet
- **App:** `graphrag_showcase.py` (Gradio)

## The pipeline

Every dataset goes through the same five steps in the showcase app. Nothing is hand-modelled: the schema, the
vertices and the edges all come from the LLM reading the text.

1. **Infer a schema.** Up to 6 sample documents, spread evenly over the upload, about 4,000 characters of each,
   plus the questions the graph should answer. The LLM proposes vertex labels, edge labels and typed properties; a
   reply without types is inferred again.
2. **Create the schema in HugeGraph.** Labels, property keys and range indexes, after a conflict check against
   anything already in the graph.
3. **Extract every document.** Documents are packed into chunks of up to 6,000 characters and extracted 12 at a
   time, with the same questions in the prompt. A reply cut off by the token limit is re-extracted in smaller pieces.
4. **Import.** Null values are dropped, values are converted to their schema types, names differing only in case
   are merged, and edges to unknown vertices are skipped.
5. **Ask questions.** Text2Gremlin writes a query from the live schema, HugeGraph runs it, and the LLM summarises
   the rows. Names are retried with a partial match, and follow-up questions are rewritten using recent answers.

## Datasets

Three folders, each with a hand-checked question file used as the benchmark and a `SCHEMA QUESTIONS.md` to paste
into the app.

| Dataset | Source | Documents | Questions | What the graph must hold |
|---|---|---|---|---|
| Banking fraud (`DOJ_fraud_samples/banking`) | U.S. Department of Justice press releases | 27 (.txt; PDFs also tested); the latest imports had 26 | 30 | Co-defendants, presiding judges, charges and plea status, maximum prison terms, sentencing dates, banks and amounts, prosecutors, agencies |
| Insurance fraud (`DOJ_fraud_samples/insurance`) | U.S. Department of Justice press releases | 27 | 30 | Ownership, employers, fraud victims with amounts and charges, spouses, co-conspirators, sentencing judges with months and restitution, prosecutors, agencies |
| Semiconductor companies (`WIKI_semiconductor_samples`) | Wikipedia articles | 8 | 36 | Founders, CEOs, headquarters (city, state, country), products and types, acquisitions and investments with year and amount, competitors |

Questions cover one-hop lookups, numeric and date filters, multi-hop paths, aggregations (most, top 3) and
follow-ups that refer back to an earlier answer.

## The core problem: the schema changed every time

Schema inference is one LLM call, and with reasoning on, the same prompt gives a different design on every run.
Three inferences on the same six insurance samples produced three different schemas: one had no spouse
relationship, one stored no amount on fraud, and one modelled judges as a generic `official`.

Each design was reasonable on its own, but each answered a different subset of the questions. The same 30 insurance
questions went from 30 correct on the schema of 26 October to 9 correct, 7 partly correct and 12 wrong on a schema
re-inferred two days later.

**The data was never the problem.** Judges appear in 17 of the 27 insurance releases, prosecutors' names in 19 and
spouses in 7, and the sample documents fit whole in the prompt. The prompt simply never said what the graph was
for: hugegraph-llm's schema prompt has a slot for example questions, and the app was sending it empty. The bundled
few-shot example is also tiny, with two vertex labels (`person`, `webpage`) and two edges.

Without a purpose, the model picked among equally valid designs, for example storing a sentence as
`sentence_months` on the person with no edge to the judge, so judges' names had nowhere to go.

## Approaches we tried

Measured on the insurance dataset. "Concepts" is how many of the 12 facts the questions need (spouse, judge,
sentence months, prosecutor names and so on) made it into each inferred schema, over three runs.

| Approach | Result | Decision |
|---|---|---|
| No guidance, schema inferred once | 3 runs, 3 different schemas; Q&A 9 correct, 7 partial, 12 wrong (28 questions) | Baseline |
| Domain label: *"This is an insurance fraud dataset."* | Concepts 11, 10, 7 of 12; spouses missing every time | Dropped |
| Existing extraction hints file | Concepts 8, 7, 6 of 12; it tells the model to link prosecutions to the office | Dropped |
| Checklist of required relationships | Concepts 12, 12, 12 of 12 | Works, but we design the schema for the model |
| Infer 3 drafts in parallel, merge with an LLM | Q&A 12 correct, 12 partial, 6 wrong; a merge cannot add what no draft proposed | Removed |
| **Example questions sent to schema inference** | Concepts 12, 12, 11 of 12; Q&A 23 correct, 7 partial, 0 wrong | **Adopted** |
| **The same questions also sent to extraction** | Q&A 25 correct, 5 partial, 0 wrong; 28, 2, 0 in the next import | **Adopted** |

### The approach we kept: tell it what the graph is for

The app's old hints box was replaced by **Questions this graph should answer**. The user types the questions the
graph must support, one per line, and never names a label or relationship. The model still designs the schema from
the documents; the questions only state the goal, which is how a real project starts.

- **Schema inference** receives them in hugegraph-llm's "Query Examples" slot.
- **Extraction** receives them as a note: *"The graph built from these documents must answer the questions below.
  Extract every fact they need."* This recovered links the model used to skip, such as a prosecutor credited only
  in a release's closing line.
- The questions are general ("Which judge sentenced each defendant, to how many months, and with how much
  restitution?"), so they do not copy the benchmark questions.

The insurance `SCHEMA QUESTIONS.md`, as pasted into the app:

```text
Which judge sentenced each defendant, to how many months, and with how much restitution?
Who is each defendant's spouse?
Who conspired with whom?
Which prosecutors (people) prosecuted each defendant, and which agencies investigated them?
Which organization did each defendant defraud, for how much, and on what charge?
Which companies did each defendant own, and where were they employed?
How old is each person, and what is their occupation?
```

## Extraction: what we added on top of hugegraph-ai

hugegraph-ai supplies the extraction flow (`GRAPH_EXTRACT` and `PropertyGraphExtract`), its default prompt, a filter
that drops items outside the schema, and the importer. Used as is, it splits text into 500-character paragraphs,
extracts one chunk at a time, and passes whatever the LLM returns straight to HugeGraph. The showcase wraps that
flow at run time without changing the hugegraph-ai code, and adds four layers.

### What goes into the prompt

| Addition | Why it mattered on our data |
|---|---|
| **Document-sized chunks**: up to 6,000 characters for hosted models (1,200 for Ollama), packed at blank lines so a record is never cut | A chunk that said only "Luneke" created a second person next to "Aaron T. Luneke" and split his facts between the two |
| **Edge-properties rule**: every edge must carry a `properties` key, even an empty one | hugegraph-llm silently discards an edge without it, so every edge of a label with no properties was lost |
| **The schema questions as extraction notes**: "The graph built from these documents must answer the questions below. Extract every fact they need." | Recovered Joseph Baldwin's links to all five officers, Belletieri's $6M restitution, and (in some runs) Decristofaro's unnamed victim |
| **Cached instructions**: the fixed part of the prompt is marked for caching and written once before the parallel calls | Same answers at lower cost: about a third off input tokens per import |

### Recovering from failed calls

| Addition | Why it mattered on our data |
|---|---|
| **Output limit raised** from hugegraph-llm's 256 tokens to 16,384 | A chunk's graph JSON runs to thousands of tokens; the eight-defendant release used all of 8,192 |
| **Split retry**: a reply with no valid JSON is re-extracted in halves, recursively, and the pieces are combined; PDF text without blank lines is split at line breaks | One cut-off reply used to drop the whole document |
| **Parallel extraction**: 12 calls at a time across all documents of an import | Chunks were extracted one after another, which made ingest slow on hosted models |

### Cleaning the output before import

hugegraph-ai has no step between the LLM's reply and the importer. These run on every document, in this order:

| Addition | Why it mattered on our data |
|---|---|
| **Drop null values** | Null properties were sent to HugeGraph as values |
| **Convert values to schema types**: "$4.3 million" to 4300000, "approximately $373,000" to 373000, "one count" to 1, "Sept. 9, 2026" to a date | A wrong type made the importer skip the whole vertex, and every edge to it then failed the file with `Invalid vertex id`; number and date filters only work on typed values |
| **Merge names that differ only in letter case**, preferring the spelling already in the graph | "JAMES KEATING" and "James Keating" became two vertices |
| **Skip edges to vertices that do not exist** | One rejected edge made HugeGraph abort the rest of the file |

Surname folding ("Luneke" into "Aaron T. Luneke") was removed: with document-sized chunks a document's full name
and its surname-only mentions reach the model together, and the last three imports had nothing to fold. Case
merging stayed: it merged 16 names in the latest banking import.

### Safety nets and property indexes

Three attempts at schema inference, the schema conflict check, the split retry, dropping null values, skipping edges
to missing vertices and the partial-name retry guard against rare failures. One import of each dataset was run
without them: none of those failures happened, and the scores held (insurance 28 correct of 30, banking 29 of 30,
semiconductor 32 of 36, none wrong). They are kept because each failure did happen earlier in the project: a cut-off
PDF reply, a schema created over an old one, a file aborted by one bad edge.

Property indexes are needed. Without them, banking fell from 29 to 20 correct: HugeGraph refuses a query that starts
by filtering on a property, such as `g.V().hasLabel('defendant').has('age', P.gt(50))`, with "Don't accept query
based on properties [age] that are not indexed". Filters reached through another step still worked, for example
judges whose defendants are older than 50.

### Showing what was lost

- Each file's import report lists what the schema dropped (labels or properties the LLM used that the schema lacks)
  and how many names were merged.
- A warning names any schema label that received nothing, which usually means a vertex was keyed on a property most
  documents lack.

The full stage-by-stage list, including the question-answering side, is in
[hugegraph-ai-additions.md](hugegraph-ai-additions.md).

## Pipeline fixes along the way

| Problem | Cause | Fix |
|---|---|---|
| A whole banking release vanished from the graph when imported as PDF, taking 8 defendants and most of Judge Shipp's cases with it | PDF text has no blank lines, so the retry that splits an over-long reply found nothing to split at and gave up silently | The retry now falls back to splitting at line breaks |
| Dense documents were cut off mid-JSON | 8 defendants × 7 agencies of edges, plus reasoning, filled the 8,192-token output limit | Extraction limit raised to 16,384 tokens |
| Every extraction call re-sent the same ~1,500 tokens of instructions | No prompt caching | The fixed instructions are cached; on a 27-document import, 36,000 of 98,000 input tokens came from the cache, about a third off input and 10% off the import |
| 10 of 30 banking questions failed with "expect TEXT for 'age'" | The schema reply left out `propertykeys`, and hugegraph-llm typed every property TEXT | A schema with every property TEXT counts as a failed attempt and is inferred again; the summary warns if every attempt is |
| A benchmark run stopped for 28 minutes on one question | LiteLLM's default timeout is 6,000 seconds, and hugegraph-llm retries rate limits but not timeouts or dropped connections | Every LLM call times out after 300 seconds, and timeouts and dropped connections are retried twice |
| "Which prosecutor handled the most defendants?" named a defendant | The model wrote a reversed query, corrected itself in the same reply, and the app ran the first query | The app now runs the last query in a reply |
| Correct answers were graded as wrong | The answer files were written from one graph and missed facts a later graph found | Answers were re-checked against the source texts: five banking answers and the Avago questions in semiconductor were updated |

## Where it stands

Latest benchmark per dataset, graded by hand against the answer files and the source documents.

| Dataset | Questions | Correct | Partly correct | Wrong |
|---|---|---|---|---|
| Banking | 30 | 29 | 1 | 0 |
| Insurance | 30 | 28 | 2 | 0 |
| Semiconductor | 36 | 32 | 4 | 0 |

All three are imports of 29 October, each inferred from that dataset's `SCHEMA QUESTIONS.md`. Semiconductor is
scored on the answer file that accepts either name for Avago's acquisitions.

| Dataset | What is still partly correct |
|---|---|
| Banking | #18: aggravated identity theft's two-year minimum stored as its maximum, and two union crimes without a maximum |
| Insurance | #19, #27: charges are stored only on fraud edges, so wire-fraud counts are low |
| Semiconductor | #16 ASM International without headquarters, #17 Nvidia without a founding year, #20 an acquisition year misread, #21 generic products such as "CPU" |

The scores moved a lot on the way here. Insurance went from 9 correct (schema inferred without questions) to 23
(questions sent to schema inference) to 25 and then 28 (questions also sent to extraction). Banking scored 17 in one
import whose schema had no types, and 29 once types were checked.

Every run behind these numbers is listed with its answers file and score in
[benchmarks/README.md](../benchmarks/README.md).

## Limitations: why some answers are still wrong

None of the remaining errors come from the import pipeline. They come from how the source text is written, how names
vary, and the fact that an LLM does not give the same output twice.

### The text itself is ambiguous

- **Unnamed entities.** Both releases about Christian Decristofaro call his victim only "the home repair insurance
  provider". Graph vertices are keyed by name, so a nameless victim often gets no vertex and its fraud edge is lost.
- **Unclear scope.** "Assistant U.S. Attorney Joseph Baldwin … who is prosecuting the case" does not say which of
  five officers "the case" covers.
- **Legal nuance.** Aggravated identity theft carries a mandatory *minimum* of two years; the graph stored it as the
  maximum.

### One thing, several names

- **Variants:** "AMD" and "Advanced Micro Devices" as two companies; "FBI" and "Federal Bureau of Investigation";
  "Conrad D'Haiti" with a straight and a curly apostrophe; Judge "Julia" and "Julie" E. Kobick.
- **Renames:** Avago Technologies bought Broadcom Corporation and became Broadcom Inc. Some imports file its
  acquisitions under Avago, others under Broadcom Inc., so a question about one name misses the other.
- This is entity resolution, the classic hard problem in knowledge graphs. Fixing it needs an alias table, a
  "renamed to" relationship or a merge step after import.

### Schema choices limit answers

- Insurance charges were stored only on the `defrauded` edge, so people without a known victim have no recorded
  charge and "Which charge appears most often?" undercounts.
- A detail the questions never mention may be left out: restitution was missed once, and US states only appeared
  after the headquarters question asked for them.
- A schema reply can leave out the property types. In one banking import every property became TEXT, and 10 of 30
  questions failed on age, amount and date filters (17 correct instead of 29). Schema inference now infers again when
  every property is TEXT, and warns if all attempts are.

### Runs differ

- The same release produced Decristofaro's fraud edge in 3 of 3 test runs, then not in one full import and again
  in the next.
- AMD had all 8 founders in one import, only Jerry Sanders in the next two (under two names, "AMD" and "Advanced
  Micro Devices"), and all 8 again in the latest; its co-founders are listed in a single semicolon-separated
  sentence.
- Avago Technologies is a separate company in some imports and part of Broadcom Inc. in others.
- For a benchmark that must repeat exactly, save the schema and re-use it rather than re-inferring it.

### Writing the query

- When both ends of an edge share a label (`person -[prosecuted_by]-> person`), only the edge's name says which end
  is which, and the model can reverse it.
- Queries are written without few-shot examples. Examples generated from the schema were tried, but they need the
  Ollama embedding model, were never available in the recent runs, and were removed.

## Running the demo

- Reset the graph, paste the dataset's `SCHEMA QUESTIONS.md` into **Questions this graph should answer**, click
  *Infer schema*, and check the summary shows the relationships you need before creating the graph.
- Leave the questions in the box for the ingest, and upload every document: one banking release
  (`12_…loan-broker…`) was left out of four imports.
- For a dependable script, use the reliable question sets (9 insurance, 17 banking), which every inferred schema so
  far has supported.
- Show one or two of the misses on purpose. Which ones appear depends on the import, so check the graph before
  the demo: "Who defrauded an organization of more than 1,000,000 dollars?" shows the unnamed victim when it is
  missing, "Who founded AMD?" shows one company under two names when AMD is split, and "Which companies did Avago
  Technologies or Broadcom acquire?" shows a renamed company. They explain the limits of building a graph from text
  better than a perfect run does.
