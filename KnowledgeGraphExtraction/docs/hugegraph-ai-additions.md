# Could hugegraph-ai do this alone?

`graphrag_showcase.py` is a dataset-agnostic text-to-knowledge-graph app on top of hugegraph-ai: it infers a
schema from your own sample documents and the questions the graph should answer, extracts every document against
that schema, and answers questions with Text2Gremlin. This page lists what it takes from hugegraph-ai, what it adds
on top, what we tried and removed, and which additions would be worth contributing back. The examples come from the
three datasets we tested it on: DOJ bank fraud and insurance fraud press releases (27 files each) and eight
Wikipedia articles on semiconductor companies.

*Checked against the local hugegraph-ai checkout (hugegraph-llm), HugeGraph server 1.8.0, Claude Sonnet via the
Cloudera AI Gateway, 29 Oct 2026.*

## Short answer: partly

hugegraph-ai provides the whole core pipeline, and the showcase calls it at every step. On the test datasets,
though, running hugegraph-ai alone would have hit almost every failure we fixed:

- Prompt problems can be fixed by hand in hugegraph-ai's own UI, by editing the extraction and Text2Gremlin prompts.
- hugegraph-ai's schema prompt already has a slot for example questions; the showcase is what fills it.
- A few gaps are covered elsewhere in the HugeGraph ecosystem: Hubble for property indexes and graph visualization.
- Cleaning the extracted data before import, recovering from failed extractions, failed queries and stalled LLM
  calls, the schema guards and follow-up questions all needed new code.

## Size of the layer we added

| | |
|---|---|
| `graphrag_showcase.py` | 1,910 lines |
| `agent_log.py` (call log panel) | 229 lines |
| Unit tests | 136, all passing |
| Runtime patches to hugegraph-llm classes | 6, on 4 methods, in memory only |

The patches add a timeout and retry to `LiteLLMClient.generate`, make `GremlinGenerateSynthesize._extract_response`
run the last query in a reply, wrap `PropertyGraphExtract.extract_property_graph_by_llm` three times (prompt
caching, the split retry and the recorder), and replace `PropertyGraphExtract.run` to extract chunks in parallel.
Nothing in the hugegraph-ai checkout is modified (`git status` is clean).

## Stage by stage

| Stage | From hugegraph-ai | Added by the showcase |
|---|---|---|
| **Schema inference** | `BUILD_SCHEMA` flow (`SchemaBuilder`) with its few-shot schema and its "Query Examples" slot; `CheckSchema` fills in keys and property keys | Excerpts from up to 6 sample files within one character budget; design rules (keys every document has, direct edges, INT or DOUBLE for quantities, DATE for dates); the user's questions sent in the Query Examples slot; up to three attempts when the reply is malformed or types every property TEXT, with a warning if every attempt does |
| **Schema creation** | `Commit2Graph.init_schema_if_need` | Range and secondary indexes on every non-key vertex property and every edge property; refuses a schema that conflicts with the one already in the graph; lists labels left over from an earlier schema |
| **Extraction** | `GRAPH_EXTRACT` flow and `PropertyGraphExtract`; default extraction prompt, `filter_item`; `read_documents` for .txt, .pdf and .docx | Chunks packed at blank lines (1,200 characters for Ollama, 6,000 for hosted models); the user's questions and an "always include properties" rule in the prompt; the fixed instructions cached; output limit raised to 16,384 tokens; 12 chunks at a time instead of one after another; a cut-off reply re-extracted in halves, split at line breaks for PDF text |
| **Cleanup before import** | None: hugegraph-llm passes extracted items straight to the importer | Convert values to schema types ("$4.3 million", "approximately $373,000", "one count", "July 14, 2026", "Sept. 9, 2026"); merge names that differ only in letter case; drop null values and edges to unknown vertices |
| **Import and report** | `IMPORT_GRAPH_DATA` flow | Files extracted in parallel, imported in upload order; per-file report of what the schema dropped; warning for schema labels that received nothing |
| **Questions** | `TEXT2GREMLIN` flow and Gremlin prompt; `run_gremlin_query` | HugeGraph rules in the prompt (labels, edge direction, partial name matching, return the values asked about, keep rows with missing properties, dates as ranges, a single top item for "most" questions); "complex query" section removed; no few-shot examples; the last query in a reply is run; empty result retried with case-insensitive partial matches; streamed LLM summary; question cards and follow-up rewriting from the last 3 answers |
| **LLM calls** | `LiteLLMClient`, with a retry for rate limits | A 300-second timeout on every call; timeouts, dropped connections and server errors retried twice |

## Could hugegraph-ai do it alone?

- **By hand**: possible through hugegraph-ai's UI or settings
- **Partly**: a workaround exists but leaves gaps
- **No**: needs new code

### Building the graph

| Addition | Problem it solved on the test data | hugegraph-ai alone |
|---|---|---|
| The user's questions sent to schema inference | With no stated purpose, the same prompt gave a different schema on every run: three runs on the same insurance samples gave three schemas, one without spouses, one without fraud amounts. Insurance fell from 30 correct to 9 correct of 28 asked. With the questions, three runs kept 12, 12 and 11 of the 12 facts the questions need, and the answers rose to 23 correct. | **By hand.** The Query Examples slot exists, but the UI passes nothing to it. |
| The same questions sent to extraction | A prosecutor credited only in a release's closing line was not linked to the defendants, and an unnamed victim got no vertex. With the questions, Joseph Baldwin was linked to all five officers and insurance rose from 23 to 25 correct. | **By hand.** Paste them into the extraction prompt. |
| Multi-file schema inference with design rules | A one-file schema keyed a `case` label on `case_number`, which 1 of 27 releases contains. Most edges came out empty. | **Partly.** The schema box takes pasted text, so excerpts can be combined by hand. The prompt has no design rules. |
| Re-infer a schema without types | One reply left out `propertykeys`, `CheckSchema` typed every property TEXT, and 10 of 30 banking questions failed with "expect TEXT for 'age'" (17 correct instead of 29). | **No.** The schema is accepted as valid. |
| Document-sized chunks | A chunk saying only "Luneke" created a second person next to "Aaron T. Luneke", splitting his facts. | **Partly.** The default is 500-character paragraph chunks, which makes this worse. The `document` split type avoids it, but nothing recovers when a reply is cut off. |
| Output limit of 16,384 tokens | hugegraph-llm's default for LiteLLM extraction is 256 tokens. One banking reply used 10,136. | **By hand.** Set the extraction token limit in the config. |
| Retry a cut-off chunk in pieces | One dense release hit the 8,192 output-token limit and was lost entirely; as a PDF it could not be split at all, because PDF text has no blank lines. | **No** |
| Parallel extraction | Chunks were extracted one after another, so ingest was slow on hosted models. | **No.** Needs the runtime patch. |
| Prompt caching | Every extraction call re-sent the same ~1,500 tokens of instructions. With caching, 36,000 of 98,000 input tokens on a 27-document import came from the cache: about a third off input, 10% off the import. | **No.** Needs the runtime patch. |
| Convert values to schema types | A whole-number value for a DOUBLE property made the importer skip the whole vertex. Every edge to it then failed the file with `Invalid vertex id`. | **No.** The importer only logs "skip it". |
| Case-duplicate, null and dangling-edge cleanup | HugeGraph rejecting an edge aborted the rest of the file; "JAMES KEATING" and "James Keating" became two people. The case merge joined 16 names in one banking import. | **No** |
| Edge-properties rule | hugegraph-llm discards an edge without a `properties` key, so every edge of a label with no properties was lost. | **By hand.** Edit the extraction prompt in the UI, or use `PROMPT_GENERATE`. |
| Vertex and edge property indexes | Without them, banking fell from 29 to 20 correct: every query that starts by filtering on a property, such as `g.V().hasLabel('defendant').has('age', P.gt(50))`, was refused. | **Partly.** Not in hugegraph-ai. Create them yourself with the HugeGraph schema API or Hubble. |
| Schema conflict guard | Creating a new schema without a reset mixed two schemas, then every import failed with `Undefined link of edge label`. | **No.** Existing labels are kept silently. |
| Reset removes the schema too | On an hstore cluster, clearing the graph kept every label, because hstore keeps the schema in PD; the next schema then hit the conflict guard. Reset now deletes the leftover labels and property keys. | **No.** `clean_hg_data` clears only the data on hstore. |
| Per-file drop report and empty-label warning | Lost data was only visible in log files. | **No** |

### Answering questions

| Addition | Problem it solved on the test data | hugegraph-ai alone |
|---|---|---|
| Remove the "complex query" prompt section | Multi-hop questions were answered with `g.V().limit(0)`. | **By hand.** Edit the Gremlin prompt in the Text2Gremlin tab. |
| HugeGraph rules: labels, edge direction, partial name matching | Queries without `hasLabel` were rejected; edges walked the wrong way. | **By hand.** Same prompt editor. |
| Rule: return the values a question is about | "Sentenced to more than 48 months?" listed names without the months; one query returned `[57, 10, 51]` with no names. | **By hand.** Same prompt editor. |
| Rule: keep rows whose property is missing | `by('role')` silently dropped organizations without a role: 2 of 5 were returned. | **By hand.** Same prompt editor. |
| Rule: compare dates as ranges | `TextP.containing('2027')` on a DATE property was rejected by HugeGraph. | **By hand.** Same prompt editor. |
| Rule: one answer for "most" questions | "Which company has the most founders?" returned a ranking of 10 instead of AMD (8). A ranking is still returned when asked for ("top 3"). | **By hand.** Same prompt editor. |
| Run the last query in a reply | "Which prosecutor handled the most defendants?" named a defendant: the model wrote a reversed query, corrected itself in the same reply, and the first query ran. | **No.** Needs the runtime patch. |
| Rewrite text filters for servers older than 1.8 | On a HugeGraph cluster older than 1.8, every partial name match failed with "Unsupported predicate: 'containing(Michael A. Shipp)'": those servers hand a `TextP` inside `has(...)` to the storage backend. The app rewrites it as `filter(__.values(...).is(TextP...))`, evaluated in memory. | **No.** Upgrade the server to 1.8, or patch the query. |
| Case-insensitive partial-match retry | "TD Bank" matched nothing because the graph stores "TD Bank, N.A."; "Bank Fraud" and "James Keating" missed "bank fraud" and "JAMES KEATING". Servers older than 1.8 have no `TextP.regex` ("No signature of method: TextP.regex()"), so there the retry ORs `TextP.containing` over the text as written, in lower and upper case and capitalized. | **No** |
| LLM summary of results | Text2Gremlin returns raw JSON. | **Partly.** The RAG answer flows write answers, but by graph retrieval rather than from the generated query's results. |
| Question cards and follow-ups | "Which judge tried him?" had no one to refer to. | **No.** The demo has no conversation history. |

### Running and debugging

| Addition | Problem it solved | hugegraph-ai alone |
|---|---|---|
| Timeout and retry on LLM calls | One gateway call hung for 28 minutes before failing with "Server disconnected". LiteLLM's default timeout is 6,000 seconds, and hugegraph-llm retries rate limits but not timeouts or dropped connections. | **No.** Needs the runtime patch. |
| Provider per role in the UI (Ollama, Claude, OpenAI) | Changing models without editing files. | **By hand.** The config tab does this, saved to `.env`. |
| Graph overview and visualization | Seeing what was extracted. | **Partly.** Available in Hubble, not in hugegraph-ai. |
| Agent log panel | Seeing every LLM, embedding and HugeGraph call with timings. | **No.** Log files only. |

## Tried and removed

| Addition | Why it was removed |
|---|---|
| Domain hints box | A domain label ("This is an insurance fraud dataset.") kept 11, 10 and 7 of 12 needed facts, and the existing hints file 8, 7 and 6. The questions box replaced it. |
| Infer three schema drafts and merge them with an LLM | Insurance scored 12 correct, 12 partly correct and 6 wrong: a merge cannot add a relationship that no draft proposed. |
| Few-shot query examples generated from the schema | They need the Ollama embedding model for their index. It was never running, so all 181 recent queries had none, and the scores were unaffected. |
| Folding surname-only names into full names | With document-sized chunks a document's full name and its surname-only mentions reach the model together; the last three imports had nothing to fold. |

## What failed without the additions

These are real failures from our ingest and question runs. Each one is silent or misleading in hugegraph-ai on its
own.

**Vertices skipped over a number type.** The model wrote `931500` for a DOUBLE. The importer requires a Python
float, skipped the person, and the next edge failed the file. Six files were lost this way in one run.

```text
Property type/format 'fraud_amount' is not correct, skip it
Invalid vertex id '1:Mercedes Henry'
```

**One person, two vertices.** Only the first chunk of the release used the full name, so the bank, prosecutors and
agencies attached to a separate vertex.

```text
chunk 0: Aaron T. Luneke -> charged_with bank fraud
chunk 1: Luneke -> defrauded Bank of the Valley
chunk 2: FBI, FDIC-OIG, ... -> investigated Luneke
```

**Filters refused.** No index existed for the property, so HugeGraph refused the query instead of scanning.

```text
g.V().hasLabel('defendant').has('age', P.gt(50))
Don't accept query based on properties [age] that are not indexed in label 'defendant',
may not match secondary/range condition
```

**Everything stored as text.** A schema reply left out `propertykeys`, so every property was typed TEXT and number
and date filters were rejected.

```text
propertykeys: amount TEXT, age TEXT, max_prison_years TEXT, sentencing_date TEXT
has('age', P.gt(50))   -> Invalid data type of query value '50', expect TEXT for 'age'
```

**Dates searched as text.** On a DATE property the model wrote a text match, which HugeGraph rejects.

```text
has('sentencing_date', TextP.containing('2027'))
Invalid property value '2027' for key 'sentencing_date': Expected date format is: [yyyy-MM-dd, ...]
```

**Mixed schemas.** A new schema was created on top of an old one. The old `charged_with` definition stayed, and
every document with a charge failed.

```text
Undefined link of edge label 'charged_with': 'defendant' -> 'charge'
```

**Cut-off extraction.** An eight-defendant release used the full output budget and returned broken JSON.
hugegraph-llm drops the chunk; imported as PDF, the release could not be split and was lost with all eight
defendants.

```text
completion_tokens=8192
1 chunk(s) without valid graph JSON
```

**The rejected query ran.** The model corrected itself in the same reply, and hugegraph-llm took the first block.

```text
g.V().hasLabel('person').in('prosecuted_by')...     <- run: names a defendant
Wait - the edge goes from the defendant (source) to the prosecutor (target). Corrected query:
g.V().hasLabel('person').where(__.in('prosecuted_by'))...   <- correct: Benjamin D. Bleiberg
```

**A call that never returned.** One Text2Gremlin call waited 1,712 seconds before the gateway dropped it.

```text
Text2Gremlin failed: litellm.InternalServerError: AnthropicException - Server disconnected without sending a response.
```

**Text filters refused by an older server.** On a cluster older than HugeGraph 1.8, a partial name match was
handed to the storage backend.

```text
g.V().hasLabel('judge').has('name', TextP.containing('Michael A. Shipp'))
Unsupported predicate: 'containing(Michael A. Shipp)'
```

**Exact match on a name.** The question used the everyday name or other letter case; the graph stores the legal
name or capitals.

```text
has('name', 'TD Bank')                                      -> []
has('name', TextP.regex('(?i).*\\QTD Bank\\E.*'))           -> Cheungkin Lam, Edward Low
```

## Tested on three datasets

Each dataset was ingested through the app with a schema inferred from its own documents and its
`SCHEMA QUESTIONS.md`, and every question in its questions file was then asked through the app's own pipeline
(Text2Gremlin, retry, summary, follow-up rewriting) with Claude. Answers were graded by hand against the questions
file and the source documents.

| Dataset | Inferred labels | Questions | Correct | Partly correct | Wrong |
|---|---|---|---|---|---|
| DOJ insurance fraud (27 releases) | `person`, `organization`; edges for sentencing (months, restitution), spouses, co-conspirators, prosecutors, fraud (amount, charge) | 30 | 28 | 2 | 0 |
| DOJ bank fraud (26 of 27 releases) | `defendant`, `judge`, `crime` (maximum prison years), `bank`, `prosecutor`, `agency` | 30 | 29 | 1 | 0 |
| Semiconductor companies (8 Wikipedia articles) | `company`, `person`, `product`, `city` (state, country) | 36 | 32 | 4 | 0 |

These runs used imports of 29 Oct 2026 without the safety nets (schema retries, conflict guard, split retry, null
and dangling-edge cleanup, partial-name retry): none of the failures they guard against occurred, and the scores
matched or beat the runs with them. The safety nets stay because each failure did happen earlier. One more banking
import without property indexes scored 20 correct, 2 partly correct and 8 wrong.

Every run behind these numbers, from the 27 Oct model comparison to the runs above, is listed with its answers file
and score in [benchmarks/README.md](../benchmarks/README.md).

The remaining partial answers come from extraction, not the pipeline:

- Charges are stored only on the `defrauded` edge, so insurance wire-fraud counts are low (#19, #27).
- Aggravated identity theft's two-year mandatory minimum was stored as its maximum (banking #18).
- ASM International has no headquarters edge, Nvidia lost its founding year, an acquisition year was misread, and
  generic products such as "CPU" were extracted (semiconductor #16, #17, #20, #21).

Spot checks against the sources match: James Wilson's "12 years" stored as 144 months, Nvidia founded by Jensen
Huang, Chris Malachowsky and Curtis Priem, Broadcom acquiring VMware for $69 billion, AMD founded by Jerry Sanders
and seven Fairchild colleagues. Each "most" question returns one answer with its count (AMD, 8 founders).

The questions files were written from a graph of 26 Oct and missed facts later graphs found; five banking answers,
one banking note and the Avago answers in semiconductor were corrected against the source texts.

## Worth contributing to hugegraph-ai

These fix behaviour that is arguably a bug upstream rather than a showcase convenience. Each would let us delete
code from the showcase.

1. **Convert property types in `Commit2Graph`** instead of skipping the vertex: whole numbers for FLOAT and DOUBLE,
   amounts and dates as written in documents.
2. **Keep importing when one edge fails**, and report skipped items instead of only logging them.
3. **Detect schema conflicts in `init_schema_if_need`** rather than silently keeping existing label definitions.
4. **Create property indexes** when the schema is created, so range filters work out of the box.
5. **Reject or retry a schema reply without `propertykeys`** instead of typing every property TEXT.
6. **Extract chunks in parallel in `PropertyGraphExtract.run`**, which would remove our runtime patch.
7. **Set a request timeout and retry transient errors in `LiteLLMClient`**, not only rate limits.
8. **Take the last query block in `GremlinGenerateSynthesize`**, so a corrected query is the one that runs.
9. **Ship HugeGraph rules in the default Text2Gremlin prompt**: scope lookups with `hasLabel`, compare dates as
   ranges, use `coalesce` for optional properties, return the values a question filters on, return one top item
   for "most" questions.
10. **Expose the schema prompt's Query Examples in the UI**, so users can say what the graph is for.
11. **Rewrite `has(key, TextP...)` for servers older than 1.8 in `run_gremlin_query`**, or document that text
    filters need HugeGraph 1.8. Those servers also have no `TextP.regex`.

## Still open in the showcase

- **Name variants remain separate**, within and across documents: "Julia" and "Julie E. Kobick", "FBI" and
  "Federal Bureau of Investigation", "AMD" and "Advanced Micro Devices" in some imports, "Maureen", "Maureen Wilson"
  and "Maureen Ann Wilson". This is entity resolution, the main gap on any dataset; it needs an alias table or a
  merge step after import.
- **Straight and curly apostrophes** make two vertices ("Conrad D'Haiti" and "Conrad D’Haiti"); the case merge
  could treat them as the same.
- **Renamed companies are not linked**: Avago Technologies became Broadcom Inc., and imports file its acquisitions
  under either name. The semiconductor answers accept both.
- **Unnamed entities**: a victim called only "the home repair insurance provider" gets a vertex in some imports and
  not in others, so its fraud edge comes and goes.
- **Inconsistent values for the same thing**: product categories as "processor", "processors" and "CPU", so filters
  on them find only part of the answer.
- **Missing middle initials** ("Aaron Luneke") are not matched. hugegraph-ai's vertex-name embeddings
  (`UPDATE_VID_EMBEDDINGS`) could cover this.
- **Imports differ from run to run**: the questions keep the schema's relationships, but label names and some facts
  change between imports. For a benchmark that must repeat exactly, keep one schema and re-ingest against it.

---

*Source: `graphrag_showcase.py`, its tests, and the hugegraph-llm source in our local checkout. The press releases
name real people; charged defendants are presumed innocent. Treat the graph as demo data.*
