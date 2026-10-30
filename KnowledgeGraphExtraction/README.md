# Enterprise HugeGraph-AI & GraphRAG Demo

This project demonstrates a GraphRAG (Retrieval-Augmented Generation) application using **Apache HugeGraph** as the backend database and a local **Ollama** model for schema inference, data extraction, and natural language querying. No API key is needed by default; Claude or OpenAI can be used instead (see [Optional: Claude or OpenAI](#optional-claude-or-openai)).

It is designed to run against a locally cloned, editable version of the `hugegraph-ai` repository.

## Prerequisites
* **macOS** (Apple Silicon/Intel) with **Homebrew** (`brew --version`), or **Linux: RHEL 9** (or Rocky/Alma 9, x86_64)
  with `sudo` for `dnf`
* **Docker** or **Podman**, only for running a local HugeGraph server (not needed when using an existing server)
* **Ollama** (for the local LLM; installed in step 4)

Where the commands differ, each step shows a macOS block and a Linux (RHEL 9) block.

---

## 1. Clone the HugeGraph Source Code
Clone the official Apache HugeGraph AI repository anywhere you like, check out the tested commit, and point
`HUGEGRAPH_AI_HOME` at the clone. `requirements.txt` uses this variable to find the local packages when you run
`pip install` (step 3).

```bash
# Any location works; this is just an example
export HUGEGRAPH_AI_HOME="$HOME/src/hugegraph-ai"
git clone https://github.com/apache/hugegraph-ai.git "$HUGEGRAPH_AI_HOME"
git -C "$HUGEGRAPH_AI_HOME" checkout da11249745180471420890fd5a44aee93bc33bae   # tested commit, 2026-10-20
```

The demo and `constraints.txt` are tested against this commit. Later commits on `main` need Python 3.12 or newer
and add a `hugegraph-llm-sdk` package that `requirements.txt` does not install, so `pip install` fails on them.

To keep it for later, either put it in `connection.properties` in the demo folder (your own, git-ignored settings
file, copied from `connection.properties.example`; see step 5) as an absolute path:
```properties
HUGEGRAPH_AI_HOME=/home/you/src/hugegraph-ai
```
or export it from your shell profile (`~/.zshrc` on macOS, `~/.bashrc` on Linux):
```bash
echo "export HUGEGRAPH_AI_HOME=\"$HUGEGRAPH_AI_HOME\"" >> ~/.zshrc
```

## 2. Set Up the Python Environment
At the tested commit (step 1), the `hugegraph-llm` module strictly requires Python 3.10 or 3.11.

**macOS:** to avoid `tcl-tk` compilation errors, use Homebrew's pre-compiled Python 3.11.
```bash
brew install python@3.11
cd /path/to/HUGEGRAPH-AI-EXAMPLES          # the repo; the virtual environment lives at its top level
$(brew --prefix python@3.11)/bin/python3.11 -m venv .venv
```

**Linux (RHEL 9):** Python 3.11 is in the AppStream repository.
```bash
sudo dnf install -y python3.11 python3.11-pip python3.11-devel gcc git
cd /path/to/HUGEGRAPH-AI-EXAMPLES          # the repo; the virtual environment lives at its top level
python3.11 -m venv .venv
```

Then, on either platform, activate the virtual environment and work from the demo folder:
```bash
source .venv/bin/activate
cd KnowledgeGraphExtraction
```
*(You should see `(.venv)` in your terminal prompt. Verify by running `python --version`, which should output `3.11.x`)*

## 3. Install Dependencies
`requirements.txt` installs the web UI, LiteLLM, and the two packages from your clone as editable installs,
located through `HUGEGRAPH_AI_HOME` (pip expands `${VAR}` in requirements files):

```text
# Web UI framework
gradio>=4.0.0

# Claude / OpenAI access (optional providers)
litellm>=1.0.0

# Editable local packages from the cloned repo; set HUGEGRAPH_AI_HOME to wherever you cloned it, checked out at
# the tested commit da11249 (README step 1)
-e ${HUGEGRAPH_AI_HOME}/hugegraph-python-client
-e ${HUGEGRAPH_AI_HOME}/hugegraph-llm
```

Install the requirements. pip reads only environment variables, so when `HUGEGRAPH_AI_HOME` is not exported, the
first line takes it from `connection.properties`:
```bash
export HUGEGRAPH_AI_HOME="${HUGEGRAPH_AI_HOME:-$(awk -F= '$1 ~ /^[ \t]*HUGEGRAPH_AI_HOME[ \t]*$/ {sub(/^[^=]*=[ \t]*/, ""); sub(/[ \t]*$/, ""); print; exit}' connection.properties 2>/dev/null)}"
echo "$HUGEGRAPH_AI_HOME"        # must print your clone's path; if empty, set it as in step 1
pip install -r requirements.txt -c constraints.txt
```

**Linux (RHEL 9) only:** hugegraph-llm pins `pycgraph==3.2.2`, whose Linux wheel was built with GCC 13 and fails to
load on RHEL 9's GCC 11 runtime (`` /lib64/libstdc++.so.6: version `GLIBCXX_3.4.32' not found ``). From 3.2.4 on,
the wheels need only `GLIBCXX_3.4.22`. Replace it with 3.2.5; pip then warns that hugegraph-llm requires 3.2.2,
which is safe to ignore:
```bash
pip install --no-deps --force-reinstall pycgraph==3.2.5
```

`constraints.txt` pins every dependency to a set of versions known to work together (checked on macOS, and for
Linux x86_64 wheels). hugegraph-llm leaves almost all of its dependencies unpinned, so without it pip starts from
the newest releases and, when they conflict, backtracks through hundreds of `gradio` and `litellm` versions,
downloading each one ("This is taking longer than usual"). To try newer versions, leave out `-c constraints.txt`;
after a working install, refresh the file with
`pip freeze --exclude-editable | grep -viE '^(hugegraph|pip|setuptools|wheel)(==| @)' > constraints.txt`.

## 4. Install Ollama and Pull the Models
**macOS:**
```bash
brew install ollama
brew services start ollama     # or run `ollama serve` in a separate terminal
```

**Linux (RHEL 9):** the install script sets Ollama up as a systemd service. It needs `zstd` to unpack the
download ("This version requires zstd for extraction"), which RHEL 9 does not install by default.
```bash
sudo dnf install -y zstd
curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl enable --now ollama
```
Without a GPU, the chat and extraction models below are slow on CPU. On such hosts, use Claude or OpenAI for chat
and extraction (see [Optional: Claude or OpenAI](#optional-claude-or-openai)) and pull only `nomic-embed-text`,
which runs fine on CPU.

Then pull the models:
```bash
#ollama pull qwen3-coder        # (Optional)chat / Text2Gremlin (30B MoE, ~18 GB; 8B models work but query poorly)
#ollama pull qwen2.5:14b        # (Optional) extraction (~9 GB)
ollama pull nomic-embed-text   # embeddings
```
Verify Ollama is up: `curl http://127.0.0.1:11434/api/tags` should list the models you pulled.

## 5. Run the HugeGraph Database
Either run a local HugeGraph server via Docker on port `8080` (on RHEL 9, `podman` takes the same arguments):
```bash
docker run -itd --name=hugegraph -p 8080:8080 -p 8182:8182 hugegraph/hugegraph
```

or use an existing HugeGraph server, such as one on a cluster. In the demo folder, copy the template and set the
address, graph, graph space and login in the copy:

```bash
cp connection.properties.example connection.properties
chmod 600 connection.properties      # it will hold passwords
```

```properties
HUGEGRAPH_URL=ccycloud-1.example.site:8080
HUGEGRAPH_GRAPH=hugegraph
HUGEGRAPH_GRAPHSPACE=
HUGEGRAPH_USER=admin
HUGEGRAPH_PASSWORD=<password>
```

`connection.properties.example` is committed and lists every key with its default; the app never reads it.
`connection.properties` is git-ignored and needs only the keys you change: keys left out use the defaults. It can
also hold the LLM settings (`LLM_PROVIDER`, `ANTHROPIC_BASE_URL`, `OLLAMA_HOST_IP`, ...), with the same names as
the environment variables. An exported environment variable wins over the file, so a stale `export` in your shell
or `~/.bashrc` silently overrides it; check with `env | grep -E 'HUGEGRAPH|ANTHROPIC|CLAUDE|OLLAMA|LLM_'`.

- **Graph space:** blank uses the server's default (`DEFAULT` on HugeGraph 1.7 and later). Set it when the graph
  lives in another graph space.
- **Graph:** if the server has no graph by that name, *Create graph with this schema* creates it. It copies the
  configuration of another graph in the same graph space, so it gets the same backend; in an empty graph space it
  uses `HUGEGRAPH_BACKEND` (`rocksdb` standalone, `hstore` on a cluster with PD; default `rocksdb`). Creating a
  graph needs a login with the graph space role on the server. Servers older than 1.7 have no graph spaces, and
  the graph must already exist there.
- **Reset** deletes all data and schema in the configured graph, so use a graph that no other work depends on.
  On an hstore cluster the server's clear call keeps the schema, so Reset also deletes the labels and property
  keys one by one.

## 6. Launch the Application
With Ollama and HugeGraph running and the `.venv` activated, start the GraphRAG showcase:

```bash
python graphrag_showcase.py    # http://127.0.0.1:8002  (use --host / --port to change)
```

**On a remote Linux host** (e.g. a cluster node close to HugeGraph), the app listens on that host's `127.0.0.1`.

1. Start it in `tmux` (`sudo dnf install -y tmux`), so it keeps running when your SSH session ends. Call the
   virtual environment's Python by path: a new tmux shell may not have the venv activated, and a bare `python`
   there can be another interpreter (`SyntaxError` on a type annotation, or `No module named 'gradio'`).
   ```bash
   tmux new -s graphrag -c /path/to/HUGEGRAPH-AI-EXAMPLES/KnowledgeGraphExtraction \
     '../.venv/bin/python graphrag_showcase.py; exec bash'
   ```
   With a gateway token, export it inside the quoted command, e.g.
   `'export ANTHROPIC_AUTH_TOKEN=$(cat ~/.config/anthropic_token); ../.venv/bin/python graphrag_showcase.py; exec bash'`.
   Detach with Ctrl+B then D; `tmux attach -t graphrag` shows the app's log again.
2. On your laptop, open an SSH tunnel and leave that terminal open. With `-N` it prints nothing while it works;
   `ExitOnForwardFailure` makes it exit with an error instead if the local port is taken:
   ```bash
   ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:8002:127.0.0.1:8002 user@rhel-host
   ```
3. Open `http://127.0.0.1:8002`. If it does not load, `curl -s -o /dev/null -w '%{http_code}\n'
   http://127.0.0.1:8002` should print `200` on the host (app running) and then on your laptop (tunnel working).

The UI has no login, so prefer the tunnel over `--host 0.0.0.0`. If you do bind to all interfaces, open the port
with `sudo firewall-cmd --add-port=8002/tcp` and restrict who can reach it.

The host needs outbound access to PyPI and the Ollama installer while setting up, and to the LLM provider while
running. LiteLLM's tokenizer (tiktoken) also downloads a file on first use; on hosts without internet access, copy
a filled cache from another machine and point `TIKTOKEN_CACHE_DIR` at it.

1. **Build tab** - upload a few sample documents, list the questions the graph should answer, infer and edit the
   schema, create it in HugeGraph, then ingest more documents of the same format.
   Put the questions in **Questions this graph should answer**, one per line (each dataset has a ready-made
   `SCHEMA QUESTIONS.md`), and leave them there for the ingest. Schema inference keeps the entities and
   relationships they need, and extraction keeps the facts they need; the design still comes from the documents.
   Without them, the same samples give a different schema on every run.
   Schema inference reads excerpts of up to 6 samples (spread evenly if you upload more) within one character budget
   (6,000 for Ollama, 24,000 for hosted models), and asks for a schema that fits all of them. A schema inferred from a
   single document tends to fit only that one, e.g. keying a label on a case number that other documents lack.
   Creating the schema also indexes every non-key vertex property and every edge property (range indexes for
   numbers and dates), so filters such as `has('age', gt(50))` or `g.E().hasLabel('defrauded').has('amount', gt(1000000))`
   are accepted.
   **Create graph** is refused if the schema clashes with one already in the graph (an edge label with other
   endpoints, a vertex label with another key or missing properties, a property with another type): HugeGraph
   would keep the old definitions and every import would fail. Use **Reset** to clear the graph first. Labels left
   over from an earlier schema are listed after a successful create.
2. **Explore tab** - label counts and a graph visualization, plus natural-language questions answered via Text2Gremlin and summarized by the LLM. The generated Gremlin and raw
   results appear as soon as the query runs; the summary (count plus up to 10 examples) streams in as it is written.

The ingest report flags, per file, what the extraction produced but the schema could not hold (labels or properties
not in the schema, edges with the wrong endpoint labels, edges to vertices missing from the chunk, replies without
valid JSON). hugegraph-llm drops these silently, so a long list usually means the document does not fit the schema.
It also lists schema labels that received nothing in the run, e.g. a vertex keyed on a case number most documents
lack, along with the edges that needed it; questions relying on those labels will return empty results.

Property values are also converted to their schema types before import (`931500` to `931500.0` for a DOUBLE;
`"$1,000,000"`, `"approximately $373,000"` or `"4.3 million"` to a number; `"one count"` to 1; `"July 14, 2026"`,
`"Sept. 9, 2026"` or `"2026-07-14 00:00:00"` to a DATE). This only helps when the schema gives such properties a
numeric or DATE type; with TEXT they are stored as written and number or date filters do not work. hugegraph-llm's importer skips a
whole vertex when one value has the wrong type, and every edge to it then fails the file with
`Invalid vertex id`. A value that cannot be converted is dropped on its own and listed in the ingest report.

A schema reply sometimes leaves out its `propertykeys`, and every property is then typed TEXT, so number and date
filters fail ("expect TEXT for 'age'"). Schema inference treats such a reply as a failed attempt and infers again; if
every attempt comes back all TEXT, the last schema is shown with a warning so you can infer again or set the types.

Every LLM call has a 300-second timeout (`LLM_TIMEOUT`, in seconds; LiteLLM's own default is 6,000), and a call that
times out or loses its connection is retried twice (`LLM_RETRIES`), after 1 and 2 seconds. Streamed summaries get the
timeout but no retry. Rate limits are retried by hugegraph-llm itself.

Some steps guard against failures that are rare but lose a file or an answer when they happen: three attempts at
schema inference, the schema conflict check, re-extracting a cut-off reply in pieces, dropping null values, skipping
edges to vertices that do not exist, and retrying an empty query result with partial name matches. They cost
nothing when a dataset does not need them.

Questions are answered by hugegraph-llm's Text2Gremlin, with no few-shot examples: hugegraph-llm's bundled ones are
about movies and companies, and the rules added to its prompt (below) cover what examples would teach.

The prompt asks for exact matches on people's full names and partial matches (`TextP.containing`) on
organization, agency and place names, which are often stored longer than asked (`TD Bank, N.A.` for "TD Bank").
It also asks queries to return the values a question filters on or ranks by (amounts, ages, counts, durations)
next to each name, so "Who was sentenced to more than 48 months?" lists the months as well.
If a query still returns nothing, its text filters, exact (`has('name', 'TD Bank')`) or `TextP.containing`, are
relaxed to a case-insensitive partial match and the query is run once more, so "TD Bank" finds "TD Bank, N.A." and
"James Keating" finds "JAMES KEATING"; the answer then says a partial match was used.

HugeGraph servers older than 1.8 reject a text filter inside `has(...)` ("Unsupported predicate:
'containing(TD Bank)'"), because they hand it to the storage backend. The app asks the server for its version once
and, for older servers, rewrites `has('name', TextP.containing('TD Bank'))` as
`filter(__.values('name').is(TextP.containing('TD Bank')))` before the query runs. The result is the same; the shown
Gremlin is the rewritten one.
Those servers also lack `TextP.regex`, so their partial-match retry tries the text as written, in lower and
upper case and capitalized, with `TextP.containing`.

Answered questions stay on the page as cards, with a new question box below. Follow-up questions can refer back
to earlier answers ("Which judge tried him?", "Who prosecuted them?"): with **Follow-ups use the last 3 answers**
ticked, the chat model first rewrites the follow-up into a standalone question using the last three questions,
answers and query results (e.g. "Which judge tried Aaron T. Luneke?"), and the answer starts with
*Interpreted as: ...*. Questions that are already standalone are asked unchanged. Untick the box to ask each
question on its own; this also saves one LLM call per question.

The **Agent log** panel at the bottom shows every background call - LLM prompts and replies, embedding requests,
and HugeGraph REST / Gremlin requests and responses - newest first, with timings and errors. It auto-refreshes
every 2 s and keeps the last 300 calls in memory only.

Override defaults with `OLLAMA_CHAT_MODEL`, `OLLAMA_EXTRACT_MODEL`, `OLLAMA_TEXT2GQL_MODEL`, `OLLAMA_EMBEDDING_MODEL`, `OLLAMA_HOST_IP`, `OLLAMA_PORT`,
`HUGEGRAPH_URL`, `HUGEGRAPH_GRAPH`, `HUGEGRAPH_GRAPHSPACE`, `HUGEGRAPH_USER`, `HUGEGRAPH_PASSWORD`,
`HUGEGRAPH_BACKEND`, set as environment variables or in `connection.properties` (step 5), or use the in-app Settings
panel.

Tests (`pip install pytest` once): `python -m pytest tests -q`

---

## Optional: Claude or OpenAI
Set `LLM_PROVIDER` before launching (or pick the provider in the in-app Settings panel). Chat, extraction and
Text2Gremlin then go through LiteLLM, so model names carry a LiteLLM prefix (`anthropic/...`, `openai/...`).
Steps 1-3 and 5 above still apply.

| Variable | Claude (`LLM_PROVIDER=claude`) | OpenAI (`LLM_PROVIDER=openai`) |
|---|---|---|
| API key | `ANTHROPIC_API_KEY`, or `ANTHROPIC_AUTH_TOKEN` | `OPENAI_API_KEY` |
| Base URL (optional) | `ANTHROPIC_BASE_URL` (e.g. a gateway) | `OPENAI_BASE_URL` (OpenAI-compatible proxy) |
| Chat model | `CLAUDE_MODEL`, default `anthropic/claude-sonnet-5-5`; if unset and `ANTHROPIC_DEFAULT_SONNET_MODEL` is set, `anthropic/<that value>` | `OPENAI_CHAT_MODEL`, default `openai/gpt-4.1` |
| Extraction model | `CLAUDE_EXTRACT_MODEL`, default: the chat model | `OPENAI_EXTRACT_MODEL`, default: the chat model |
| Text2Gremlin model | `CLAUDE_TEXT2GQL_MODEL`, default: the chat model | `OPENAI_TEXT2GQL_MODEL`, default: the chat model |
| Embeddings | Ollama `OLLAMA_EMBEDDING_MODEL` (Anthropic has no embedding API) | `OPENAI_EMBEDDING_MODEL`, default `openai/text-embedding-3-small` |

The chat model infers the schema and writes answers; extraction and Text2Gremlin use it too unless overridden.
A cheaper extraction model is faster but may miss relationships: on the SWIFT samples, `anthropic/claude-haiku-4-5`
extracted twice as fast as Sonnet but found only about two thirds of the edges.
The Settings panel has a box for each role (blank = chat model); with Ollama, `OLLAMA_EXTRACT_MODEL` and
`OLLAMA_TEXT2GQL_MODEL` do the same.

With Claude, keep Ollama running with `nomic-embed-text` pulled (step 4): Text2Gremlin loads an embedding
model even though this app asks it for no examples.

Documents are split into chunks at blank lines and each chunk is extracted on its own: up to 1,200 characters for
Ollama (its default context window), up to 6,000 for hosted models, which keeps a typical document in one chunk so
the model sees a full name and the later surname-only mentions together. If a dense chunk's reply is cut off at the
model's output-token limit (no valid JSON), the chunk is re-extracted in halves, down to 1,200 characters, and the
replies are combined. Chunks go to a hosted model 12 at a time;
set `EXTRACT_WORKERS` to change that (lower it if you hit rate limits). With Ollama, chunks are sent one at a time,
since Ollama handles one request at a time unless `OLLAMA_NUM_PARALLEL` is raised; set both to use concurrency there.

### Claude (direct)
```bash
export LLM_PROVIDER=claude
export ANTHROPIC_API_KEY="sk-ant-..."
python graphrag_showcase.py
```

### Claude (via an AI gateway)
```bash
export LLM_PROVIDER=claude
# macOS: fetch the auth token from the Keychain
export ANTHROPIC_AUTH_TOKEN=$(security find-generic-password -a "$USER" -s "ANTHROPIC_AUTH_TOKEN" -w)
# Linux: read it from a file only you can read (chmod 600 ~/.config/anthropic_token)
# export ANTHROPIC_AUTH_TOKEN=$(cat ~/.config/anthropic_token)
export ANTHROPIC_BASE_URL="https://ai-gateway.example.com"
export ANTHROPIC_DEFAULT_SONNET_MODEL="claude-sonnet-4-6"   # used as anthropic/claude-sonnet-4-6
python graphrag_showcase.py
```
The model must be one your gateway key may use; otherwise every call fails with 403 `key_model_access_denied ...
This key can only access models=[...]`, which names the allowed models. The same settings can go in
`connection.properties` instead of being exported (the token is best kept out of the file).

### OpenAI
```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY="sk-..."
python graphrag_showcase.py
```

The Settings panel's API key box is left blank on purpose: a blank key means "use the environment variable",
so keys are never sent to the browser.

---

## Results

Each dataset was inferred from its own documents and `SCHEMA QUESTIONS.md`, imported, and asked every question in
its questions file through the app's pipeline with Claude Sonnet. Answers were graded by hand against the questions
file and the source documents (imports of 29 October 2026):

| Dataset | Questions | Correct | Partly correct | Wrong |
|---|---|---|---|---|
| DOJ bank fraud (`DOJ_fraud_samples/banking`) | 30 | 29 | 1 | 0 |
| DOJ insurance fraud (`DOJ_fraud_samples/insurance`) | 30 | 28 | 2 | 0 |
| Semiconductor companies (`WIKI_semiconductor_samples`) | 36 | 32 | 4 | 0 |

The partly correct answers come from extraction: a two-year mandatory minimum stored as a maximum, charges stored
only on fraud edges, a missing headquarters or founding year, and generic products such as "CPU". Scores vary
between imports, mostly with how names come out: AMD under one name or two, Avago Technologies separate from
Broadcom Inc. or merged into it, an unnamed fraud victim kept or dropped. For a dependable demo script, use the
reliable question sets (`... QUESTIONS - reliable.md`: 9 insurance, 17 banking), which every inferred schema so far
has supported.

Two tests showed what the pipeline needs. Without the safety nets listed above, each dataset scored the same or
better, because none of the failures they guard against occurred. Without property indexes, banking fell from 29
to 20 correct, because HugeGraph refuses a query that starts by filtering on a property it has not indexed.

To repeat a run, load a dataset's graph in the app, then:

```bash
../.venv/bin/python benchmarks/bench_models.py "DOJ_fraud_samples/banking/BANKING FRAUD QUESTIONS to ASK.md" \
    benchmarks/results/banking_qa.json claude:anthropic/claude-sonnet
```

[docs/text-to-graph-findings.md](docs/text-to-graph-findings.md) explains how the scores got here and what still
goes wrong; [docs/hugegraph-ai-additions.md](docs/hugegraph-ai-additions.md) lists what the app adds to hugegraph-ai
and why; [benchmarks/README.md](benchmarks/README.md) lists every run with its answers file and score; and
[docs/model-choice.md](docs/model-choice.md) explains why the demo runs on Claude Sonnet.

## Sample documents: DOJ fraud press releases (`DOJ_fraud_samples/`)
54 real U.S. Department of Justice press releases (public domain) about bank and insurance fraud, one `.txt`
per release with a `Title / Date / Office / Source` header. They are narrative text, which is what LLM
schema inference and extraction are good at.

* `schema_sample.txt` - one insurance and one banking release (~4,900 chars); use as the *sample* for schema inference,
  or upload several releases from different cases (e.g. one file from each of 4-6 number prefixes).
* `banking/` (27 files) and `insurance/` (27 files) - 30 cases. Files sharing a number prefix are the same case
  (charged -> pleaded guilty -> sentenced), e.g. the Boston bank fraud ring, the TD Bank insiders, the failed
  First National Bank of Lindsay and the $20M Wilson life-insurance fraud, so defendants, banks, insurers,
  judges and agencies recur and link the graph.
* `extraction_hints.txt` - notes on name normalisation, org types, offenses and amounts. The app no longer has a
  hints box; the benchmark scripts can still pass these notes to extraction.
* Before *Infer schema*, paste the questions from the dataset's `SCHEMA QUESTIONS.md` into **Questions this graph
  should answer**, and leave them there for the ingest. The schema then keeps the judges, spouses and
  co-conspirators those questions need, and extraction keeps the facts they need (e.g. a prosecutor credited only
  in a closing line, or a victim the text does not name).
* `doj_fraud_schema.json` - hand-written person / organization / agency / offense schema; paste into the
  schema box if the inferred one is off.
* `sources.csv` - original URL, office and date of every file; `NOTICE.txt` - provenance and caveats.

Example questions: *Which banks did Shaun Christian defraud?*, *Who was sentenced by Judge Julia E. Kobick?*,
*Which insurance agents were charged with wire fraud?*, *Which defendants were investigated by the FDIC Office
of Inspector General?*

## Sample documents: semiconductor companies (`WIKI_semiconductor_samples/`)
A second, unrelated test set: the opening sections (about 5,000 characters each) of the English Wikipedia articles
on Nvidia, AMD, Intel, Arm Holdings, Qualcomm, TSMC, ASML and Broadcom, in `articles/`. They cover founders and
executives, headquarters, products, acquisitions and investments. `sources.csv` lists the article URLs and
`NOTICE.txt` the license (CC BY-SA 4.0). There is no hand-written schema: upload all eight as samples, paste
`SCHEMA QUESTIONS.md` into **Questions this graph should answer**, infer the schema, and ingest the same files.
Upload only the files in `articles/`, not `NOTICE.txt` or the `.md` files.

Example questions: *Who founded Qualcomm?*, *Which companies are headquartered in Santa Clara, California?*,
*Which companies did Broadcom acquire for more than 1,000 million dollars?*, *Who invested in TSMC?*

## Building a larger DOJ dataset
`DOJ_fraud_samples/` is a hand-picked set. More bank and insurance fraud releases can be collected from the public
DOJ press-release API (no API key) by hand in a browser, in the same layout: one `.txt` per release with a
`Title / Date / Office / Source` header, plus `sources.csv` and `NOTICE.txt`.

### By hand, using the API in a browser
Good for 30-60 releases (about 3-5 minutes each), and lets you follow whole cases so the graph stays connected.

1. **List releases.** Open this URL in a browser; it returns press releases whose title contains the keyword,
   newest first, 50 per page:
   ```
   https://www.justice.gov/api/v1/press_releases.json?pagesize=50&sort=date&direction=DESC&parameters[title]=insurance
   ```
   * Firefox shows the JSON as a collapsible tree; in Chrome tick *Pretty-print*.
   * For banking, change the keyword to `bank%20fraud`, `credit%20union`, `mortgage%20fraud` or `loan%20fraud`.
   * Add `&page=1`, `&page=2`, ... for older releases; stop when the dates are older than you need.
   * Each entry under `results` has `title`, `date` (a Unix timestamp; convert it with any epoch converter),
     `component[0].name` (the office), `url` (the readable release page) and `topic`.
2. **Choose releases.**
   * Keep cases where the victim is a bank, credit union, lender, insurer or insurance agency.
   * Skip health care, Medicare/Medicaid/TRICARE, COVID/PPP/EIDL, unemployment and Social Security/disability
     fraud: their victims are government programs, which the sample schema does not model.
   * Follow cases: keep the *charged*, *pleaded guilty* and *sentenced* releases of a case together. Titles often
     omit the defendant's name, but follow-up releases usually reuse the title wording, so search a distinctive
     phrase (e.g. `parameters[title]=new%20bedford%20couple`). Recurring people, banks and judges are what link
     the graph across files.
3. **Save each release as text.** Open its `url`, copy the text from the first paragraph to the last paragraph
   about the case, and paste it into a plain-text editor (TextEdit: *Format -> Make Plain Text*). Put this header
   at the top, followed by one blank line:
   ```
   Title: <title>
   Date: <YYYY-MM-DD>
   Office: <component[0].name, e.g. USAO - Massachusetts>
   Source: <url>
   ```
   * Delete the closing boilerplate: presumption-of-innocence and maximum-penalty paragraphs, contact names,
     emails and phone numbers, *Updated* / *Topics* / *Component* lines, images and `* * *` separators.
   * Keep one blank line between paragraphs; the app packs chunks at blank lines.
   * Save as UTF-8 in `DOJ_fraud_dataset/insurance/` or `DOJ_fraud_dataset/banking/`, named
     `NN_YYYY-MM-DD_short-title.txt`, where `NN` is a number you give each case (all releases of one case share it).
4. **Record sources.** Create `DOJ_fraud_dataset/sources.csv` with the columns
   `file,case,date,office,title,url` (one row per file, e.g. `insurance/03_2025-10-15_title-company-sentenced.txt,ins-03,...`)
   and copy `DOJ_fraud_samples/NOTICE.txt` into the folder.
5. **Check** that every file starts with the 4-line header and a blank line, no boilerplate is left, and the
   files are `.txt` (the app also takes `.pdf` / `.docx`, not `.csv`).

### Loading the dataset
1. **Clear** the graph, then paste `DOJ_fraud_samples/doj_fraud_schema.json`, or upload 4-6 of the new files as
   samples and list the questions the graph should answer to infer a schema.
2. Ingest a first batch of 10-20 files and read the ingest report:
   dropped items and empty labels show where the schema needs extending before you ingest everything.

People are keyed by name, so in a large dataset two different people with the same name become one vertex. Releases
that are not grouped into cases have fewer recurring defendants, so the graph is more loosely connected. Ingest cost
grows with the number of files: roughly 2-4 extraction calls per release.
