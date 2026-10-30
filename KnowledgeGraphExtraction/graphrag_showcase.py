"""A minimal GraphRAG showcase running on local Ollama models by default (Claude or OpenAI optional).

Workflow:
1. Build: upload one sample document -> LLM infers a graph schema -> create it in HugeGraph,
   then upload more documents of the same shape to extract and import vertices/edges.
2. Explore: view graph statistics and a sample visualization, ask a question in natural
   language -> Text2Gremlin -> execute on HugeGraph -> LLM summarizes the results.

Run (with .venv activated): python graphrag_showcase.py
Pick a hosted model with LLM_PROVIDER=claude or LLM_PROVIDER=openai (see README.md).
"""

import argparse
import functools
import html
import json
import inspect
import os
import queue
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import gradio as gr
import litellm
import requests

from hugegraph_llm.config import huge_settings, llm_settings, prompt, resource_path
from hugegraph_llm.flows import FlowName
from hugegraph_llm.flows.scheduler import SchedulerSingleton
from hugegraph_llm.models.llms.init_llm import get_chat_llm, get_extract_llm
from hugegraph_llm.models.llms.litellm import LiteLLMClient
from hugegraph_llm.operators.common_op.check_schema import CheckSchema
from hugegraph_llm.operators.document_op.chunk_split import SPLIT_TYPE_DOCUMENT
from hugegraph_llm.operators.hugegraph_op.commit_to_hugegraph import Commit2Graph
from hugegraph_llm.operators.llm_op.gremlin_generate import GremlinGenerateSynthesize
from hugegraph_llm.operators.llm_op.property_graph_extract import (
    PropertyGraphExtract,
    filter_item,
    generate_extract_property_graph_prompt,
)
from hugegraph_llm.utils.hugegraph_utils import clean_hg_data, get_hg_client, run_gremlin_query
from hugegraph_llm.utils.log import log
from hugegraph_llm.utils.vector_index_utils import read_documents

import agent_log

# Local models have small context windows; schema inference only needs a representative sample.
# The budget is shared by all sample documents, so inferring from several files does not grow the prompt.
MAX_SAMPLE_CHARS = 6000
HOSTED_SAMPLE_CHARS = 24000
# A schema inferred from one document fits that document's story; excerpts from several make the model keep
# only what the documents share. Larger uploads are sampled evenly so every group of files is represented.
MAX_SAMPLE_FILES = 6
MAX_RESULT_CHARS = 6000
# The summary is written at local-model speed (~20 tokens/s), so it gets a sample of rows and a short list;
# the full result is shown in the UI's raw-results panel.
SUMMARY_MAX_ROWS = 20
SUMMARY_LIST_ITEMS = 10
# Chunks are packed from blank-line separated blocks so a record (e.g. one paragraph or form entry) is never cut;
# kept small enough that prompt + JSON output fit Ollama's default context window. Hosted models get chunks that
# hold most documents whole: each chunk is extracted on its own, so a chunk that only says "Luneke" cannot know
# the full name given in an earlier one.
CHUNK_CHARS = 1200
HOSTED_CHUNK_CHARS = 6000
SCHEMA_ATTEMPTS = 3
# Hosted extraction is bound by output tokens (~15s per chunk), so the chunks of every document being ingested
# share one pool of concurrent LLM calls. Ollama serves one request at a time unless OLLAMA_NUM_PARALLEL is raised,
# so it stays sequential by default.
HOSTED_EXTRACT_WORKERS = 12
RANGE_INDEX_TYPES = {"INT", "LONG", "FLOAT", "DOUBLE", "DATE"}

# Appended to the default Text2Gremlin prompt via the flow's prompt override (braces would break .format).
GREMLIN_RULES = """# HugeGraph Rules
- Always scope vertex lookups with a label before filtering: g.V().hasLabel('LABEL').has('KEY', 'VALUE').
  HugeGraph rejects has('name', ...) without hasLabel(...) because a label's primary key is not separately indexed.
- Respect edge direction from the schema: an edge goes OUT of its source_label and IN to its target_label.
  From a target vertex, use in('edge_label') to reach the source; from a source vertex, use out('edge_label').
- Match a person's full name exactly as written in the question (original casing).
- Match organizations, agencies, places and other names with TextP.containing, because stored names are often
  longer (a stored 'Acme Corporation, Inc.' does not equal 'Acme', nor 'Suffolk County, New York' equal 'New York'):
  g.V().hasLabel('LABEL').has('KEY', TextP.containing('VALUE')).
- When the question filters on, sorts by or asks about a property (an amount, age, date, count, duration),
  return that value with each name: .project('name', 'KEY').by('name').by('KEY'). Across an edge, return the
  name at each end and the edge property.
- For "most", "least", "largest" or "smallest" questions, return only the top item with its count or value:
  .order().by(select('count'), desc).limit(1). Return a ranking only when the question asks for several items
  ("top 5", "rank", "list").
- Compare DATE properties as ranges of 'yyyy-MM-dd' values, never with TextP: for a year or month use
  has('KEY', P.gte('2027-01-01')).has('KEY', P.lt('2028-01-01')).
- Properties other than the key may be missing on some vertices or edges, and by('KEY') drops every row that
  lacks KEY. When a property is only returned, not filtered on, use by(coalesce(values('KEY'), constant(''))).
- Always write a real query, chaining several in()/out() steps when the question needs it.
  Never answer with g.V().limit(0).

"""
# hugegraph-llm's default prompt tells the model to return g.V().limit(0) for "complex" questions, because its
# RAG flows can fall back to other retrieval. Text2Gremlin is the only way this app answers, so drop that branch.
COMPLEX_QUERY_SECTION = re.compile(r"## Complex Query Detection:.*?(?=## Gremlin Query Generation)", re.DOTALL)
DECLINED_GREMLIN = re.compile(r"^\s*g\.V\(\)\.limit\(0\)\s*;?\s*$")
# A two-argument has('key', 'text') or has('key', TextP.containing('text')) filter; the 3-argument
# has(label, key, value) form is left alone.
EXACT_TEXT_FILTER = re.compile(
    r"""\.has\(\s*(['"])(\w+)\1\s*,\s*"""
    # TextP.containing('text') and its closing bracket, or a plain 'text'; matching them as a pair keeps the bracket
    # of an enclosing step, as in where(outE('charged_with').has('plea_status', 'pending')).
    r"""(?:TextP\.containing\(\s*(['"])((?:(?!\3).)+)\3\s*\)|(['"])((?:(?!\5).)+)\5)\s*\)"""
)
# A TextP filter inside has(...), e.g. has('name', TextP.containing('TD Bank')). HugeGraph servers before 1.8 hand it to
# the storage backend, which rejects it ("Unsupported predicate: 'containing(TD Bank)'"); 1.8 evaluates it in memory.
# For those servers it is rewritten as filter(__.values('name').is(TextP.containing('TD Bank'))), which returns the
# same elements and is always evaluated in memory.
TEXT_PREDICATE_FILTER = re.compile(
    r"""(?<!\w)has\(\s*(['"])(\w+)\1\s*,\s*(TextP\.\w+\(\s*(['"])(?:(?!\4).)*\4\s*\))\s*\)"""
)
# The rewritten form of a TextP.containing filter, so the partial-match retry can relax it too.
IS_TEXT_CONTAINS = re.compile(r"""\.is\(\s*TextP\.containing\(\s*(['"])((?:(?!\1).)+)\1\s*\)\s*\)""")
TEXT_FILTER_SERVER_VERSION = (1, 8, 0)
_SERVER_VERSIONS: Dict[str, Optional[Tuple[int, ...]]] = {}
# Same default as hugegraph-llm's Gremlin execution, which appends .limit(100) to queries without a limit.
QUERY_LIMIT = 100
SUPPORTED_SUFFIXES = (".txt", ".pdf", ".docx")
# The schema-build prompt relies on a few-shot example to convey HugeGraph's schema format.
FEW_SHOT_SCHEMA_PATH = Path(resource_path) / "prompt_examples" / "schema_examples.json"

SUMMARY_PROMPT = """You are a helpful analyst. A user asked a question about a knowledge graph.
The question was translated to the Gremlin query below and executed on HugeGraph.

Question: {question}

Gremlin:
{gremlin}

Raw result ({shown} of {total} rows, JSON, may be truncated):
{result}

Answer the question using ONLY the raw result, and keep it brief (at most about 150 words);
the user sees the full raw result separately. Start with a one-sentence direct answer that states
how many results there are when that matters ({total} in total). Then give at most {list_items}
bullet points or table rows for the most relevant entities and relationships, naming each entity and
adding only the properties the question asks about; if there are more, end with "...and N more".
If the result is empty or does not answer the question, say so plainly and do not invent facts.
"""


PROVIDERS = ("ollama", "claude", "openai")
LLM_ROLES = ("chat", "extract", "text2gql")
# hugegraph-llm caps LiteLLM extraction at 256 tokens, far too few for a chunk's vertex/edge JSON
# (a 1200-char SWIFT chunk yields ~3-4k tokens). A dense 5k-char document chunk (8 defendants x 7 agencies) needs
# more than 8192 with reasoning tokens counted, so this is double the chat role's cap.
HOSTED_EXTRACT_TOKENS = 16384


# Settings can also live in connection.properties next to this file, as KEY=VALUE lines with the same names as the
# environment variables (# or ! starts a comment), so a remote HugeGraph cluster needs nothing exported.
# connection.properties is the shared, committed template; connection.properties.local (git-ignored) holds this
# machine's values and passwords. An environment variable wins over both, and the local file over the shared one.
CONNECTION_PROPERTIES = Path(__file__).resolve().parent / "connection.properties"
LOCAL_PROPERTIES = CONNECTION_PROPERTIES.with_name("connection.properties.local")


def read_properties(path: Path) -> Dict[str, str]:
    """KEY=VALUE pairs from a properties file; an empty dict when the file does not exist."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    props = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line[0] in "#!":
            continue
        key, sep, value = line.partition("=")
        if sep:
            props[key.strip()] = value.strip()
    return props


def setting(name: str, default: str = "") -> str:
    """An environment variable, else connection.properties.local, else connection.properties, else the default."""
    value = os.getenv(name)
    if value is not None:
        return value
    props = {**read_properties(CONNECTION_PROPERTIES), **read_properties(LOCAL_PROPERTIES)}
    return props.get(name, default)


def provider_defaults(provider: str) -> Dict[str, str]:
    """Model names and credentials for one provider, overridable through environment variables or connection.properties.

    A blank extract/text2gql model means "same as the chat model". Anthropic has no embedding API,
    so Claude keeps embeddings on Ollama.
    """
    if provider == "claude":
        # Cloudera AI Gateway setups export ANTHROPIC_DEFAULT_SONNET_MODEL / ANTHROPIC_AUTH_TOKEN.
        gateway_model = setting("ANTHROPIC_DEFAULT_SONNET_MODEL")
        return {
            "chat_model": setting("CLAUDE_MODEL")
            or (f"anthropic/{gateway_model}" if gateway_model else "anthropic/claude-sonnet-5-5"),
            # Haiku extracts twice as fast but missed ~1/3 of the edges on the SWIFT samples, so it is opt-in.
            "extract_model": setting("CLAUDE_EXTRACT_MODEL", ""),
            "text2gql_model": setting("CLAUDE_TEXT2GQL_MODEL", ""),
            "embedding_model": setting("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
            "api_key": setting("ANTHROPIC_API_KEY") or setting("ANTHROPIC_AUTH_TOKEN", ""),
            "api_base": setting("ANTHROPIC_BASE_URL", ""),
        }
    if provider == "openai":
        return {
            "chat_model": setting("OPENAI_CHAT_MODEL", "openai/gpt-4.1"),
            "extract_model": setting("OPENAI_EXTRACT_MODEL", ""),
            "text2gql_model": setting("OPENAI_TEXT2GQL_MODEL", ""),
            "embedding_model": setting("OPENAI_EMBEDDING_MODEL", "openai/text-embedding-3-small"),
            "api_key": setting("OPENAI_API_KEY", ""),
            "api_base": setting("OPENAI_BASE_URL", ""),
        }
    return {
        # 8B models (e.g. llama3.1) run but often pick the wrong label/direction in Text2Gremlin
        "chat_model": setting("OLLAMA_CHAT_MODEL", "qwen3-coder"),
        "extract_model": setting("OLLAMA_EXTRACT_MODEL", "qwen3-coder"),
        "text2gql_model": setting("OLLAMA_TEXT2GQL_MODEL", "qwen3-coder"),
        "embedding_model": setting("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
        "api_key": "",
        "api_base": "",
    }


def default_config() -> Dict[str, Any]:
    """Defaults for a local setup, overridable through environment variables or connection.properties."""
    provider = setting("LLM_PROVIDER", "ollama").strip().lower()
    if provider not in PROVIDERS:
        raise ValueError(f"LLM_PROVIDER must be one of {', '.join(PROVIDERS)}, got '{provider}'")
    return {
        "provider": provider,
        **provider_defaults(provider),
        "ollama_host": setting("OLLAMA_HOST_IP", "127.0.0.1"),
        "ollama_port": int(setting("OLLAMA_PORT", "11434")),
        "graph_url": setting("HUGEGRAPH_URL", "127.0.0.1:8080"),
        # Blank means the server's default graph space (DEFAULT on HugeGraph 1.7 and later).
        "graph_space": setting("HUGEGRAPH_GRAPHSPACE", ""),
        "graph_name": setting("HUGEGRAPH_GRAPH", "hugegraph"),
        "graph_user": setting("HUGEGRAPH_USER", "admin"),
        "graph_pwd": setting("HUGEGRAPH_PASSWORD", "admin"),
    }


def apply_config(
    provider: str,
    chat_model: str,
    embedding_model: str,
    api_key: str,
    api_base: str,
    ollama_host: str,
    ollama_port: int,
    graph_url: str,
    graph_name: str,
    graph_user: str,
    graph_pwd: str,
    extract_model: str = "",
    text2gql_model: str = "",
    graph_space: Optional[str] = None,
) -> str:
    """Point every LLM role (chat/extract/text2gql/embedding) at the chosen provider, in memory only.

    Each role may use its own model; a blank extract/text2gql model falls back to the chat model.
    Ollama is called natively; Claude and OpenAI go through LiteLLM, which routes by the model prefix.
    """
    if provider not in PROVIDERS:
        raise gr.Error(f"Unknown provider '{provider}'. Use one of: {', '.join(PROVIDERS)}")
    # A blank key box falls back to the environment so keys never have to be sent to the browser.
    api_key = api_key or provider_defaults(provider)["api_key"]
    port = int(ollama_port)
    models = {"chat": chat_model, "extract": extract_model or chat_model, "text2gql": text2gql_model or chat_model}
    for role in LLM_ROLES:
        if provider == "ollama":
            setattr(llm_settings, f"{role}_llm_type", "ollama/local")
            setattr(llm_settings, f"ollama_{role}_host", ollama_host)
            setattr(llm_settings, f"ollama_{role}_port", port)
            setattr(llm_settings, f"ollama_{role}_language_model", models[role])
        else:
            setattr(llm_settings, f"{role}_llm_type", "litellm")
            setattr(llm_settings, f"litellm_{role}_language_model", models[role])
            setattr(llm_settings, f"litellm_{role}_api_key", api_key or None)
            setattr(llm_settings, f"litellm_{role}_api_base", api_base or None)
    if provider != "ollama":
        llm_settings.litellm_extract_tokens = max(llm_settings.litellm_extract_tokens, HOSTED_EXTRACT_TOKENS)

    if provider == "openai":
        llm_settings.embedding_type = "litellm"
        llm_settings.litellm_embedding_model = embedding_model
        llm_settings.litellm_embedding_api_key = api_key or None
        llm_settings.litellm_embedding_api_base = api_base or None
        embedding_at = "OpenAI"
    else:
        llm_settings.embedding_type = "ollama/local"
        llm_settings.ollama_embedding_host = ollama_host
        llm_settings.ollama_embedding_port = port
        llm_settings.ollama_embedding_model = embedding_model
        embedding_at = f"Ollama {ollama_host}:{port}"
    llm_settings.language = "EN"

    huge_settings.graph_url = graph_url
    huge_settings.graph_name = graph_name
    huge_settings.graph_user = graph_user
    huge_settings.graph_pwd = graph_pwd
    # None keeps the configured graph space; blank means the server's default.
    space = setting("HUGEGRAPH_GRAPHSPACE") if graph_space is None else graph_space
    huge_settings.graph_space = space.strip() or None

    chat_at = f"Ollama {ollama_host}:{port}" if provider == "ollama" else (api_base or provider)
    if len(set(models.values())) == 1:
        model_note = f"`{chat_model}`"
    else:
        model_note = ", ".join(f"{role} `{model}`" for role, model in models.items())
    status = (
        f"Using {model_note} ({chat_at}) + embeddings `{embedding_model}` ({embedding_at}); "
        f"HugeGraph `{f'{huge_settings.graph_space}/' if huge_settings.graph_space else ''}{graph_name}` at {graph_url}"
    )
    if provider != "ollama" and not api_key:
        status += f"\n\n**Warning:** no API key set for {provider}."
    return status


def _as_paths(files: Any) -> List[str]:
    if not files:
        return []
    if not isinstance(files, (list, tuple)):
        files = [files]
    return [f if isinstance(f, str) else f.name for f in files]


def pack_chunks(text: str, limit: int = CHUNK_CHARS) -> List[str]:
    """Split text at blank lines and greedily pack whole blocks into chunks of at most `limit` chars.

    A block longer than `limit` stays whole unless it exceeds 3x the limit, in which case it is packed
    line by line instead, so very long unbroken prose still fits the model context.
    """
    blocks: List[str] = []
    for block in (b.strip() for b in text.split("\n\n")):
        if not block:
            continue
        if len(block) > 3 * limit:
            blocks.extend(pack_chunks(block.replace("\n", "\n\n"), limit))
        else:
            blocks.append(block)
    chunks: List[str] = []
    for block in blocks:
        if chunks and len(chunks[-1]) + 2 + len(block) <= limit:
            chunks[-1] += "\n\n" + block
        else:
            chunks.append(block)
    return chunks


def read_file_text(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported file type '{suffix}'. Use one of: {', '.join(SUPPORTED_SUFFIXES)}")
    # read_documents expects file-like objects exposing `.name`
    return read_documents([SimpleNamespace(name=path)], None)[0]


def parse_schema(schema_text: str) -> Dict[str, Any]:
    try:
        schema = json.loads(schema_text or "")
    except json.JSONDecodeError as exc:
        raise ValueError(f"Schema is not valid JSON: {exc.msg} (line {exc.lineno})") from exc
    if not isinstance(schema, dict):
        raise ValueError("Schema must be a JSON object.")
    # CheckSchema validates labels and fills in primary/nullable keys and missing property keys.
    schema = CheckSchema(schema).run()["schema"]
    if not schema["vertexlabels"]:
        raise ValueError("Schema has no vertex labels.")
    vertex_names = {vl["name"] for vl in schema["vertexlabels"]}
    for el in schema["edgelabels"]:
        el.setdefault("properties", [])
        unknown = {el["source_label"], el["target_label"]} - vertex_names
        if unknown:
            raise ValueError(
                f"Edge label '{el['name']}' references unknown vertex label(s): {', '.join(sorted(unknown))}"
            )
    return schema


def summarize_schema(schema: Dict[str, Any]) -> str:
    lines = ["**Vertex labels**"]
    for vl in schema["vertexlabels"]:
        lines.append(
            f"- `{vl.get('name')}` (key: {', '.join(vl.get('primary_keys', []))}) - {', '.join(vl.get('properties', []))}"
        )
    lines.append("\n**Edge labels**")
    for el in schema["edgelabels"]:
        lines.append(f"- `{el.get('source_label')}` -[`{el.get('name')}`]-> `{el.get('target_label')}`")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tab 1: build the knowledge graph
# ---------------------------------------------------------------------------
# hugegraph-llm discards any edge without a "properties" key, and models tend to omit it when the edge label has no
# properties, so every edge of such a schema would be lost.
EDGE_PROPERTIES_RULE = (
    "## Edge Properties\n"
    'Every edge object must include a "properties" key, even when its edge label declares no properties: '
    'write "properties": {}. Edges without it are discarded.\n'
)


def extraction_prompt(hints: str = "") -> str:
    """The default extraction prompt with the edge-properties rule, plus optional user-supplied domain notes."""
    base = f"{prompt.extract_graph_prompt}\n\n{EDGE_PROPERTIES_RULE}"
    hints = (hints or "").strip()
    if not hints:
        return base
    return f"{base}\n## Domain Hints\nUse these notes to interpret the input text:\n{hints}\n"


def with_hints(text: str, hints: str = "") -> str:
    hints = (hints or "").strip()
    return f"Domain notes (not data):\n{hints}\n\nData sample:\n{text}" if hints else text


def chunk_chars() -> int:
    """Extraction chunk size: small for Ollama's context window, document-sized for hosted models."""
    return CHUNK_CHARS if llm_settings.chat_llm_type == "ollama/local" else HOSTED_CHUNK_CHARS


def sample_chars() -> int:
    """Schema-inference sample budget: small for Ollama's context window, larger for hosted models."""
    return MAX_SAMPLE_CHARS if llm_settings.chat_llm_type == "ollama/local" else HOSTED_SAMPLE_CHARS


def pick_samples(paths: List[str], limit: int = MAX_SAMPLE_FILES) -> List[str]:
    """Up to `limit` paths spread evenly over the upload, keeping the first and last."""
    if len(paths) <= limit:
        return list(paths)
    if limit == 1:
        return paths[:1]
    return [paths[round(i * (len(paths) - 1) / (limit - 1))] for i in range(limit)]


# Rules that apply however many samples there are; without the last one a schema may type every property as TEXT,
# and filters such as has('age', gt(50)) then fail on HugeGraph.
SCHEMA_DESIGN_RULES = (
    "- Use primary keys that every document provides, such as a name, never an ID or number that only "
    "some documents state.\n"
    "- Prefer direct edges between entities over a hub vertex (e.g. an event or case) that every edge must "
    "pass through.\n"
    "- Give quantities such as ages, amounts, counts and durations numeric data types (INT or DOUBLE), and "
    "dates the DATE type."
)


def build_schema_sample(paths: List[str], budget: int) -> str:
    """Design rules plus the sample: one document as is, or labelled excerpts of several sharing the budget."""
    if len(paths) == 1:
        intro = (
            "The data below is one sample of a collection of similar documents. Design a schema that fits the "
            "whole collection, not only this sample:\n" + SCHEMA_DESIGN_RULES
        )
        return f"{intro}\n\n{read_file_text(paths[0])[:budget]}"
    share = budget // len(paths)
    excerpts = [
        f"--- Document {i} of {len(paths)} ({Path(path).name}) ---\n{read_file_text(path)[:share]}"
        for i, path in enumerate(paths, start=1)
    ]
    # Asking only for what fits "all" documents makes models return a bare minimum (person, organization,
    # works_for), so the intro asks for everything that recurs and only excludes one-off details.
    intro = (
        f"The data below is excerpts from {len(paths)} different documents of the same kind. "
        "Design one schema for the whole collection:\n"
        "- Include every kind of entity, relationship and property that appears in at least two of the "
        "documents, so the graph captures what the documents are about; leave out details only one mentions.\n"
        + SCHEMA_DESIGN_RULES
    )
    return "\n\n".join([intro] + excerpts)


def question_lines(questions: str) -> List[str]:
    """The questions in the UI box, one per line, without list markers or blank lines."""
    lines = [re.sub(r"^\s*(?:[-*\u2022]|\d+[.)])\s*", "", line).strip() for line in (questions or "").splitlines()]
    return [line for line in lines if line]


def query_examples(questions: str) -> Optional[str]:
    """The questions in the JSON shape hugegraph-llm's schema prompt lists as "Query Examples".

    They tell the model what the graph is for, so it keeps the relationships those questions need (a sentencing
    judge, a spouse) instead of choosing among equally valid designs. The schema itself is still inferred from the
    documents. hugegraph-llm drops an example without a "gremlin" key, so it is sent empty.
    """
    examples = [{"description": line, "gremlin": ""} for line in question_lines(questions)]
    return json.dumps(examples, ensure_ascii=False) if examples else None


def question_hints(questions: str) -> str:
    """The questions as extraction notes, so extraction keeps the facts they need.

    Without them the model skips links it does not see a use for: a prosecutor credited in a closing line is not
    linked to the defendants, and a victim the text does not name gets no vertex, so its fraud edge is lost.
    """
    lines = question_lines(questions)
    if not lines:
        return ""
    return "The graph built from these documents must answer the questions below. Extract every fact they need:\n" + (
        "\n".join(lines)
    )


ALL_TEXT_WARNING = (
    "**Warning:** every property is TEXT, so filters on numbers and dates (ages, amounts, dates) will fail. "
    "Click *Infer schema* again, or set types such as INT, DOUBLE and DATE in `propertykeys` before creating the graph."
)


def all_text_properties(schema: Dict[str, Any]) -> bool:
    """True when the schema has properties besides its keys and every one is TEXT.

    That happens when the reply leaves out `propertykeys`: CheckSchema then types every property TEXT, and a filter
    like has('age', gt(50)) is rejected by HugeGraph ("expect TEXT for 'age'").
    """
    keys = {key for vl in schema["vertexlabels"] for key in vl.get("primary_keys", [])}
    types = {pk["name"]: str(pk.get("data_type", "TEXT")).upper() for pk in schema.get("propertykeys", [])}
    others = [name for name in types if name not in keys]
    return bool(others) and all(types[name] == "TEXT" for name in others)


def infer_schema(sample_files, hints: str = "", scheduler=None, questions: str = "") -> Tuple[str, str]:
    paths = pick_samples(_as_paths(sample_files))
    if not paths:
        raise gr.Error("Please upload at least one sample document first.")
    try:
        text = build_schema_sample(paths, sample_chars())
    except Exception as exc:  # pylint: disable=broad-exception-caught
        raise gr.Error(f"Could not read sample: {exc}") from exc
    scheduler = scheduler or SchedulerSingleton.get_instance()
    few_shot = FEW_SHOT_SCHEMA_PATH.read_text(encoding="utf-8")
    error: Exception = RuntimeError("no attempt made")
    # Small local models sometimes return malformed schemas, and any model can leave out the property types; a fresh
    # attempt usually fixes either. The last all-TEXT schema is still returned, with a warning, so it can be edited.
    attempts = SCHEMA_ATTEMPTS
    for attempt in range(1, attempts + 1):
        try:
            schema = parse_schema(
                scheduler.schedule_flow(
                    FlowName.BUILD_SCHEMA, [with_hints(text, hints)], query_examples(questions), few_shot
                )
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            log.warning("Schema inference attempt %s/%s failed: %s", attempt, attempts, exc)
            error = exc
            continue
        all_text = all_text_properties(schema)
        if all_text and attempt < attempts:
            log.warning("Schema inference attempt %s/%s typed every property TEXT; inferring again", attempt, attempts)
            continue
        summary = summarize_schema(schema)
        if all_text:
            summary = f"{ALL_TEXT_WARNING}\n\n{summary}"
        return json.dumps(schema, ensure_ascii=False, indent=2), summary
    raise gr.Error(f"Schema inference failed after {attempts} attempts: {error}")


def create_index_labels(schema_api, schema: Dict[str, Any]) -> List[str]:
    """Index vertex and edge properties so Text2Gremlin filters like has('age', gt(30)) are accepted by HugeGraph.

    Edges matter too: g.E().hasLabel('defrauded').has('amount', gt(1000000)) is refused without an edge index.
    A property that alone forms a vertex label's primary key is skipped: HugeGraph refuses to index it and
    serves hasLabel(...).has(pk, ...) lookups through the vertex id instead.
    """
    data_types = {pk["name"]: str(pk.get("data_type", "TEXT")).upper() for pk in schema["propertykeys"]}
    targets = [
        ("onV", vl["name"], prop)
        for vl in schema["vertexlabels"]
        for prop in vl["properties"]
        if vl["primary_keys"] != [prop]
    ]
    targets += [("onE", el["name"], p) for el in schema.get("edgelabels", []) for p in el.get("properties", [])]
    created = []
    for on, label, prop in targets:
        name = f"{label}By{prop[:1].upper()}{prop[1:]}"
        index = getattr(schema_api.indexLabel(name), on)(label).by(prop)
        index = index.range() if data_types.get(prop) in RANGE_INDEX_TYPES else index.secondary()
        index.ifNotExist().create()
        created.append(name)
    return created


def _edge_links(edge_label: Dict[str, Any]) -> set:
    """(source, target) pairs of an edge label; HugeGraph 1.5+ reports them as `links`, older versions as fields."""
    links = {(src, dst) for link in edge_label.get("links") or [] for src, dst in link.items()}
    if edge_label.get("source_label"):
        links.add((edge_label["source_label"], edge_label.get("target_label")))
    return links


def schema_conflicts(schema: Dict[str, Any], existing: Optional[Dict[str, Any]]) -> List[str]:
    """Ways the schema clashes with the one already in HugeGraph.

    Schema creation only adds what does not exist yet, so a label already in the graph keeps its old definition and
    every later import that relies on the new one fails (e.g. "Undefined link of edge label 'charged_with'").
    """
    if not existing:
        return []
    conflicts = []
    old_types = {pk["name"]: str(pk.get("data_type", "")).upper() for pk in existing.get("propertykeys") or []}
    for pk in schema.get("propertykeys", []):
        new_type, old_type = str(pk.get("data_type", "")).upper(), old_types.get(pk["name"])
        if old_type and new_type and old_type != new_type:
            conflicts.append(f"property `{pk['name']}` is {old_type} in the graph, {new_type} in this schema")
    old_vertices = {vl["name"]: vl for vl in existing.get("vertexlabels") or []}
    for vl in schema["vertexlabels"]:
        old = old_vertices.get(vl["name"])
        if old is None:
            continue
        if list(old.get("primary_keys") or []) != list(vl["primary_keys"]):
            conflicts.append(
                f"vertex `{vl['name']}` is keyed on {old.get('primary_keys')} in the graph, {vl['primary_keys']} here"
            )
        missing = [p for p in vl["properties"] if p not in (old.get("properties") or [])]
        if missing:
            conflicts.append(f"vertex `{vl['name']}` in the graph has no propert(ies) {', '.join(missing)}")
    old_edges = {el["name"]: el for el in existing.get("edgelabels") or []}
    for el in schema["edgelabels"]:
        old = old_edges.get(el["name"])
        if old is None:
            continue
        if (el["source_label"], el["target_label"]) not in _edge_links(old):
            graph_links = ", ".join(f"{src} -> {dst}" for src, dst in sorted(_edge_links(old)))
            conflicts.append(
                f"edge `{el['name']}` is {graph_links} in the graph, {el['source_label']} -> {el['target_label']} here"
            )
        missing = [p for p in el.get("properties", []) if p not in (old.get("properties") or [])]
        if missing:
            conflicts.append(f"edge `{el['name']}` in the graph has no propert(ies) {', '.join(missing)}")
    return conflicts


def _existing_schema(committer) -> Optional[Dict[str, Any]]:
    try:
        return committer.schema.getSchema()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        log.warning("Could not read the existing graph schema, skipping the conflict check: %s", exc)
        return None


def ensure_graph_exists() -> str:
    """Create the configured graph when the server lacks it; returns a note for the status, or "" if it existed.

    HugeGraph 1.7 and later keep graphs in graph spaces (DEFAULT unless HUGEGRAPH_GRAPHSPACE names another) and create
    one with POST /graphspaces/{space}/graphs/{name}. The new graph copies the configuration of another graph in the
    space when there is one, so it gets the same backend (hstore on a cluster, rocksdb standalone); otherwise it uses
    HUGEGRAPH_BACKEND (default rocksdb). An older server without graph spaces is left alone.
    """
    url = huge_settings.graph_url
    base = url if url.startswith("http") else f"http://{url}"
    space = huge_settings.graph_space or "DEFAULT"
    name = huge_settings.graph_name
    auth = (huge_settings.graph_user, huge_settings.graph_pwd)
    graphs_url = f"{base.rstrip('/')}/graphspaces/{quote(space)}/graphs"
    listing = requests.get(graphs_url, auth=auth, timeout=15)
    if listing.status_code == 404:
        return ""
    if not listing.ok:
        raise RuntimeError(
            f"Could not list the graphs in graph space `{space}`: {listing.status_code} {listing.text[:300]}"
        )
    graphs = sorted(listing.json().get("graphs", []))
    if name in graphs:
        return ""
    backend = setting("HUGEGRAPH_BACKEND").strip()
    if graphs and not backend:
        params, body = {"clone_graph_name": graphs[0]}, {}
        how = f"with the configuration of `{graphs[0]}`"
    else:
        backend = backend or "rocksdb"
        params = {}
        body = {
            "gremlin.graph": "org.apache.hugegraph.HugeFactory",
            "backend": backend,
            "serializer": "binary",
            "store": name,
        }
        how = f"with backend `{backend}`"
    created = requests.post(f"{graphs_url}/{quote(name)}", params=params, json=body, auth=auth, timeout=120)
    if not created.ok:
        raise RuntimeError(
            f"Graph `{name}` does not exist in graph space `{space}` and could not be created: "
            f"{created.status_code} {created.text[:300]}"
        )
    log.info("Created graph %s in graph space %s %s", name, space, how)
    return f"Created graph `{name}` in graph space `{space}` {how}."


def create_graph_schema(schema_text: str, committer=None) -> str:
    if not (schema_text or "").strip():
        raise gr.Error("The schema box is empty. Click *Infer schema* (or paste a schema) first.")
    try:
        schema = parse_schema(schema_text)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        raise gr.Error(f"Create schema failed: {exc}") from exc
    try:
        created_note = ensure_graph_exists()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        log.error("Create schema failed: %s", exc)
        raise gr.Error(f"Create schema failed: {exc}") from exc
    committer = committer or Commit2Graph()
    existing = _existing_schema(committer)
    conflicts = schema_conflicts(schema, existing)
    if conflicts:
        log.error("Create schema refused, it conflicts with the graph: %s", "; ".join(conflicts))
        raise gr.Error(
            "This schema conflicts with the one already in the graph, and HugeGraph would keep the old definitions, "
            "so imports would fail. Delete all graph data and schema (Reset, below) first, or rename these labels: "
            + "; ".join(conflicts)
        )
    try:
        committer.init_schema_if_need(schema)
        indexes = create_index_labels(committer.schema, schema)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        log.error("Create schema failed: %s", exc)
        raise gr.Error(f"Create schema failed: {exc}") from exc
    status = (
        f"Schema created in graph `{huge_settings.graph_name}` with "
        f"{len(schema['vertexlabels'])} vertex label(s), {len(schema['edgelabels'])} edge label(s) "
        f"and {len(indexes)} index label(s). Now upload documents below to ingest them."
    )
    new_labels = {vl["name"] for vl in schema["vertexlabels"]} | {el["name"] for el in schema["edgelabels"]}
    leftover = sorted(
        {item["name"] for key in ("vertexlabels", "edgelabels") for item in (existing or {}).get(key) or []}
        - new_labels
    )
    if leftover:
        status += (
            f"\n\n**Note:** the graph also still has {len(leftover)} label(s) from an earlier schema "
            f"({', '.join(f'`{name}`' for name in leftover)}) and their data. Use Reset to start from an empty graph."
        )
    return f"{created_note}\n\n{status}" if created_note else status


def existing_vertex_ids(ids: List[str]) -> set:
    if not ids:
        return set()
    # json.dumps yields double-quoted, escaped Gremlin string literals
    return {str(v) for v in _gremlin_data(f"g.V({', '.join(json.dumps(i) for i in ids)}).id()")}


def existing_primary_key_values(label: str, key: str) -> List[str]:
    return [str(v) for v in _gremlin_data(f"g.V().hasLabel({json.dumps(label)}).values({json.dumps(key)})")]


def drop_null_properties(extracted: Dict[str, Any]) -> int:
    """Remove null property values; HugeGraph rejects them and the importer would abort the whole file."""
    dropped = 0
    for item in extracted.get("vertices", []) + extracted.get("edges", []):
        props = item.get("properties")
        if not isinstance(props, dict):
            continue
        for key in [k for k, v in props.items() if v is None]:
            del props[key]
            dropped += 1
    return dropped


_INVALID = object()
# "$4.3 million", "approximately $373,000", "more than $500,000", "8 counts", "one count": an optional qualifier,
# a number or number word, an optional scale and an optional unit word.
_QUALIFIER = (
    r"(?:(?:approximately|approx\.?|about|around|roughly|nearly|almost|over|more than|less than|at least|up to|~)"
    r"\s+)?"
)
_NUMBER_WORDS = {
    w: n for n, w in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
        "seventeen eighteen nineteen twenty".split()
    )
}
_AMOUNT = re.compile(
    rf"^{_QUALIFIER}(?:US)?\$?\s*(-?[\d,]*\.?\d+|{'|'.join(_NUMBER_WORDS)})\s*(thousand|million|billion)?"
    r"(?:\s+[a-z]+)?$",
    re.IGNORECASE,
)
_SCALE = {"thousand": 1e3, "million": 1e6, "billion": 1e9}
_MONTH_NAMES = "january february march april may june july august september october november december".split()
# Full names and the usual abbreviations ("Sept." included), so "Mayday 3" is not read as a date.
_MONTHS = {name: i for i, full in enumerate(_MONTH_NAMES, start=1) for name in (full, full[:3])} | {"sept": 9}
_LONG_DATE = re.compile(r"^([a-z]+)\.?\s+(\d{1,2}),?\s+(\d{4})$", re.IGNORECASE)  # "July 14, 2026", "Sept. 9, 2026"
_US_DATE = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{4})$")  # "07/14/2026"


def _parse_date(value: str) -> Optional[str]:
    value = value.strip()
    if match := re.match(r"^(\d{4}-\d{2}-\d{2})(?:$|[ T])", value):
        return match[1]
    month = day = year = None
    if (match := _LONG_DATE.match(value)) and match[1].lower() in _MONTHS:
        month, day, year = _MONTHS[match[1].lower()], int(match[2]), int(match[3])
    elif match := _US_DATE.match(value):
        month, day, year = int(match[1]), int(match[2]), int(match[3])
    if month is None or not 1 <= day <= 31 or not 1 <= month <= 12:
        return None
    return f"{year:04d}-{month:02d}-{day:02d}"


def _coerce(data_type: str, value: Any) -> Any:
    """`value` in the Python type hugegraph-llm's importer requires for `data_type`, or _INVALID."""
    if data_type in ("DOUBLE", "FLOAT", "INT", "LONG", "BYTE"):
        number: Any = _INVALID
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            number = value
        elif isinstance(value, str) and (match := _AMOUNT.match(value.strip())):
            digits = _NUMBER_WORDS.get(match[1].lower(), match[1])
            number = float(str(digits).replace(",", "")) * _SCALE.get((match[2] or "").lower(), 1)
        if number is _INVALID:
            return _INVALID
        if data_type in ("DOUBLE", "FLOAT"):
            return float(number)
        return int(number) if float(number).is_integer() else _INVALID
    if data_type == "DATE":
        parsed = _parse_date(value) if isinstance(value, str) else None
        return parsed or _INVALID
    if data_type == "BOOLEAN":
        if isinstance(value, bool):
            return value
        return {"true": True, "false": False}.get(str(value).strip().lower(), _INVALID)
    if data_type == "TEXT" and isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return value


def coerce_property_types(extracted: Dict[str, Any], schema: Dict[str, Any]) -> Tuple[int, Counter]:
    """Convert property values to their schema types; drop the ones that cannot be converted.

    hugegraph-llm's importer skips a whole vertex when one property has the wrong Python type (931500 for a DOUBLE
    needs to be 931500.0), and every edge to that vertex then aborts the file with "Invalid vertex id". Returns
    (values converted, Counter of dropped 'label.property'). Primary keys are left alone.
    """
    types = {
        pk["name"]: str(pk.get("data_type", "TEXT")).upper()
        for pk in schema.get("propertykeys", [])
        if str(pk.get("cardinality", "SINGLE")).upper() == "SINGLE"
    }
    keys = {vl["name"]: set(vl.get("primary_keys", [])) for vl in schema.get("vertexlabels", [])}
    converted, dropped = 0, Counter()
    for item in extracted.get("vertices", []) + extracted.get("edges", []):
        props = item.get("properties")
        if not isinstance(props, dict):
            continue
        for key, value in list(props.items()):
            if key not in types or key in keys.get(item.get("label"), ()):
                continue
            fixed = _coerce(types[key], value)
            if fixed is _INVALID:
                del props[key]
                dropped[f"{item.get('label')}.{key}"] += 1
            elif fixed != value or type(fixed) is not type(value):
                props[key] = fixed
                converted += 1
    return converted, dropped


def merge_case_duplicates(
    extracted: Dict[str, Any], schema: Dict[str, Any], existing_values=existing_primary_key_values
) -> int:
    """Rewrite primary-key values that differ only in case to one canonical spelling.

    HugeGraph ids are case-sensitive, so 'alice chen' and 'Alice Chen' would become two vertices. The
    spelling already in the graph wins, then the first one seen in this batch. Edges inside the batch
    follow automatically because the importer maps extracted vertex ids to the ids HugeGraph assigns.
    """
    single_keys = {vl["name"]: vl["primary_keys"][0] for vl in schema["vertexlabels"] if len(vl["primary_keys"]) == 1}
    canonical: Dict[Tuple[str, str], str] = {}
    loaded = set()
    merged = 0
    for vertex in extracted.get("vertices", []):
        label, props = vertex.get("label"), vertex.get("properties") or {}
        key = single_keys.get(label)
        if key is None or not isinstance(props.get(key), str):
            continue
        if label not in loaded:
            loaded.add(label)
            for value in existing_values(label, key):
                canonical.setdefault((label, value.casefold()), value)
        value = props[key]
        spelling = canonical.setdefault((label, value.casefold()), value)
        if spelling != value:
            props[key] = spelling
            merged += 1
    return merged


NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}


def _name_tokens(value: str) -> List[str]:
    """Words of a proper name ('Dwayne A. Brown Jr.' -> Dwayne, A., Brown), or [] if it does not look like one."""
    tokens = value.replace(",", " ").split()
    while len(tokens) > 1 and tokens[-1].strip(".").casefold() in NAME_SUFFIXES:
        tokens.pop()
    if not tokens or not all(re.fullmatch(r"[^\W\d_][\w.'’-]*", t) and t[0].isupper() for t in tokens):
        return []
    return tokens


def drop_dangling_edges(extracted: Dict[str, Any], existing_ids=existing_vertex_ids) -> int:
    """Remove edges whose endpoints are neither extracted nor already in the graph.

    The LLM sometimes abbreviates an endpoint (e.g. '1:Alice' for 'Alice Chen'); HugeGraph rejects such an
    edge and the import would abort, losing every later edge of the document.
    """
    batch_ids = {str(v.get("id")) for v in extracted.get("vertices", [])}
    endpoints = {str(e.get(k)) for e in extracted.get("edges", []) for k in ("outV", "inV")}
    known = batch_ids | existing_ids(sorted(endpoints - batch_ids))
    edges = extracted.get("edges", [])
    kept = [e for e in edges if str(e.get("outV")) in known and str(e.get("inV")) in known]
    extracted["edges"] = kept
    return len(edges) - len(kept)


_EXTRACTION_CAPTURES: List[Tuple[Counter, List[str]]] = []
_CAPTURE_LOCK = threading.Lock()


@contextmanager
def capture_extractions(chunks: List[str]) -> Iterator[List[str]]:
    """Collect the raw LLM extraction responses to these chunks while the block runs (one per chunk).

    Documents are extracted concurrently, so each reply is routed by the chunk it answers. A chunk that two
    documents share goes to whichever capture claims it first, which is equivalent since the input is identical.
    """
    entry: Tuple[Counter, List[str]] = (Counter(chunks), [])
    with _CAPTURE_LOCK:
        _EXTRACTION_CAPTURES.append(entry)
    try:
        yield entry[1]
    finally:
        with _CAPTURE_LOCK:
            _EXTRACTION_CAPTURES[:] = [e for e in _EXTRACTION_CAPTURES if e is not entry]


def record_extraction(raw: str, chunk: str) -> None:
    with _CAPTURE_LOCK:
        for pending, captured in _EXTRACTION_CAPTURES:
            if pending[chunk] > 0:
                pending[chunk] -= 1
                captured.append(raw)
                return


# Every extraction call repeats the same instructions (~3k tokens, about 80% of its input) ahead of the document.
# Anthropic models cache a marked prefix, so later calls read it at a fraction of the input price. The prompt order is
# unchanged: the instructions, then the document and the schema.
# The cache lives 5 minutes from its last use; an import idle for longer than this re-warms it.
PROMPT_CACHE_WARM_SECS = 240
_CACHE_WARMED: Dict[Tuple[str, str], Tuple[threading.Event, float]] = {}
_CACHE_WARM_LOCK = threading.Lock()


def caches_prompts(llm) -> bool:
    return isinstance(llm, LiteLLMClient) and str(llm.model).startswith("anthropic/")


def warm_prompt_cache(llm, prefix: List[Dict[str, Any]]) -> None:
    """Write the cached prefix with one short call before the parallel ones, which would otherwise each write a copy."""
    key = (llm.model, prefix[0]["text"])
    with _CACHE_WARM_LOCK:
        entry = _CACHE_WARMED.get(key)
        owner = entry is None or time.monotonic() - entry[1] > PROMPT_CACHE_WARM_SECS
        if owner:
            entry = (threading.Event(), time.monotonic())
            _CACHE_WARMED[key] = entry
        else:
            _CACHE_WARMED[key] = (entry[0], time.monotonic())
    if not owner:
        entry[0].wait(timeout=60)
        return
    try:
        litellm.completion(
            model=llm.model,
            messages=[{"role": "user", "content": prefix + [{"type": "text", "text": "Reply OK."}]}],
            temperature=llm.temperature,
            max_tokens=16,
            api_key=llm.api_key,
            base_url=llm.api_base,
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        log.warning("Prompt cache warm-up failed, extracting without it: %s", exc)
    finally:
        entry[0].set()


# hugegraph-llm calls LiteLLM without a timeout, and LiteLLM's default is 6,000 seconds: one gateway call hung for 28
# minutes before failing with "Server disconnected". hugegraph-llm's own retry covers rate limits but not timeouts or
# dropped connections. The timeout has to leave room for the longest extraction reply (16,384 tokens).
LLM_TIMEOUT_SECS = 300
LLM_RETRIES = 2
TRANSIENT_LLM_ERRORS = (
    litellm.Timeout,
    litellm.APIConnectionError,
    litellm.InternalServerError,
    litellm.ServiceUnavailableError,
)


def install_llm_timeouts() -> None:
    """Give every LiteLLM call a timeout (LLM_TIMEOUT seconds) and retry LiteLLMClient.generate on transient errors.

    A timed-out or dropped call is tried again LLM_RETRIES times, after 1 s, 2 s, ... Streamed summaries get the
    timeout but no retry, since part of the text may already be on screen.
    """
    litellm.request_timeout = float(os.getenv("LLM_TIMEOUT", "").strip() or LLM_TIMEOUT_SECS)
    original = LiteLLMClient.generate
    if getattr(original, "_retries_transient", False):
        return

    def generate(self, messages=None, prompt=None):
        retries = max(0, int(os.getenv("LLM_RETRIES", "").strip() or LLM_RETRIES))
        for attempt in range(retries + 1):
            try:
                return original(self, messages=messages, prompt=prompt)
            except TRANSIENT_LLM_ERRORS as exc:
                if attempt == retries:
                    raise
                log.warning("LLM call failed (%s); retry %s of %s", type(exc).__name__, attempt + 1, retries)
                time.sleep(2**attempt)
        raise AssertionError("unreachable")

    wrapped = functools.wraps(original)(generate)
    wrapped._retries_transient = True  # type: ignore[attr-defined]
    LiteLLMClient.generate = wrapped


def install_last_gremlin_block() -> None:
    """Run the last gremlin block of a Text2Gremlin reply; hugegraph-llm runs the first.

    The model sometimes writes a query, notices a mistake (usually an edge direction between two vertices of the
    same label) and writes a corrected one below it: "Wait - the edge goes from the defendant to the prosecutor.
    Corrected query: ...". The first block is the one it rejected.
    """
    original = GremlinGenerateSynthesize._extract_response
    if getattr(original, "_takes_last_block", False):
        return

    def extract(self, response: str, label: str = "gremlin") -> str:
        blocks = re.findall(f"```{label}(.*?)```", response or "", re.DOTALL)
        if len(blocks) > 1:
            log.info("Text2Gremlin reply has %s query blocks; running the last, corrected one", len(blocks))
            return blocks[-1].strip()
        return original(self, response, label)

    wrapped = functools.wraps(original)(extract)
    wrapped._takes_last_block = True  # type: ignore[attr-defined]
    GremlinGenerateSynthesize._extract_response = wrapped


def install_text_filter_rewrite() -> None:
    """Rewrite TextP filters in every generated query before hugegraph-llm runs it, for servers older than 1.8.

    Install after install_last_gremlin_block, so the query that runs is the rewritten last block.
    """
    original = GremlinGenerateSynthesize._extract_response
    if getattr(original, "_rewrites_text_filters", False):
        return

    def extract(self, response: str, label: str = "gremlin") -> str:
        gremlin = original(self, response, label)
        return rewrite_text_filters(gremlin) if needs_text_filter_rewrite() else gremlin

    wrapped = functools.wraps(original)(extract)
    wrapped._rewrites_text_filters = True  # type: ignore[attr-defined]
    GremlinGenerateSynthesize._extract_response = wrapped


def install_prompt_caching() -> None:
    """Send the extraction instructions as a cached block ahead of the document; other providers are unchanged.

    Install first, so the split retry and the recorder wrap the cached call.
    """
    original = PropertyGraphExtract.extract_property_graph_by_llm
    if getattr(original, "_caches_prompt", False):
        return

    def extract(self, schema, chunk):
        if not self.example_prompt or not caches_prompts(self.llm):
            return original(self, schema, chunk)
        prefix = [{"type": "text", "text": self.example_prompt, "cache_control": {"type": "ephemeral"}}]
        warm_prompt_cache(self.llm, prefix)
        document = {"type": "text", "text": generate_extract_property_graph_prompt(chunk, schema)}
        return self.llm.generate(messages=[{"role": "user", "content": prefix + [document]}])

    wrapped = functools.wraps(original)(extract)
    wrapped._caches_prompt = True  # type: ignore[attr-defined]
    PropertyGraphExtract.extract_property_graph_by_llm = wrapped


def install_split_retry() -> None:
    """Re-extract a chunk in smaller pieces when the reply holds no valid graph JSON, and return one combined reply.

    A dense chunk can hit the model's output-token limit, which cuts the JSON off and would lose the whole chunk.
    Pieces are halved until they parse or reach CHUNK_CHARS. Install before install_extraction_recorder so the audit
    sees the combined reply.
    """
    original = PropertyGraphExtract.extract_property_graph_by_llm
    if getattr(original, "_splits_on_failure", False):
        return

    def extract(self, schema, chunk):
        raw = original(self, schema, chunk)
        if parse_llm_graph(raw) is not None or len(chunk) <= CHUNK_CHARS:
            return raw
        limit = max(CHUNK_CHARS, -(-len(chunk) // 2))
        pieces = pack_chunks(chunk, limit)
        if len(pieces) < 2:
            # PDF text has no blank lines, so the chunk is one block; split it at line breaks instead.
            pieces = pack_chunks(chunk.replace("\n", "\n\n"), limit)
        if len(pieces) < 2:
            return raw
        log.warning("Extraction of a %s-char chunk gave no valid JSON; retrying as %s pieces", len(chunk), len(pieces))
        graphs = [parse_llm_graph(extract(self, schema, piece)) for piece in pieces]
        graphs = [g for g in graphs if g is not None]
        if not graphs:
            return raw
        return json.dumps(
            {"vertices": [v for g in graphs for v in g["vertices"]], "edges": [e for g in graphs for e in g["edges"]]},
            ensure_ascii=False,
        )

    wrapped = functools.wraps(original)(extract)
    wrapped._splits_on_failure = True  # type: ignore[attr-defined]
    PropertyGraphExtract.extract_property_graph_by_llm = wrapped


def install_extraction_recorder() -> None:
    """Hand each raw extraction response to record_extraction before hugegraph-llm filters it."""
    original = PropertyGraphExtract.extract_property_graph_by_llm
    if getattr(original, "_records_extraction", False):
        return

    @functools.wraps(original)
    def recording(self, schema, chunk):
        raw = original(self, schema, chunk)
        record_extraction(raw, chunk)
        return raw

    recording._records_extraction = True  # type: ignore[attr-defined]
    PropertyGraphExtract.extract_property_graph_by_llm = recording


def extract_workers() -> int:
    """Concurrent extraction LLM calls across all documents: EXTRACT_WORKERS, else 1 for Ollama, more if hosted."""
    override = os.getenv("EXTRACT_WORKERS", "").strip()
    if override:
        return max(1, int(override))
    return 1 if llm_settings.chat_llm_type == "ollama/local" else HOSTED_EXTRACT_WORKERS


_EXTRACT_POOL: Dict[str, Any] = {"pool": None, "size": 0}
_EXTRACT_POOL_LOCK = threading.Lock()


def extract_pool() -> ThreadPoolExecutor:
    """The pool every extraction LLM call runs on, so concurrent documents share the extract_workers() limit.

    It is rebuilt when the limit changes (e.g. after changing provider); calls already queued still run.
    """
    workers = extract_workers()
    with _EXTRACT_POOL_LOCK:
        if _EXTRACT_POOL["size"] != workers:
            if _EXTRACT_POOL["pool"] is not None:
                _EXTRACT_POOL["pool"].shutdown(wait=False)
            _EXTRACT_POOL.update(pool=ThreadPoolExecutor(max_workers=workers, thread_name_prefix="extract"), size=workers)
        return _EXTRACT_POOL["pool"]


def install_parallel_extraction() -> None:
    """Run PropertyGraphExtract's per-chunk LLM calls concurrently; hugegraph-llm calls them one after another.

    Mirrors PropertyGraphExtract.run, keeping chunk order so the merged result matches a sequential run.
    hugegraph-llm's ExtractNode hands the extractor the chat LLM, so the extract-role LLM is swapped in here.
    """
    if getattr(PropertyGraphExtract.run, "_parallel", False):
        return

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        schema, chunks = context["schema"], context["chunks"]
        self.llm = get_extract_llm(llm_settings)
        context.setdefault("vertices", [])
        context.setdefault("edges", [])
        outputs = list(extract_pool().map(lambda chunk: self.extract_property_graph_by_llm(schema, chunk), chunks))
        items = []
        for chunk, output in zip(chunks, outputs):
            log.debug("[LLM] %s input: %s \n output:%s", type(self).__name__, chunk, output)
            items.extend(self._extract_and_filter_label(schema, output))
        for item in filter_item(schema, items):
            if item["type"] == "vertex":
                context["vertices"].append(item)
            elif item["type"] == "edge":
                context["edges"].append(item)
        context["call_count"] = context.get("call_count", 0) + len(chunks)
        return context

    run._parallel = True  # type: ignore[attr-defined]
    PropertyGraphExtract.run = run


def parse_llm_graph(text: str) -> Optional[Dict[str, List[Any]]]:
    """Parse an extraction response the way hugegraph-llm does; None when it holds no usable graph JSON."""
    text = re.sub(r"```\w*\n?", "", text or "").replace("```", "").strip()
    match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if not match:
        return None
    try:
        graph = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    if isinstance(graph, list):
        items = [i for i in graph if isinstance(i, dict)]
        graph = {
            "vertices": [i for i in items if i.get("type") == "vertex"],
            "edges": [i for i in items if i.get("type") == "edge"],
        }
    if not isinstance(graph, dict) or "vertices" not in graph or "edges" not in graph:
        return None
    return graph


def audit_extraction(raw_outputs: List[str], schema: Dict[str, Any]) -> Dict[str, Any]:
    """Count what hugegraph-llm will silently drop from the raw responses because it does not fit the schema."""
    vertex_props = {v["name"]: set(v.get("properties", [])) for v in schema["vertexlabels"]}
    edge_specs = {e["name"]: e for e in schema["edgelabels"]}
    drops: Dict[str, Any] = {
        "unparsed": 0,
        "malformed": 0,
        "labels": Counter(),
        "properties": Counter(),
        "no_properties": Counter(),
        "endpoints": Counter(),
        "unresolved": 0,
    }
    for raw in raw_outputs:
        graph = parse_llm_graph(raw)
        if graph is None:
            drops["unparsed"] += 1
            continue
        vertex_ids = set()
        for vertex in graph["vertices"]:
            if not isinstance(vertex, dict) or "label" not in vertex or "properties" not in vertex:
                drops["malformed"] += 1
                continue
            label = vertex["label"]
            if label not in vertex_props:
                drops["labels"][f"vertex '{label}'"] += 1
                continue
            vertex_ids.add(vertex.get("id"))
            for key in vertex["properties"] or {}:
                if key not in vertex_props[label]:
                    drops["properties"][f"{label}.{key}"] += 1
        for edge in graph["edges"]:
            if not isinstance(edge, dict) or "label" not in edge:
                drops["malformed"] += 1
                continue
            if "properties" not in edge:
                drops["no_properties"][edge["label"]] += 1
                continue
            spec = edge_specs.get(edge["label"])
            if spec is None:
                drops["labels"][f"edge '{edge['label']}'"] += 1
                continue
            out_label, in_label = edge.get("outVLabel"), edge.get("inVLabel")
            if (out_label, in_label) != (spec["source_label"], spec["target_label"]):
                drops["endpoints"][f"{edge['label']} {out_label}->{in_label}"] += 1
                continue
            if edge.get("outV") not in vertex_ids or edge.get("inV") not in vertex_ids:
                drops["unresolved"] += 1
                continue
            for key in edge.get("properties") or {}:
                if key not in spec.get("properties", []):
                    drops["properties"][f"{edge['label']}.{key}"] += 1
    return drops


def dropped_item_count(drops: Dict[str, Any]) -> int:
    """Vertices/edges discarded entirely (dropped properties and unparsed chunks are reported separately)."""
    return (
        drops["malformed"]
        + sum(drops["labels"].values())
        + sum(drops["no_properties"].values())
        + sum(drops["endpoints"].values())
        + drops["unresolved"]
    )


def format_drops(drops: Dict[str, Any]) -> str:
    def counted(counter: Counter) -> str:
        return ", ".join(f"{name} x{n}" for name, n in counter.most_common())

    parts = []
    if drops["labels"]:
        parts.append(f"labels not in schema: {counted(drops['labels'])}")
    if drops["properties"]:
        parts.append(f"properties not in schema: {counted(drops['properties'])}")
    if drops["no_properties"]:
        parts.append(f"edges without a properties key: {counted(drops['no_properties'])}")
    if drops["endpoints"]:
        parts.append(f"edges with wrong endpoint labels: {counted(drops['endpoints'])}")
    if drops["unresolved"]:
        parts.append(f"{drops['unresolved']} edge(s) pointing at vertices missing from the same chunk")
    if drops["malformed"]:
        parts.append(f"{drops['malformed']} malformed item(s)")
    if drops["unparsed"]:
        parts.append(f"{drops['unparsed']} chunk(s) without valid graph JSON")
    return "; ".join(parts)


def extract_file(path: str, schema: str, hints: str, scheduler) -> Tuple[Dict[str, Any], List[str]]:
    """Extract one document's vertices/edges; also returns the raw LLM replies for the schema-drop audit."""
    chunks = pack_chunks(read_file_text(path), chunk_chars())
    with capture_extractions(chunks) as raw_outputs:
        extracted = json.loads(
            scheduler.schedule_flow(
                FlowName.GRAPH_EXTRACT,
                schema,
                chunks,
                extraction_prompt(hints),
                "property_graph",
                split_type=SPLIT_TYPE_DOCUMENT,  # chunks are already packed above
                language="en",
            )
        )
    return extracted, raw_outputs


def empty_labels(schema: Dict[str, Any], imported: Counter) -> List[str]:
    """Schema labels that received no items, e.g. a vertex keyed on an ID most documents lack, and its edges."""
    labels = [f"vertex `{vl['name']}`" for vl in schema.get("vertexlabels", []) if not imported[vl["name"]]]
    return labels + [f"edge `{el['name']}`" for el in schema.get("edgelabels", []) if not imported[el["name"]]]


def ingest_files(
    files,
    schema_text: str,
    hints: str = "",
    scheduler=None,
    existing_ids=existing_vertex_ids,
    existing_values=existing_primary_key_values,
) -> Iterable[str]:
    """Extract vertices/edges from the files concurrently, then import them in order; yields a running report."""
    paths = _as_paths(files)
    if not paths:
        raise gr.Error("Please upload at least one document.")
    try:
        schema_dict = parse_schema(schema_text)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    schema = json.dumps(schema_dict, ensure_ascii=False)

    scheduler = scheduler or SchedulerSingleton.get_instance()
    report = [f"Ingesting {len(paths)} file(s)...\n"]
    report += [f"- `{Path(path).name}`: extracting..." for path in paths]
    totals = {"vertices": 0, "edges": 0, "skipped": 0, "merged": 0, "failed": 0, "dropped": 0}
    imported: Counter = Counter()
    yield "\n".join(report)
    # Documents are extracted concurrently but imported one at a time in upload order, because case merging and the
    # dangling-edge check look up what earlier documents already put in the graph.
    pool = ThreadPoolExecutor(max_workers=min(len(paths), extract_workers()), thread_name_prefix="ingest")
    futures = [pool.submit(extract_file, path, schema, hints, scheduler) for path in paths]
    try:
        for line, (path, future) in enumerate(zip(paths, futures), start=1):
            name = Path(path).name
            drop_note = ""
            try:
                extracted, raw_outputs = future.result()
                # hugegraph-llm discards whatever does not fit the schema without reporting it; surface that here.
                drops = audit_extraction(raw_outputs, schema_dict)
                drop_note = format_drops(drops)
                totals["dropped"] += dropped_item_count(drops)
                if drop_note:
                    log.warning("Schema dropped extracted data from %s: %s", name, drop_note)
                n_vertices, n_edges = len(extracted.get("vertices", [])), len(extracted.get("edges", []))
                if not n_vertices and not n_edges:
                    report[line] = f"- `{name}`: nothing extracted (does the document match the schema?)"
                    totals["failed"] += 1
                    continue
                nulls = drop_null_properties(extracted)
                if nulls:
                    log.info("Dropped %s null property value(s) from %s", nulls, name)
                converted, bad_types = coerce_property_types(extracted, schema_dict)
                if converted:
                    log.info("Converted %s property value(s) to their schema types in %s", converted, name)
                merged = merge_case_duplicates(extracted, schema_dict, existing_values)
                skipped = drop_dangling_edges(extracted, existing_ids)
                n_edges -= skipped
                scheduler.schedule_flow(FlowName.IMPORT_GRAPH_DATA, extracted, schema)
                imported.update(item.get("label") for item in extracted.get("vertices", []) + extracted["edges"])
                totals["vertices"] += n_vertices
                totals["edges"] += n_edges
                totals["skipped"] += skipped
                totals["merged"] += merged
                report[line] = f"- `{name}`: imported {n_vertices} vertices, {n_edges} edges"
                notes = []
                if bad_types:
                    notes.append(
                        f"dropped {sum(bad_types.values())} property value(s) of the wrong type "
                        f"({', '.join(f'{k} x{n}' for k, n in bad_types.most_common())})"
                    )
                if merged:
                    notes.append(f"merged {merged} name(s) differing only in case")
                if skipped:
                    notes.append(f"skipped {skipped} edge(s) pointing to unknown vertices")
                if notes:
                    report[line] += f" ({'; '.join(notes)})"
            except Exception as exc:  # pylint: disable=broad-exception-caught
                log.error("Ingest of %s failed: %s", name, exc)
                totals["failed"] += 1
                report[line] = f"- `{name}`: failed - {exc}"
            finally:
                if drop_note:
                    report[line] += f"\n  - dropped by the schema: {drop_note}"
            yield "\n".join(report)
    finally:
        # If the UI abandons the run, documents not yet started are skipped instead of extracted for nothing.
        pool.shutdown(wait=False, cancel_futures=True)

    report.append(
        f"\n**Done.** {totals['vertices']} vertices and {totals['edges']} edges imported "
        f"({totals['merged']} case duplicate(s) merged, "
        f"{totals['skipped']} dangling edge(s) skipped); "
        f"{totals['failed']} file(s) failed or empty. Open the **Explore** tab to query the graph."
    )
    if totals["dropped"]:
        report[-1] += (
            f"\n\n**Warning:** {totals['dropped']} extracted vertex/edge item(s) did not fit the schema and were "
            "dropped (see the per-file notes). If that matters, extend the schema."
        )
    unused = empty_labels(schema_dict, imported)
    if unused and totals["vertices"] + totals["edges"]:
        report[-1] += (
            f"\n\n**Warning:** {len(unused)} schema label(s) got nothing from these documents: "
            f"{', '.join(unused)}. Questions that rely on them will return empty results. Common causes: a vertex "
            "keyed on a property most documents lack (its edges vanish with it), or a label that fits only the "
            "sample document. Revise the schema, or re-infer it from several samples."
        )
    yield "\n".join(report)


SCHEMA_REMOVE_WAIT_SECS = 120


def remove_leftover_schema(sleep=time.sleep) -> None:
    """Delete the labels and property keys a clear left behind.

    On hstore the schema lives in PD meta and survives the clear API. Label removal runs as a server task, and
    HugeGraph refuses to remove a vertex label an edge label links or a property key a label uses, so each kind
    waits until the previous one is gone. Removing a label also removes its index labels.
    """
    schema = get_hg_client().schema()
    removers = (("edgelabels", schema.edgeLabel), ("vertexlabels", schema.vertexLabel),
                ("propertykeys", schema.propertyKey))
    for kind, label in removers:
        names = [item["name"] for item in (schema.getSchema() or {}).get(kind, [])]
        for name in names:
            label(name).remove()
        waited = 0
        while names and (schema.getSchema() or {}).get(kind):
            if waited >= SCHEMA_REMOVE_WAIT_SECS:
                raise RuntimeError(f"the server still has {kind} after {SCHEMA_REMOVE_WAIT_SECS} s")
            sleep(1)
            waited += 1


def clear_graph() -> str:
    try:
        clean_hg_data()
        remove_leftover_schema()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        raise gr.Error(f"Clear failed: {exc}") from exc
    return (
        f"All data **and schema** in `{huge_settings.graph_name}` were deleted. "
        "Click *Create graph with this schema* again before ingesting."
    )


# ---------------------------------------------------------------------------
# Tab 2: explore & query
# ---------------------------------------------------------------------------
def _gremlin_data(query: str) -> List[Any]:
    return run_gremlin_query(query, fmt=False).get("data", [])


def _display_name(vertex: Dict[str, Any]) -> str:
    props = vertex.get("properties") or {}
    for key in ("name", "title", "id"):
        if props.get(key) not in (None, ""):
            return str(props[key])
    return str(vertex.get("id"))


def build_graph_view(vertices: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
    """Convert HugeGraph vertices/edges into vis-network nodes/edges."""
    nodes = {
        str(v["id"]): {
            "id": str(v["id"]),
            "label": _display_name(v),
            "group": v.get("label"),
            "title": f"{v.get('label')}: {json.dumps(v.get('properties', {}), ensure_ascii=False)}",
        }
        for v in vertices
    }
    links = []
    for e in edges:
        src, dst = str(e.get("outV")), str(e.get("inV"))
        for vid, vlabel in ((src, e.get("outVLabel")), (dst, e.get("inVLabel"))):
            # Edges may reference vertices outside the sampled set; show them as bare nodes.
            nodes.setdefault(vid, {"id": vid, "label": vid.split(":", 1)[-1], "group": vlabel, "title": vid})
        links.append({"from": src, "to": dst, "label": e.get("label"), "arrows": "to"})
    return {"nodes": list(nodes.values()), "edges": links}


def render_graph_html(view: Dict[str, List[Dict]]) -> str:
    if not view["nodes"]:
        return "<p>The graph is empty. Ingest some documents first.</p>"
    page = f"""<!doctype html><html><head><meta charset="utf-8">
<script src="https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
<style>html,body,#g{{margin:0;height:100%;font-family:sans-serif}}</style></head>
<body><div id="g"></div><script>
const data = {json.dumps(view, ensure_ascii=False)};
new vis.Network(document.getElementById("g"),
  {{nodes: new vis.DataSet(data.nodes), edges: new vis.DataSet(data.edges)}},
  {{nodes: {{shape: "dot", size: 14}}, edges: {{font: {{size: 10, align: "middle"}}, smooth: true}},
    physics: {{stabilization: {{iterations: 150}}}}}});
</script></body></html>"""
    return f'<iframe style="width:100%;height:520px;border:0" srcdoc="{html.escape(page)}"></iframe>'


def _count_table(rows: List[Any]) -> List[List[Any]]:
    counts = rows[0] if rows and isinstance(rows[0], dict) else {}
    return sorted(([label, n] for label, n in counts.items()), key=lambda r: -r[1])


def load_overview(sample_size: int = 100) -> Tuple[str, List[List[Any]], List[List[Any]], str]:
    try:
        vertex_counts = _count_table(_gremlin_data("g.V().groupCount().by(label)"))
        edge_counts = _count_table(_gremlin_data("g.E().groupCount().by(label)"))
        n = int(sample_size)
        vertices = _gremlin_data(f"g.V().limit({n})")
        edges = _gremlin_data(f"g.E().limit({n * 2})")
    except Exception as exc:  # pylint: disable=broad-exception-caught
        raise gr.Error(f"Could not read graph `{huge_settings.graph_name}`: {exc}") from exc
    totals = (
        f"**{sum(r[1] for r in vertex_counts)}** vertices across {len(vertex_counts)} label(s), "
        f"**{sum(r[1] for r in edge_counts)}** edges across {len(edge_counts)} label(s)."
    )
    return totals, vertex_counts, edge_counts, render_graph_html(build_graph_view(vertices, edges))


def pick_execution(res: Dict[str, Any]) -> Tuple[str, Optional[Any]]:
    """Choose the first generated Gremlin whose execution succeeded and returned data."""
    fallback: Tuple[str, Optional[Any]] = (res.get("template_gremlin") or res.get("raw_gremlin") or "", None)
    for gremlin_key, exec_key in (
        ("template_gremlin", "template_execution_result"),
        ("raw_gremlin", "raw_execution_result"),
    ):
        gremlin, executed = res.get(gremlin_key), res.get(exec_key)
        if not gremlin or not executed:
            continue
        try:
            parsed = json.loads(executed)
        except (TypeError, json.JSONDecodeError):
            continue  # an error message, not a result
        data = parsed.get("data") if isinstance(parsed, dict) else None
        if data:
            return gremlin, data
        if fallback[1] is None:
            fallback = (gremlin, data if data is not None else [])
    return fallback


def _ignore_case_predicate(quote: str, text: str, case_variants: bool) -> Optional[str]:
    """A predicate matching text anywhere in a value, in any letter case; None when text cannot be quoted."""
    if case_variants:
        # Servers before 1.8 run a Gremlin without TextP.regex, so try the usual spellings with TextP.containing.
        variants = dict.fromkeys(
            (text, text.lower(), text.upper(), text.title(), text[:1].upper() + text[1:].lower())
        )
        return ".or(".join(f"TextP.containing({quote}{v}{quote})" for v in variants) + ")" * (len(variants) - 1)
    if "\\E" in text:  # would end the \Q...\E quoting early
        return None
    return f"TextP.regex({quote}(?i).*\\\\Q{text}\\\\E.*{quote})"


def _ignore_case_partial(match: "re.Match[str]", case_variants: bool = False) -> str:
    quote, text = (match[3], match[4]) if match[3] else (match[5], match[6])
    predicate = _ignore_case_predicate(quote, text, case_variants)
    if predicate is None:
        return match[0]
    key = f"{match[1]}{match[2]}{match[1]}"
    # The filter(...) form keeps the text predicate out of the backend query on servers before 1.8.
    return f".filter(__.values({key}).is({predicate}))" if case_variants else f".has({key}, {predicate})"


def _ignore_case_is(match: "re.Match[str]", case_variants: bool = False) -> str:
    predicate = _ignore_case_predicate(match[1], match[2], case_variants)
    return match[0] if predicate is None else f".is({predicate})"


def server_version() -> Optional[Tuple[int, ...]]:
    """The configured HugeGraph server's version, asked once per URL; None when it cannot be read."""
    url = huge_settings.graph_url
    if url not in _SERVER_VERSIONS:
        base = url if url.startswith("http") else f"http://{url}"
        try:
            core = requests.get(f"{base.rstrip('/')}/versions", timeout=5).json()["versions"]["core"]
            match = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", core)
            _SERVER_VERSIONS[url] = tuple(int(part or 0) for part in match.groups()) if match else None
        except Exception as exc:  # pylint: disable=broad-exception-caught
            log.warning("Could not read the HugeGraph server version: %s", exc)
            _SERVER_VERSIONS[url] = None
    return _SERVER_VERSIONS[url]


def needs_text_filter_rewrite() -> bool:
    """True for servers older than 1.8, or of unknown version: the rewritten query returns the same results."""
    version = server_version()
    return version is None or version < TEXT_FILTER_SERVER_VERSION


def rewrite_text_filters(gremlin: str) -> str:
    """Turn has(key, TextP...) filters into filter(__.values(key).is(TextP...)) for servers older than 1.8."""
    return TEXT_PREDICATE_FILTER.sub(
        lambda match: f"filter(__.values({match[1]}{match[2]}{match[1]}).is({match[3]}))", gremlin or ""
    )


def relax_exact_matches(gremlin: str, case_variants: bool = False) -> Optional[str]:
    """The query with each text filter, exact or TextP.containing, turned into a case-insensitive partial match.

    None when the query has no such filter. Used to retry an empty result, since stored names often carry words the
    question leaves out ('TD Bank' vs 'TD Bank, N.A.') or are written in other letter case ('JAMES KEATING',
    'bank fraud' vs 'Bank Fraud'). The text is quoted with \\Q...\\E, so regex characters in it match literally.
    With case_variants, for servers without TextP.regex, it matches the text as written, in lower and upper case,
    and capitalized.
    """
    relaxed = EXACT_TEXT_FILTER.sub(functools.partial(_ignore_case_partial, case_variants=case_variants), gremlin)
    relaxed = IS_TEXT_CONTAINS.sub(functools.partial(_ignore_case_is, case_variants=case_variants), relaxed)
    return relaxed if relaxed != gremlin else None


def retry_with_partial_matches(gremlin: str, run_query=None) -> Tuple[Optional[str], Any]:
    """Run the relaxed form of an empty-result query; (relaxed query, data) when it finds something."""
    old_server = needs_text_filter_rewrite()
    relaxed = relax_exact_matches(gremlin or "", case_variants=old_server)
    if not relaxed:
        return None, None
    if old_server:
        relaxed = rewrite_text_filters(relaxed)
    query = relaxed if "limit(" in relaxed.lower() else f"{relaxed}.limit({QUERY_LIMIT})"
    try:
        data = (run_query or _gremlin_data)(query)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        log.warning("Partial-match retry failed: %s", exc)
        return None, None
    return (relaxed, data) if data else (None, None)


def build_summary_prompt(question: str, gremlin: str, data: Any) -> str:
    rows = data if isinstance(data, list) else [data]
    shown = rows[:SUMMARY_MAX_ROWS]
    result = json.dumps(shown, ensure_ascii=False, default=str)
    if len(result) > MAX_RESULT_CHARS:
        result = result[:MAX_RESULT_CHARS] + " ...(truncated)"
    return SUMMARY_PROMPT.format(
        question=question,
        gremlin=gremlin,
        result=result,
        shown=len(shown),
        total=len(rows),
        list_items=SUMMARY_LIST_ITEMS,
    )


def showcase_gremlin_prompt() -> str:
    # Escape braces: the flow fills the prompt with str.format.
    rules = GREMLIN_RULES.replace("{", "{{").replace("}", "}}")
    base = COMPLEX_QUERY_SECTION.sub("", prompt.gremlin_generate_prompt)
    base = base.replace(" (Executed only if the query is not complex)", "")
    marker = "Graph Schema:"
    if marker in base:
        return base.replace(marker, rules + marker, 1)
    return rules + base


def fetch_graph_schema() -> Optional[Dict[str, Any]]:
    try:
        return get_hg_client().schema().getSchema()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        log.warning("Could not load graph schema for query templates: %s", exc)
        return None


def _token_text(token: Any) -> str:
    """Streaming callbacks pass a str (Ollama, OpenAI) or a LiteLLM chunk object."""
    if isinstance(token, str):
        return token
    choices = getattr(token, "choices", None)
    if choices:
        return getattr(choices[0].delta, "content", None) or ""
    return ""


def stream_llm_text(llm, prompt: str) -> Iterator[str]:
    """Yield the reply as it grows. hugegraph-llm clients stream differently (a token generator for Ollama and
    OpenAI, a blocking call with a per-chunk callback for LiteLLM), so tokens are collected through the
    callback while a worker thread drives the call. Clients without streaming fall back to one generate()."""
    streaming = getattr(llm, "generate_streaming", None)
    if streaming is None:
        yield llm.generate(prompt=prompt)
        return
    tokens: "queue.Queue[Any]" = queue.Queue()
    done = object()

    def on_token(token):
        text = _token_text(token)
        if text:
            tokens.put(text)

    def run():
        try:
            result = streaming(prompt=prompt, on_token_callback=on_token)
            if inspect.isgenerator(result):
                for _ in result:  # the callback already receives each token
                    pass
        except Exception as exc:  # pylint: disable=broad-exception-caught
            tokens.put(exc)
        finally:
            tokens.put(done)

    threading.Thread(target=run, daemon=True).start()
    text = ""
    while True:
        item = tokens.get()
        batch = [item]
        while item is not done and not isinstance(item, Exception):  # drain what arrived meanwhile
            try:
                item = tokens.get_nowait()
            except queue.Empty:
                break
            batch.append(item)
        for piece in batch:
            if isinstance(piece, Exception):
                raise piece
        text += "".join(p for p in batch if isinstance(p, str))
        if batch[-1] is done:
            break
        yield text
    yield text or "(The model returned an empty summary.)"


def stream_answer(
    question: str, scheduler=None, llm=None, schema=None, run_query=None
) -> Iterator[Tuple[str, str, Any]]:
    """Yield (answer, gremlin, raw result): the query and raw result first, then the summary as it is written."""
    if not question or not question.strip():
        raise gr.Error("Please type a question.")
    scheduler = scheduler or SchedulerSingleton.get_instance()
    schema = schema if schema is not None else fetch_graph_schema()
    try:
        res = scheduler.schedule_flow(
            FlowName.TEXT2GREMLIN,
            question,
            0,  # no few-shot examples: hugegraph-llm's bundled ones are about movies, and the rules cover the rest
            huge_settings.graph_name,  # fetch the live schema from HugeGraph
            showcase_gremlin_prompt(),
            ["template_gremlin", "raw_gremlin", "template_execution_result", "raw_execution_result"],
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        raise gr.Error(f"Text2Gremlin failed: {exc}") from exc

    gremlin, data = pick_execution(res)
    if DECLINED_GREMLIN.match(gremlin or ""):
        yield (
            "The model did not write a query for this question (it answered `g.V().limit(0)`). "
            "Try splitting it into simpler questions, or use a larger model.",
            gremlin,
            None,
        )
        return
    if data is None:
        errors = res.get("template_execution_result") or res.get("raw_execution_result") or "no query generated"
        yield f"The generated Gremlin could not be executed:\n\n```\n{errors}\n```", gremlin, None
        return
    note = ""
    if not data:
        relaxed, relaxed_data = retry_with_partial_matches(gremlin, run_query)
        if not relaxed:
            yield "The query ran but returned no results. Try rephrasing the question.", gremlin, data
            return
        gremlin, data = relaxed, relaxed_data
        note = "_No exact match for a name in the question, so it was matched partially, ignoring letter case._\n\n"
    yield f"{note}_Found {len(data)} result(s); summarizing..._", gremlin, data
    try:
        for summary in stream_llm_text(llm or get_chat_llm(llm_settings), build_summary_prompt(question, gremlin, data)):
            yield note + summary, gremlin, data
    except Exception as exc:  # pylint: disable=broad-exception-caught
        log.error("Summarization failed: %s", exc)
        yield f"{note}(Summarization failed: {exc}) Raw results are shown below.", gremlin, data


def answer_question(
    question: str, scheduler=None, llm=None, schema=None, run_query=None
) -> Tuple[str, str, Any]:
    """Non-streaming form of stream_answer: the final (answer, gremlin, raw result)."""
    final: Tuple[str, str, Any] = ("", "", None)
    for final in stream_answer(question, scheduler, llm, schema, run_query):
        pass
    return final


# Follow-ups are rewritten into standalone questions from the last few turns, so Text2Gremlin still sees one
# self-contained question. Raw results are included because summaries list only a sample.
CONTEXT_TURNS = 3
CONTEXT_RESULT_CHARS = 1500
FOLLOW_UP_PROMPT = """You rewrite a follow-up question about a knowledge graph into a standalone question.

Earlier questions, answers and query results (oldest first):
{turns}

Follow-up question: {question}

Rewrite the follow-up so it can be understood without the conversation: replace references such as "he", "she",
"they", "them", "that company", "those people" or "the same city" with the names they refer to, taken from the
answers and results above and spelled exactly as written there. Keep the rest of the question's wording. If the
question is already standalone, return it unchanged. Reply with the question only, on one line."""


def _context_turn(number: int, item: Dict[str, Any]) -> str:
    results = json.dumps(item.get("raw"), ensure_ascii=False, default=str)
    if len(results) > CONTEXT_RESULT_CHARS:
        results = results[:CONTEXT_RESULT_CHARS] + "...(truncated)"
    question = item.get("standalone") or item["question"]
    return f"Q{number}: {question}\nA{number}: {item['answer']}\nResults{number}: {results}"


def standalone_question(question: str, history: List[Dict[str, Any]], llm=None) -> str:
    """The follow-up rewritten with names from recent answers; the question as asked without history or on error."""
    if not history:
        return question
    turns = "\n\n".join(_context_turn(n, item) for n, item in enumerate(history[-CONTEXT_TURNS:], start=1))
    try:
        reply = (llm or get_chat_llm(llm_settings)).generate(
            prompt=FOLLOW_UP_PROMPT.format(turns=turns, question=question)
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        log.warning("Could not rewrite the follow-up question, asking it as written: %s", exc)
        return question
    lines = [line.strip().strip("`\"'").strip() for line in (reply or "").splitlines() if line.strip()]
    return lines[0] if lines else question


def ask_with_history(
    question: str,
    history: List[Dict[str, Any]],
    use_context: bool = True,
    answer_fn=stream_answer,
    rewrite_fn=standalone_question,
) -> Iterator[Tuple]:
    """Stream (answer, gremlin, raw, history, question box) for the Explore tab's question-and-answer list.

    With use_context, a follow-up is first rewritten into a standalone question from the previous answers, and the
    answer says how it was interpreted. The answer streams into the live area under the question box; once complete
    it moves into the history as its own card and the box is cleared for the next question.
    """
    asked = (question or "").strip()
    standalone = asked
    if asked and use_context and history:
        yield "_Reading the follow-up against the previous answers..._", "", None, gr.skip(), gr.skip()
        standalone = rewrite_fn(asked, history)
    note = f"_Interpreted as: {standalone}_\n\n" if standalone != asked else ""
    final = None
    for final in answer_fn(standalone or question):
        yield (note + final[0], final[1], final[2], gr.skip(), gr.skip())
    if final is None:
        return
    answer, gremlin, data = final
    entry = {"question": asked, "answer": note + answer, "gremlin": gremlin, "raw": data}
    if note:
        entry["standalone"] = standalone
    yield "", "", None, list(history or []) + [entry], ""


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
AGENT_LOG_REFRESH_SECONDS = 2.0


def show_agent_log(kinds: List[str]) -> Tuple[str, int]:
    return agent_log.LOG.render(kinds), agent_log.LOG.version


def poll_agent_log(kinds: List[str], live: bool, seen_version: int):
    """Timer callback: re-render only when auto-refresh is on and the log changed since the last render."""
    if not live or seen_version == agent_log.LOG.version:
        return gr.skip(), seen_version
    return show_agent_log(kinds)


def clear_agent_log(kinds: List[str]) -> Tuple[str, int]:
    agent_log.LOG.clear()
    return show_agent_log(kinds)


def create_ui() -> gr.Blocks:
    agent_log.install()
    install_llm_timeouts()
    install_last_gremlin_block()
    install_text_filter_rewrite()
    install_prompt_caching()  # innermost, so retries and the recorder go through the cached call
    install_split_retry()  # before the recorder, so the audit sees the combined reply
    install_extraction_recorder()
    install_parallel_extraction()
    cfg = default_config()
    apply_config(**cfg)

    with gr.Blocks(title="HugeGraph GraphRAG Showcase") as demo:
        gr.Markdown("# HugeGraph GraphRAG Showcase\nBuild a knowledge graph from documents, then ask it questions.")

        with gr.Accordion(f"Settings (LLM provider: {cfg['provider']} + HugeGraph)", open=False):
            with gr.Row():
                provider = gr.Dropdown(list(PROVIDERS), value=cfg["provider"], label="LLM provider")
                chat_model = gr.Textbox(cfg["chat_model"], label="Chat model (schema, answers)")
                extract_model = gr.Textbox(cfg["extract_model"], label="Extraction model (blank = chat model)")
                text2gql_model = gr.Textbox(cfg["text2gql_model"], label="Text2Gremlin model (blank = chat model)")
            with gr.Row():
                embedding_model = gr.Textbox(cfg["embedding_model"], label="Embedding model")
            with gr.Row():
                api_key = gr.Textbox(
                    "", label="API key (blank = from environment)", type="password"
                )
                api_base = gr.Textbox(cfg["api_base"], label="API base URL (optional, e.g. a gateway)")
                ollama_host = gr.Textbox(cfg["ollama_host"], label="Ollama host")
                ollama_port = gr.Number(cfg["ollama_port"], label="Ollama port", precision=0)
            with gr.Row():
                graph_url = gr.Textbox(cfg["graph_url"], label="HugeGraph URL")
                graph_space = gr.Textbox(cfg["graph_space"], label="Graph space", placeholder="server default")
                graph_name = gr.Textbox(cfg["graph_name"], label="Graph")
                graph_user = gr.Textbox(cfg["graph_user"], label="User")
                graph_pwd = gr.Textbox(cfg["graph_pwd"], label="Password", type="password")
            gr.Markdown(
                "Claude and OpenAI model names use LiteLLM prefixes (`anthropic/...`, `openai/...`). "
                "Claude has no embedding API, so it keeps embeddings on Ollama."
            )
            apply_btn = gr.Button("Apply settings", size="sm")
            config_status = gr.Markdown()
            def fill_provider_defaults(name: str) -> Tuple[str, str, str, str, str]:
                defaults = provider_defaults(name)
                return tuple(
                    defaults[key]
                    for key in ("chat_model", "extract_model", "text2gql_model", "embedding_model", "api_base")
                )

            provider.change(
                fill_provider_defaults,
                provider,
                [chat_model, extract_model, text2gql_model, embedding_model, api_base],
            )
            apply_btn.click(
                apply_config,
                [
                    provider,
                    chat_model,
                    embedding_model,
                    api_key,
                    api_base,
                    ollama_host,
                    ollama_port,
                    graph_url,
                    graph_name,
                    graph_user,
                    graph_pwd,
                    extract_model,
                    text2gql_model,
                    graph_space,
                ],
                config_status,
            )

        with gr.Tab("1. Build knowledge graph"):
            gr.Markdown(
                "### Step 1 - Infer a schema from sample documents\n"
                f"Upload a few typical documents (up to {MAX_SAMPLE_FILES} are used): a schema inferred from "
                "a single document tends to fit only that document, e.g. keying a label on an ID that the "
                "others lack. Review the inferred schema before creating the graph: small local models may copy "
                "labels from the built-in example (e.g. `webpage`, `roommate`). Edit or delete anything that does "
                "not fit your data, or click *Infer schema* again."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    sample_file = gr.File(
                        label="Sample documents (.txt / .pdf / .docx)",
                        file_count="multiple",
                        file_types=list(SUPPORTED_SUFFIXES),
                    )
                    infer_btn = gr.Button("Infer schema", variant="primary")
                    schema_summary = gr.Markdown()
                with gr.Column(scale=2):
                    schema_box = gr.Code(label="Graph schema (editable)", language="json", lines=18)
            questions_box = gr.Textbox(
                label="Questions this graph should answer (optional)",
                info="One per line, e.g. 'Which judge sentenced each defendant, and to how many months?'. Schema "
                "inference keeps the entities and relationships these questions need, and extraction keeps the facts "
                "they need; the design still comes from the documents.",
                lines=4,
            )
            create_btn = gr.Button("Create graph with this schema", variant="primary")
            create_status = gr.Markdown()

            with gr.Group(visible=False) as ingest_group:
                gr.Markdown("### Step 2 - Ingest documents with the same format")
                docs = gr.File(label="Documents", file_count="multiple", file_types=list(SUPPORTED_SUFFIXES))
                ingest_btn = gr.Button("Extract & import", variant="primary")
                ingest_report = gr.Markdown()

            with gr.Accordion("Reset", open=False):
                clear_btn = gr.Button("Delete all graph data and schema", variant="stop", size="sm")
                clear_status = gr.Markdown()

            def infer_from_ui(files, questions):
                return infer_schema(files, questions=questions)

            infer_btn.click(infer_from_ui, [sample_file, questions_box], [schema_box, schema_summary])
            create_btn.click(create_graph_schema, schema_box, create_status).success(
                lambda: gr.update(visible=True), None, ingest_group
            )
            def ingest_from_ui(files, schema, questions):
                yield from ingest_files(files, schema, question_hints(questions))

            ingest_btn.click(ingest_from_ui, [docs, schema_box, questions_box], ingest_report)
            clear_btn.click(clear_graph, None, clear_status).success(
                lambda: gr.update(visible=False), None, ingest_group
            )

        with gr.Tab("2. Explore & query"):
            with gr.Row():
                sample_size = gr.Slider(10, 500, value=100, step=10, label="Vertices to visualize")
                refresh_btn = gr.Button("Refresh overview", variant="primary")
            totals = gr.Markdown()
            with gr.Row():
                vertex_table = gr.Dataframe(headers=["Vertex label", "Count"], interactive=False)
                edge_table = gr.Dataframe(headers=["Edge label", "Count"], interactive=False)
            graph_view = gr.HTML()
            refresh_btn.click(load_overview, sample_size, [totals, vertex_table, edge_table, graph_view])

            gr.Markdown("### Ask the graph")
            qa_history = gr.State([])

            # Answered questions stay on the page as cards, oldest first, above the box for the next question.
            @gr.render(inputs=qa_history)
            def show_qa_history(history):
                for number, item in enumerate(history, start=1):
                    with gr.Group():
                        gr.Markdown(f"**Q{number}. {item['question']}**")
                        gr.Markdown(item["answer"])
                        with gr.Accordion("Generated Gremlin & raw results", open=False):
                            gr.Code(item["gremlin"], label="Gremlin", language="python")
                            gr.JSON(item["raw"], label="Raw results")

            question = gr.Textbox(
                label="Question",
                placeholder="e.g. Who works at the same company as Alice? Follow-ups can refer back: 'Where is it based?'",
            )
            with gr.Row():
                ask_btn = gr.Button("Ask", variant="primary")
                use_context = gr.Checkbox(True, label=f"Follow-ups use the last {CONTEXT_TURNS} answers")
                clear_qa_btn = gr.Button("Clear questions", size="sm")
            answer = gr.Markdown()
            with gr.Accordion("Generated Gremlin & raw results", open=False):
                gremlin_box = gr.Code(label="Gremlin", language="python")
                raw_box = gr.JSON(label="Raw results")
            ask_outputs = [answer, gremlin_box, raw_box, qa_history, question]
            ask_btn.click(ask_with_history, [question, qa_history, use_context], ask_outputs)
            question.submit(ask_with_history, [question, qa_history, use_context], ask_outputs)
            clear_qa_btn.click(lambda: [], None, qa_history)

        with gr.Accordion("Agent log - background LLM, embedding and HugeGraph calls", open=False):
            gr.Markdown(
                f"Newest first. Shows the last 50 calls and keeps the last {agent_log.MAX_ENTRIES} in memory "
                f"(cleared on restart); requests and responses longer than {agent_log.MAX_TEXT} characters keep their start and end."
            )
            with gr.Row():
                log_kinds = gr.CheckboxGroup(list(agent_log.KINDS), value=list(agent_log.KINDS), label="Show")
                log_live = gr.Checkbox(True, label=f"Auto-refresh every {AGENT_LOG_REFRESH_SECONDS:g} s")
                log_refresh = gr.Button("Refresh", size="sm")
                log_clear = gr.Button("Clear", size="sm")
            log_box = gr.Code(agent_log.LOG.render(), language=None, lines=25, max_lines=40, interactive=False)
            log_version = gr.State(agent_log.LOG.version)
            for trigger in (log_refresh.click, log_kinds.change):
                trigger(show_agent_log, log_kinds, [log_box, log_version])
            log_clear.click(clear_agent_log, log_kinds, [log_box, log_version])
            gr.Timer(AGENT_LOG_REFRESH_SECONDS).tick(
                poll_agent_log, [log_kinds, log_live, log_version], [log_box, log_version]
            )

    return demo


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="HugeGraph GraphRAG showcase (Ollama by default, Claude or OpenAI optional)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    create_ui().queue().launch(server_name=args.host, server_port=args.port, theme=gr.themes.Soft())
