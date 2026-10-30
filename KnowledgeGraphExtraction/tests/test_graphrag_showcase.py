import json
from collections import Counter
from types import SimpleNamespace

import gradio as gr
import pytest

from hugegraph_llm.config import huge_settings, llm_settings
import graphrag_showcase as app
from hugegraph_llm.flows import FlowName
from hugegraph_llm.models.embeddings.init_embedding import get_embedding
from hugegraph_llm.models.llms.init_llm import get_chat_llm, get_extract_llm, get_text2gql_llm
from hugegraph_llm.models.embeddings.ollama import OllamaEmbedding
from hugegraph_llm.models.llms.litellm import LiteLLMClient
from hugegraph_llm.models.llms.ollama import OllamaClient


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate(self, prompt=None, messages=None, **kwargs):
        self.calls.append({"prompt": prompt, "messages": messages, "kwargs": kwargs})
        if not self.responses:
            raise AssertionError("FakeLLM has no remaining responses")
        return self.responses.pop(0)


SCHEMA = {
    "vertexlabels": [
        {"id": 1, "name": "person", "primary_keys": ["name"], "properties": ["name"], "nullable_keys": []},
        {"id": 2, "name": "company", "primary_keys": ["name"], "properties": ["name"], "nullable_keys": []},
    ],
    "edgelabels": [{"name": "works_at", "source_label": "person", "target_label": "company", "properties": []}],
    "propertykeys": [{"name": "name", "data_type": "TEXT", "cardinality": "SINGLE"}],
}
SCHEMA_TEXT = json.dumps(SCHEMA)


class FakeScheduler:
    def __init__(self, responses):
        self.responses = dict(responses)
        self.calls = []

    def schedule_flow(self, flow_name, *args, **kwargs):
        self.calls.append((flow_name, args, kwargs))
        response = self.responses[flow_name]
        if callable(response):
            return response(*args, **kwargs)
        if isinstance(response, Exception):
            raise response
        return response


ENSURE_GRAPH_EXISTS = app.ensure_graph_exists


@pytest.fixture(autouse=True)
def isolate_connection(monkeypatch, tmp_path):
    """No test reads the real connection.properties or contacts a HugeGraph server to create a graph."""
    monkeypatch.setattr(app, "CONNECTION_PROPERTIES", tmp_path / "connection.properties")
    monkeypatch.setattr(app, "ensure_graph_exists", lambda: "")
    monkeypatch.setattr(app, "needs_text_filter_rewrite", lambda: False)


@pytest.fixture(autouse=True)
def restore_settings():
    llm_snapshot, huge_snapshot = llm_settings.model_dump(), huge_settings.model_dump()
    yield
    for key, value in llm_snapshot.items():
        setattr(llm_settings, key, value)
    for key, value in huge_snapshot.items():
        setattr(huge_settings, key, value)


PROVIDER_ENV = (
    "LLM_PROVIDER",
    "OLLAMA_CHAT_MODEL",
    "CLAUDE_MODEL",
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "CLAUDE_EXTRACT_MODEL",
    "CLAUDE_TEXT2GQL_MODEL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "OPENAI_CHAT_MODEL",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    # hugegraph-llm copies its .env into os.environ on import, so these depend on the developer's .env
    "OLLAMA_EMBEDDING_MODEL",
    "OPENAI_EMBEDDING_MODEL",
)


@pytest.fixture
def clean_env(monkeypatch):
    for name in PROVIDER_ENV:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_default_config_uses_ollama_and_qwen_unless_overridden(clean_env):
    cfg = app.default_config()
    assert (cfg["provider"], cfg["chat_model"]) == ("ollama", "qwen3-coder")

    clean_env.setenv("OLLAMA_CHAT_MODEL", "llama3.1")
    assert app.default_config()["chat_model"] == "llama3.1"


def test_default_config_rejects_unknown_provider(clean_env):
    clean_env.setenv("LLM_PROVIDER", "gemini")
    with pytest.raises(ValueError, match="LLM_PROVIDER"):
        app.default_config()


def test_claude_defaults_follow_gateway_variables(clean_env):
    assert app.provider_defaults("claude")["chat_model"] == "anthropic/claude-sonnet-5-5"

    clean_env.setenv("ANTHROPIC_DEFAULT_SONNET_MODEL", "claude-sonnet")
    clean_env.setenv("ANTHROPIC_AUTH_TOKEN", "tok")
    clean_env.setenv("ANTHROPIC_BASE_URL", "https://gw.example")
    defaults = app.provider_defaults("claude")
    assert defaults == {
        "chat_model": "anthropic/claude-sonnet",
        "extract_model": "",
        "text2gql_model": "",
        "embedding_model": "nomic-embed-text",
        "api_key": "tok",
        "api_base": "https://gw.example",
    }


def test_blank_environment_variable_falls_back_to_the_default(clean_env):
    # A freshly generated hugegraph-llm .env has OLLAMA_EMBEDDING_MODEL= blank, and hugegraph-llm copies it into
    # os.environ on import.
    clean_env.setenv("OLLAMA_EMBEDDING_MODEL", "")
    clean_env.setenv("OPENAI_EMBEDDING_MODEL", "  ")
    assert app.provider_defaults("claude")["embedding_model"] == "nomic-embed-text"
    assert app.provider_defaults("openai")["embedding_model"] == "openai/text-embedding-3-small"
    assert app.default_config()["embedding_model"] == "nomic-embed-text"


def test_apply_config_routes_every_llm_role_to_ollama():
    app.apply_config("ollama", "qwen2.5", "nomic-embed-text", "", "", "10.0.0.5", "11500", "hg:8080", "kg", "u", "p")

    for factory in (get_chat_llm, get_extract_llm, get_text2gql_llm):
        client = factory(llm_settings)
        assert isinstance(client, OllamaClient)
        assert client.model == "qwen2.5"
    embedding = get_embedding(llm_settings)
    assert embedding.model == "nomic-embed-text"
    assert (llm_settings.ollama_embedding_host, llm_settings.ollama_embedding_port) == ("10.0.0.5", 11500)
    assert (huge_settings.graph_url, huge_settings.graph_name, huge_settings.graph_user) == ("hg:8080", "kg", "u")


def test_apply_config_routes_claude_through_litellm_and_keeps_ollama_embeddings(clean_env):
    status = app.apply_config(
        "claude", "anthropic/claude-x", "nomic-embed-text", "key", "https://gw", "127.0.0.1", 11434, "hg", "g", "u", "p"
    )

    for factory in (get_chat_llm, get_extract_llm, get_text2gql_llm):
        client = factory(llm_settings)
        assert isinstance(client, LiteLLMClient)
        assert (client.model, client.api_key, client.api_base) == ("anthropic/claude-x", "key", "https://gw")
    assert get_extract_llm(llm_settings).max_tokens >= app.HOSTED_EXTRACT_TOKENS
    assert isinstance(get_embedding(llm_settings), OllamaEmbedding)
    assert "Warning" not in status


def test_claude_defaults_every_role_to_the_chat_model_unless_overridden(clean_env):
    defaults = app.provider_defaults("claude")
    assert (defaults["extract_model"], defaults["text2gql_model"]) == ("", "")

    clean_env.setenv("CLAUDE_EXTRACT_MODEL", "anthropic/claude-x")
    clean_env.setenv("CLAUDE_TEXT2GQL_MODEL", "anthropic/claude-y")
    defaults = app.provider_defaults("claude")
    assert (defaults["extract_model"], defaults["text2gql_model"]) == ("anthropic/claude-x", "anthropic/claude-y")


def test_apply_config_gives_each_role_its_own_model(clean_env):
    status = app.apply_config(
        "claude", "anthropic/sonnet", "nomic-embed-text", "key", "", "127.0.0.1", 11434, "hg", "g", "u", "p",
        extract_model="anthropic/haiku",
    )

    assert get_chat_llm(llm_settings).model == "anthropic/sonnet"
    assert get_extract_llm(llm_settings).model == "anthropic/haiku"
    assert get_text2gql_llm(llm_settings).model == "anthropic/sonnet"  # blank falls back to the chat model
    assert "extract `anthropic/haiku`" in status


def test_apply_config_routes_openai_chat_and_embeddings_through_litellm(clean_env):
    clean_env.setenv("OPENAI_API_KEY", "sk-env")
    status = app.apply_config(
        "openai", "openai/gpt-x", "openai/embed-x", "", "", "127.0.0.1", 11434, "hg", "g", "u", "p"
    )

    client = get_text2gql_llm(llm_settings)
    assert isinstance(client, LiteLLMClient)
    assert (client.model, client.api_key, client.api_base) == ("openai/gpt-x", "sk-env", None)
    # get_embedding() cannot build LiteLLM embeddings (missing dimension); flows use Embeddings() instead.
    assert llm_settings.embedding_type == "litellm"
    assert (llm_settings.litellm_embedding_model, llm_settings.litellm_embedding_api_key) == ("openai/embed-x", "sk-env")
    assert "Warning" not in status


def test_apply_config_warns_when_hosted_provider_has_no_key(clean_env):
    status = app.apply_config("openai", "openai/gpt-x", "openai/e", "", "", "127.0.0.1", 11434, "hg", "g", "u", "p")
    assert "no API key" in status


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("not json", "not valid JSON"),
        ("[]", "JSON object"),
        ('{"vertices": [], "edges": []}', "does not contain 'vertexlabels'"),
        ('{"vertexlabels": [], "edgelabels": []}', "no vertex labels"),
        (
            '{"vertexlabels": [{"name": "person", "properties": ["name"]}],'
            ' "edgelabels": [{"name": "knows", "source_label": "person", "target_label": "pet"}]}',
            "unknown vertex label",
        ),
    ],
)
def test_parse_schema_rejects_invalid_schema(text, message):
    with pytest.raises(ValueError, match=message):
        app.parse_schema(text)


def test_parse_schema_fills_keys_required_to_create_the_graph():
    # Shape of the bundled few-shot example: no propertykeys, nullable_keys or edge properties.
    schema = app.parse_schema(
        json.dumps(
            {
                "vertexlabels": [{"name": "person", "primary_keys": ["name"], "properties": ["name", "age"]}],
                "edgelabels": [{"name": "knows", "source_label": "person", "target_label": "person"}],
            }
        )
    )

    assert schema["vertexlabels"][0]["nullable_keys"] == ["age"]
    assert [p["name"] for p in schema["propertykeys"]] == ["name", "age"]
    assert schema["edgelabels"][0]["properties"] == []


def test_infer_schema_sends_truncated_sample_and_returns_formatted_schema(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "sample_chars", lambda: 1000)
    sample = tmp_path / "sample.txt"
    sample.write_text("x" * 1500, encoding="utf-8")
    scheduler = FakeScheduler({FlowName.BUILD_SCHEMA: SCHEMA_TEXT})

    schema_text, summary = app.infer_schema(str(sample), scheduler=scheduler)

    flow, args, _ = scheduler.calls[0]
    assert flow == FlowName.BUILD_SCHEMA
    sent = args[0][0]
    assert sent.startswith("The data below is one sample of a collection of similar documents.")
    assert "dates the DATE type" in sent  # design rules apply to a single sample too
    assert sent.endswith("\n\n" + "x" * 1000)
    assert "vertexlabels" in json.loads(args[2])  # few-shot schema conveys the expected format
    assert json.loads(schema_text) == SCHEMA
    assert "`person` -[`works_at`]-> `company`" in summary


def test_infer_schema_sends_labelled_excerpts_of_every_sample_within_the_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "sample_chars", lambda: 3000)
    paths = []
    for name, letter in (("a.txt", "a"), ("b.txt", "b"), ("c.txt", "c")):
        (tmp_path / name).write_text(letter * 2000, encoding="utf-8")
        paths.append(str(tmp_path / name))
    scheduler = FakeScheduler({FlowName.BUILD_SCHEMA: SCHEMA_TEXT})

    app.infer_schema(paths, scheduler=scheduler)

    sent = scheduler.calls[0][1][0][0]
    assert sent.startswith("The data below is excerpts from 3 different documents of the same kind.")
    assert "appears in at least two of the documents" in sent
    assert "primary keys that every document provides" in sent
    for i, (name, letter) in enumerate((("a.txt", "a"), ("b.txt", "b"), ("c.txt", "c")), start=1):
        assert f"--- Document {i} of 3 ({name}) ---\n{letter * 1000}\n" in sent + "\n"
        assert letter * 1001 not in sent  # each file gets an equal share of the budget


def test_pick_samples_spreads_large_uploads_evenly():
    paths = [f"{i:02d}.txt" for i in range(20)]

    assert app.pick_samples(paths[:4], limit=6) == paths[:4]
    assert app.pick_samples(paths, limit=4) == ["00.txt", "06.txt", "13.txt", "19.txt"]
    assert app.pick_samples(paths, limit=1) == ["00.txt"]


def test_sample_budget_is_larger_for_hosted_models(monkeypatch):
    monkeypatch.setattr(app.llm_settings, "chat_llm_type", "ollama/local")
    assert app.sample_chars() == app.MAX_SAMPLE_CHARS
    monkeypatch.setattr(app.llm_settings, "chat_llm_type", "litellm")
    assert app.sample_chars() == app.HOSTED_SAMPLE_CHARS


def test_infer_schema_surfaces_unusable_llm_output(tmp_path):
    sample = tmp_path / "sample.txt"
    sample.write_text("Alice works at Acme.", encoding="utf-8")
    scheduler = FakeScheduler({FlowName.BUILD_SCHEMA: '{"vertexlabels": []}'})

    with pytest.raises(gr.Error, match="failed after 3 attempts: Input data does not contain"):
        app.infer_schema(str(sample), scheduler=scheduler)
    assert len(scheduler.calls) == app.SCHEMA_ATTEMPTS


def test_infer_schema_retries_until_llm_returns_valid_schema(tmp_path):
    sample = tmp_path / "sample.txt"
    sample.write_text("Alice works at Acme.", encoding="utf-8")
    outputs = iter(["not json", SCHEMA_TEXT])
    scheduler = FakeScheduler({FlowName.BUILD_SCHEMA: lambda *args: next(outputs)})

    schema_text, _ = app.infer_schema(str(sample), scheduler=scheduler)

    assert json.loads(schema_text) == SCHEMA
    assert len(scheduler.calls) == 2


def test_query_examples_take_one_question_per_line_without_list_markers():
    text = "1. Which judge sentenced Brendan Lawler?\n\n- Who is his spouse?\n  * Who prosecuted him?  \n"

    examples = json.loads(app.query_examples(text))

    assert examples == [
        {"description": "Which judge sentenced Brendan Lawler?", "gremlin": ""},
        {"description": "Who is his spouse?", "gremlin": ""},
        {"description": "Who prosecuted him?", "gremlin": ""},
    ]
    assert app.query_examples("  \n") is None


def test_question_hints_ask_extraction_for_the_facts_the_questions_need(tmp_path):
    doc = tmp_path / "a.txt"
    doc.write_text("Alice works at Acme.", encoding="utf-8")
    scheduler = FakeScheduler({FlowName.GRAPH_EXTRACT: json.dumps({"vertices": [], "edges": []})})
    hints = app.question_hints("1. Where does Alice work?\n\n- Who prosecuted Alice?")

    list(app.ingest_files([str(doc)], SCHEMA_TEXT, hints, scheduler=scheduler))

    extract_prompt_sent = scheduler.calls[0][1][2]
    assert "must answer the questions below. Extract every fact they need:\n" in extract_prompt_sent
    assert "Where does Alice work?\nWho prosecuted Alice?" in extract_prompt_sent
    assert app.question_hints(" \n") == ""


def test_infer_schema_sends_the_questions_as_query_examples(tmp_path):
    sample = tmp_path / "sample.txt"
    sample.write_text("Alice works at Acme.", encoding="utf-8")
    scheduler = FakeScheduler({FlowName.BUILD_SCHEMA: SCHEMA_TEXT})

    app.infer_schema(str(sample), scheduler=scheduler, questions="Where does Alice work?")
    app.infer_schema(str(sample), scheduler=scheduler)

    with_questions, without = (call[1][1] for call in scheduler.calls)
    assert json.loads(with_questions) == [{"description": "Where does Alice work?", "gremlin": ""}]
    assert without is None


class FakeSchemaApi:
    """Records pyhugegraph's fluent indexLabel(...).onV(...).by(...).<type>().ifNotExist().create() calls."""

    def __init__(self, existing=None):
        self.created = []
        self.existing = existing

    def getSchema(self):  # noqa: N802 - mirrors pyhugegraph
        return self.existing

    def indexLabel(self, name):  # noqa: N802 - mirrors pyhugegraph
        spec = {"name": name}
        api = self

        class Builder:
            def onV(self, label):  # noqa: N802
                spec["label"] = label
                return self

            def onE(self, label):  # noqa: N802
                spec["label"] = label
                spec["on"] = "edge"
                return self

            def by(self, prop):
                spec["prop"] = prop
                return self

            def secondary(self):
                spec["type"] = "secondary"
                return self

            def range(self):
                spec["type"] = "range"
                return self

            def ifNotExist(self):  # noqa: N802
                return self

            def create(self):
                api.created.append(spec)

        return Builder()


def test_create_index_labels_skips_sole_primary_key_and_ranges_numbers():
    schema = app.parse_schema(
        json.dumps(
            {
                "vertexlabels": [
                    {"name": "person", "primary_keys": ["name"], "properties": ["name", "age", "occupation"]},
                    {"name": "city", "primary_keys": ["name", "country"], "properties": ["name", "country"]},
                ],
                "edgelabels": [],
                "propertykeys": [{"name": "age", "data_type": "INT", "cardinality": "SINGLE"}],
            }
        )
    )
    schema_api = FakeSchemaApi()

    names = app.create_index_labels(schema_api, schema)

    assert names == ["personByAge", "personByOccupation", "cityByName", "cityByCountry"]
    assert schema_api.created[0] == {"name": "personByAge", "label": "person", "prop": "age", "type": "range"}
    assert schema_api.created[1]["type"] == "secondary"


def test_create_index_labels_indexes_edge_properties_for_g_e_filters():
    schema = app.parse_schema(
        json.dumps(
            {
                "vertexlabels": [
                    {"name": "person", "primary_keys": ["name"], "properties": ["name"]},
                    {"name": "bank", "primary_keys": ["name"], "properties": ["name"]},
                ],
                "edgelabels": [
                    {"name": "defrauded", "source_label": "person", "target_label": "bank", "properties": ["amount"]},
                    {"name": "employed_by", "source_label": "person", "target_label": "bank", "properties": ["position"]},
                    {"name": "knows", "source_label": "person", "target_label": "person", "properties": []},
                ],
                "propertykeys": [{"name": "amount", "data_type": "DOUBLE", "cardinality": "SINGLE"}],
            }
        )
    )
    schema_api = FakeSchemaApi()

    assert app.create_index_labels(schema_api, schema) == ["defraudedByAmount", "employed_byByPosition"]
    assert schema_api.created == [
        {"name": "defraudedByAmount", "label": "defrauded", "on": "edge", "prop": "amount", "type": "range"},
        {"name": "employed_byByPosition", "label": "employed_by", "on": "edge", "prop": "position", "type": "secondary"},
    ]


def test_create_graph_schema_commits_schema_and_indexes():
    committed = []

    class Committer:
        schema = FakeSchemaApi()

        def init_schema_if_need(self, schema):
            committed.append(schema)

    status = app.create_graph_schema(SCHEMA_TEXT, committer=Committer())

    assert committed == [SCHEMA]
    # SCHEMA has only sole-primary-key "name" properties, so nothing is indexable
    assert "2 vertex label(s), 1 edge label(s) and 0 index label(s)" in status


# Shaped like HugeGraph 1.8's GET schema response, which reports edge endpoints as `links`.
LIVE_SCHEMA = {
    "propertykeys": [{"name": "name", "data_type": "TEXT"}, {"name": "age", "data_type": "INT"}],
    "vertexlabels": [
        {"name": "person", "primary_keys": ["name"], "properties": ["name", "age"]},
        {"name": "company", "primary_keys": ["name"], "properties": ["name"]},
        {"name": "offense", "primary_keys": ["name"], "properties": ["name"]},
    ],
    "edgelabels": [
        {"name": "works_at", "links": [{"person": "company"}], "properties": []},
        {"name": "charged_with", "links": [{"person": "offense"}], "properties": []},
    ],
}


def _committer(existing):
    committed = []

    class Committer:
        schema = FakeSchemaApi(existing)

        def init_schema_if_need(self, schema):
            committed.append(schema)

    return Committer(), committed


def test_create_graph_schema_refuses_a_schema_that_conflicts_with_the_graph():
    clashing = json.dumps(
        {
            "vertexlabels": [
                {"name": "defendant", "primary_keys": ["name"], "properties": ["name"]},
                {"name": "charge", "primary_keys": ["name"], "properties": ["name"]},
                {"name": "person", "primary_keys": ["name"], "properties": ["name", "age", "residence"]},
            ],
            "edgelabels": [
                {"name": "charged_with", "source_label": "defendant", "target_label": "charge", "properties": []}
            ],
            "propertykeys": [{"name": "age", "data_type": "TEXT", "cardinality": "SINGLE"}],
        }
    )
    committer, committed = _committer(LIVE_SCHEMA)

    with pytest.raises(gr.Error) as err:
        app.create_graph_schema(clashing, committer=committer)

    message = str(err.value)
    assert "Delete all graph data and schema (Reset, below) first" in message
    assert "edge `charged_with` is person -> offense in the graph, defendant -> charge here" in message
    assert "vertex `person` in the graph has no propert(ies) residence" in message
    assert "property `age` is INT in the graph, TEXT in this schema" in message
    assert committed == []  # nothing is written


def test_create_graph_schema_adds_to_a_compatible_graph_and_notes_leftover_labels():
    committer, committed = _committer(LIVE_SCHEMA)

    status = app.create_graph_schema(SCHEMA_TEXT, committer=committer)

    assert committed == [SCHEMA]
    assert "still has 2 label(s) from an earlier schema (`charged_with`, `offense`)" in status


def test_create_graph_schema_on_an_empty_graph_has_no_note():
    committer, committed = _committer({"propertykeys": [], "vertexlabels": [], "edgelabels": []})

    status = app.create_graph_schema(SCHEMA_TEXT, committer=committer)

    assert committed == [SCHEMA] and "Note" not in status


def test_schema_conflicts_reads_older_source_target_fields_and_key_changes():
    existing = {
        "vertexlabels": [{"name": "person", "primary_keys": ["id"], "properties": ["id", "name"]}],
        "edgelabels": [
            {"name": "works_at", "source_label": "person", "target_label": "company", "properties": ["since"]}
        ],
    }

    assert app.schema_conflicts(SCHEMA, existing) == ["vertex `person` is keyed on ['id'] in the graph, ['name'] here"]


def test_create_graph_schema_does_not_commit_invalid_schema():
    class Committer:
        def init_schema_if_need(self, schema):
            raise AssertionError("must not commit")

    with pytest.raises(gr.Error, match="not valid JSON"):
        app.create_graph_schema("{oops", committer=Committer())


def test_ingest_files_imports_each_file_and_reports_failures(tmp_path):
    good, empty, bad = tmp_path / "a.txt", tmp_path / "b.txt", tmp_path / "c.csv"
    good.write_text("Alice works at Acme.", encoding="utf-8")
    empty.write_text("Unrelated text.", encoding="utf-8")
    bad.write_text("x", encoding="utf-8")
    extracted = {
        "vertices": [{"id": "1:Alice", "label": "person"}, {"id": "2:Acme", "label": "company"}],
        "edges": [{"label": "works_at", "outV": "1:Alice", "inV": "2:Acme"}],
    }

    def extract(schema, texts, *args, **kwargs):
        return json.dumps(extracted if "Alice" in texts[0] else {"vertices": [], "edges": []})

    scheduler = FakeScheduler({FlowName.GRAPH_EXTRACT: extract, FlowName.IMPORT_GRAPH_DATA: "{}"})

    reports = list(
        app.ingest_files(
            [str(good), str(empty), str(bad)],
            SCHEMA_TEXT,
            scheduler=scheduler,
            existing_ids=lambda ids: set(),
            existing_values=lambda label, key: [],
        )
    )

    imports = [c for c in scheduler.calls if c[0] == FlowName.IMPORT_GRAPH_DATA]
    assert len(imports) == 1
    assert imports[0][1] == (extracted, json.dumps(SCHEMA, ensure_ascii=False))
    # files are extracted concurrently, so the calls are not in upload order
    good_extract = next(c for c in scheduler.calls if c[0] == FlowName.GRAPH_EXTRACT and "Alice" in c[1][1][0])
    assert good_extract[2] == {"split_type": "document", "language": "en"}
    assert good_extract[1][1] == ["Alice works at Acme."]
    final = reports[-1]
    assert "`a.txt`: imported 2 vertices, 1 edges" in final
    assert "`b.txt`: nothing extracted" in final
    assert "`c.csv`: failed - Unsupported file type" in final
    assert (
        "2 vertices and 1 edges imported (0 case duplicate(s) merged, "
        "0 dangling edge(s) skipped); 2 file(s) failed"
        in final
    )


def test_ingest_files_warns_about_schema_labels_that_got_nothing(tmp_path):
    doc = tmp_path / "a.txt"
    doc.write_text("Alice and Bob.", encoding="utf-8")
    extracted = {"vertices": [{"id": "1:Alice", "label": "person"}, {"id": "1:Bob", "label": "person"}], "edges": []}
    scheduler = FakeScheduler(
        {FlowName.GRAPH_EXTRACT: json.dumps(extracted), FlowName.IMPORT_GRAPH_DATA: "{}"}
    )

    final = list(
        app.ingest_files(
            [str(doc)],
            SCHEMA_TEXT,
            scheduler=scheduler,
            existing_ids=lambda ids: set(),
            existing_values=lambda label, key: [],
        )
    )[-1]

    assert "**Warning:** 2 schema label(s) got nothing from these documents: vertex `company`, edge `works_at`." in final
    assert "vertex `person`" not in final


def test_empty_labels_lists_unused_vertex_and_edge_labels():
    assert app.empty_labels(SCHEMA, Counter({"person": 3, "company": 1, "works_at": 2})) == []
    assert app.empty_labels(SCHEMA, Counter({"person": 3})) == ["vertex `company`", "edge `works_at`"]


PEOPLE_SCHEMA = {
    "vertexlabels": [
        {"name": "person", "primary_keys": ["name"], "properties": ["name", "age"]},
        {"name": "organization", "primary_keys": ["name"], "properties": ["name"]},
        {"name": "offense", "primary_keys": ["name"], "properties": ["name"]},
    ],
    "edgelabels": [],
}




def test_chunk_size_holds_whole_documents_for_hosted_models(monkeypatch):
    monkeypatch.setattr(app.llm_settings, "chat_llm_type", "ollama/local")
    assert app.chunk_chars() == app.CHUNK_CHARS
    monkeypatch.setattr(app.llm_settings, "chat_llm_type", "litellm")
    assert app.chunk_chars() == app.HOSTED_CHUNK_CHARS


def test_ingest_files_skips_dangling_edges_instead_of_failing_the_file(tmp_path):
    doc = tmp_path / "a.txt"
    doc.write_text("Carol studied at Stanford like Alice Chen.", encoding="utf-8")
    extracted = {
        "vertices": [{"id": "1:Carol", "label": "person"}, {"id": "2:Stanford", "label": "company"}],
        "edges": [
            {"label": "works_at", "outV": "1:Carol", "inV": "2:Stanford"},
            {"label": "works_at", "outV": "1:Alice", "inV": "2:Stanford"},  # abbreviated, matches nothing
            {"label": "works_at", "outV": "1:Alice Chen", "inV": "2:Stanford"},  # ingested by an earlier file
        ],
    }
    scheduler = FakeScheduler({FlowName.GRAPH_EXTRACT: json.dumps(extracted), FlowName.IMPORT_GRAPH_DATA: "{}"})
    lookups = []

    def existing_ids(ids):
        lookups.append(ids)
        return {"1:Alice Chen"}

    final = list(
        app.ingest_files(
            [str(doc)],
            SCHEMA_TEXT,
            scheduler=scheduler,
            existing_ids=existing_ids,
            existing_values=lambda label, key: [],
        )
    )[-1]

    assert lookups == [["1:Alice", "1:Alice Chen"]]
    imported = scheduler.calls[1][1][0]
    assert [e["outV"] for e in imported["edges"]] == ["1:Carol", "1:Alice Chen"]
    assert "imported 2 vertices, 2 edges (skipped 1 edge(s) pointing to unknown vertices)" in final


def test_ingest_files_requires_valid_schema(tmp_path):
    doc = tmp_path / "a.txt"
    doc.write_text("text", encoding="utf-8")

    with pytest.raises(gr.Error, match="Schema"):
        list(app.ingest_files([str(doc)], "", scheduler=FakeScheduler({})))


def test_build_graph_view_adds_placeholder_nodes_for_unsampled_endpoints():
    vertices = [{"id": "1:Alice", "label": "person", "properties": {"name": "Alice"}}]
    edges = [{"label": "works_at", "outV": "1:Alice", "inV": "2:Acme", "outVLabel": "person", "inVLabel": "company"}]

    view = app.build_graph_view(vertices, edges)

    nodes = {n["id"]: n for n in view["nodes"]}
    assert nodes["1:Alice"]["label"] == "Alice"
    assert nodes["2:Acme"] == {"id": "2:Acme", "label": "Acme", "group": "company", "title": "2:Acme"}
    assert view["edges"] == [{"from": "1:Alice", "to": "2:Acme", "label": "works_at", "arrows": "to"}]


def test_render_graph_html_escapes_data_into_iframe():
    view = app.build_graph_view([{"id": "1:x", "label": "person", "properties": {"name": '"><script>'}}], [])

    rendered = app.render_graph_html(view)

    assert rendered.startswith("<iframe")
    assert "<script>" not in rendered.split('srcdoc="', 1)[1]
    assert "The graph is empty" in app.render_graph_html({"nodes": [], "edges": []})


def _exec(data):
    return json.dumps({"data": data})


@pytest.mark.parametrize(
    ("res", "expected"),
    [
        (
            {
                "template_gremlin": "t",
                "template_execution_result": "Gremlin error",
                "raw_gremlin": "r",
                "raw_execution_result": _exec([1]),
            },
            ("r", [1]),
        ),
        (
            {
                "template_gremlin": "t",
                "template_execution_result": _exec([]),
                "raw_gremlin": "r",
                "raw_execution_result": _exec([2]),
            },
            ("r", [2]),
        ),
        (
            {
                "template_gremlin": "t",
                "template_execution_result": _exec([]),
                "raw_gremlin": "r",
                "raw_execution_result": "boom",
            },
            ("t", []),
        ),
        (
            {
                "template_gremlin": "t",
                "template_execution_result": "boom",
                "raw_gremlin": "r",
                "raw_execution_result": "boom",
            },
            ("t", None),
        ),
    ],
)
def test_pick_execution_prefers_successful_non_empty_result(res, expected):
    assert app.pick_execution(res) == expected


def test_answer_question_summarizes_executed_results():
    data = [{"id": "1:Alice", "label": "person"}]
    scheduler = FakeScheduler(
        {
            FlowName.TEXT2GREMLIN: {
                "template_gremlin": "g.V().hasLabel('person')",
                "template_execution_result": _exec(data),
            },
        }
    )
    llm = FakeLLM(["Alice is a person."])

    summary, gremlin, raw = app.answer_question("Who are the people?", scheduler=scheduler, llm=llm, schema=SCHEMA)

    assert (summary, gremlin, raw) == ("Alice is a person.", "g.V().hasLabel('person')", data)
    _, args, _ = next(c for c in scheduler.calls if c[0] == FlowName.TEXT2GREMLIN)
    assert args[0] == "Who are the people?"
    assert args[2] == huge_settings.graph_name
    assert args[1] == 0  # no few-shot examples are retrieved
    assert "Always scope vertex lookups with a label" in args[3]
    assert "template_execution_result" in args[4]
    assert all(c[0] != FlowName.BUILD_EXAMPLES_INDEX for c in scheduler.calls)
    assert "Who are the people?" in llm.calls[0]["prompt"]
    assert "1:Alice" in llm.calls[0]["prompt"]


@pytest.mark.parametrize(
    ("execution", "message", "raw"),
    [("Gremlin syntax error", "could not be executed", None), (_exec([]), "returned no results", [])],
)
def test_answer_question_skips_summary_without_results(execution, message, raw):
    scheduler = FakeScheduler(
        {FlowName.TEXT2GREMLIN: {"template_gremlin": "g.V()", "template_execution_result": execution}}
    )
    llm = FakeLLM([])

    summary, _, result = app.answer_question("q", scheduler=scheduler, llm=llm, schema={})

    assert message in summary
    assert result == raw
    assert llm.calls == []


def test_build_summary_prompt_truncates_large_results():
    built = app.build_summary_prompt("q", "g.V()", ["x" * (app.MAX_RESULT_CHARS * 2)])

    assert "...(truncated)" in built
    assert len(built) < app.MAX_RESULT_CHARS + len(app.SUMMARY_PROMPT) + 100


def test_build_summary_prompt_sends_a_sample_of_rows_and_asks_for_a_short_answer():
    rows = [{"name": f"customer {i}"} for i in range(36)]
    built = app.build_summary_prompt("Which customers?", "g.V()", rows)

    assert f"({app.SUMMARY_MAX_ROWS} of 36 rows" in built and "(36 in total)" in built
    assert "customer 19" in built and "customer 20" not in built
    assert f"at most {app.SUMMARY_LIST_ITEMS}" in built and "...and N more" in built
    assert "(1 of 1 rows" in app.build_summary_prompt("q", "g.V().count()", {"count": 3})








def _person(vid, name):
    return {"id": vid, "label": "person", "properties": {"name": name}}


def test_merge_case_duplicates_prefers_graph_spelling_then_first_in_batch():
    schema = app.parse_schema(SCHEMA_TEXT)
    extracted = {
        "vertices": [
            _person("1:alice chen", "alice chen"),  # graph already has "Alice Chen"
            _person("1:Bob", "Bob"),
            _person("1:BOB", "BOB"),  # first spelling in the batch wins
            {"id": "2:acme", "label": "company", "properties": {"name": "acme"}},
        ],
        "edges": [],
    }
    lookups = []

    def existing_values(label, key):
        lookups.append((label, key))
        return {"person": ["Alice Chen"], "company": []}[label]

    merged = app.merge_case_duplicates(extracted, schema, existing_values)

    assert merged == 2
    assert [v["properties"]["name"] for v in extracted["vertices"]] == ["Alice Chen", "Bob", "Bob", "acme"]
    assert lookups == [("person", "name"), ("company", "name")]  # one graph lookup per label


def test_merge_case_duplicates_skips_multi_key_labels_and_non_string_keys():
    schema = app.parse_schema(
        json.dumps(
            {
                "vertexlabels": [
                    {"name": "city", "primary_keys": ["name", "country"], "properties": ["name", "country"]},
                    {"name": "room", "primary_keys": ["number"], "properties": ["number"]},
                ],
                "edgelabels": [],
            }
        )
    )
    extracted = {
        "vertices": [
            {"id": "c1", "label": "city", "properties": {"name": "paris", "country": "FR"}},
            {"id": "r1", "label": "room", "properties": {"number": 7}},
        ]
    }

    def existing_values(label, key):
        raise AssertionError("no lookup expected")

    assert app.merge_case_duplicates(extracted, schema, existing_values) == 0
    assert extracted["vertices"][0]["properties"]["name"] == "paris"


def test_ingest_files_merges_case_duplicates_before_import(tmp_path):
    doc = tmp_path / "a.txt"
    doc.write_text("alice chen works at Acme.", encoding="utf-8")
    extracted = {
        "vertices": [_person("1:alice chen", "alice chen"), {"id": "2:Acme", "label": "company", "properties": {}}],
        "edges": [{"label": "works_at", "outV": "1:alice chen", "inV": "2:Acme"}],
    }
    scheduler = FakeScheduler({FlowName.GRAPH_EXTRACT: json.dumps(extracted), FlowName.IMPORT_GRAPH_DATA: "{}"})

    final = list(
        app.ingest_files(
            [str(doc)],
            SCHEMA_TEXT,
            scheduler=scheduler,
            existing_ids=lambda ids: set(),
            existing_values=lambda label, key: ["Alice Chen"] if label == "person" else [],
        )
    )[-1]

    imported = scheduler.calls[1][1][0]
    assert imported["vertices"][0]["properties"]["name"] == "Alice Chen"
    assert imported["edges"][0]["outV"] == "1:alice chen"  # resolved by the importer's id mapping
    assert "(merged 1 name(s) differing only in case)" in final


def test_drop_null_properties_strips_nulls_from_vertices_and_edges():
    extracted = {
        "vertices": [{"id": "1:A", "label": "bank", "properties": {"bic": "A", "country": None}}],
        "edges": [
            {"label": "sent", "outV": "1:A", "inV": "1:B", "properties": {"timestamp": None, "amount": 0}},
            {"label": "sent", "outV": "1:A", "inV": "1:B"},
        ],
    }

    assert app.drop_null_properties(extracted) == 2
    assert extracted["vertices"][0]["properties"] == {"bic": "A"}
    assert extracted["edges"][0]["properties"] == {"amount": 0}  # falsy but valid values are kept


def test_ingest_files_does_not_send_null_edge_properties(tmp_path):
    doc = tmp_path / "a.txt"
    doc.write_text("Alice works at Acme.", encoding="utf-8")
    extracted = {
        "vertices": [_person("1:Alice", "Alice"), {"id": "2:Acme", "label": "company", "properties": {"name": "Acme"}}],
        "edges": [{"label": "works_at", "outV": "1:Alice", "inV": "2:Acme", "properties": {"since": None}}],
    }
    scheduler = FakeScheduler({FlowName.GRAPH_EXTRACT: json.dumps(extracted), FlowName.IMPORT_GRAPH_DATA: "{}"})

    list(
        app.ingest_files(
            [str(doc)],
            SCHEMA_TEXT,
            scheduler=scheduler,
            existing_ids=lambda ids: set(),
            existing_values=lambda label, key: [],
        )
    )

    imported = scheduler.calls[1][1][0]
    assert imported["edges"][0]["properties"] == {}


def test_clear_graph_reports_that_schema_is_removed(monkeypatch):
    calls = []
    monkeypatch.setattr(app, "clean_hg_data", lambda: calls.append("clear"))
    monkeypatch.setattr(app, "remove_leftover_schema", lambda: calls.append("schema"))

    message = app.clear_graph()

    assert calls == ["clear", "schema"]
    assert "**and schema**" in message
    assert "Create graph with this schema" in message


class LeftoverSchema:
    """Schema manager whose removals land a few polls later, like server tasks."""

    def __init__(self, schema, delay=2):
        self.schema, self.delay, self.removed, self.pending = schema, delay, [], []

    def getSchema(self):
        for item in list(self.pending):
            item[2] -= 1
            if item[2] <= 0:
                self.pending.remove(item)
                self.schema[item[0]] = [x for x in self.schema[item[0]] if x["name"] != item[1]]
        return {kind: list(items) for kind, items in self.schema.items()}

    def _remover(self, kind):
        def label(name):
            def remove():
                self.removed.append((kind, name))
                self.pending.append([kind, name, self.delay])
            return SimpleNamespace(remove=remove)
        return label

    def __getattr__(self, attr):
        kinds = {"edgeLabel": "edgelabels", "vertexLabel": "vertexlabels", "propertyKey": "propertykeys"}
        if attr in kinds:
            return self._remover(kinds[attr])
        raise AttributeError(attr)


def test_remove_leftover_schema_removes_edges_then_vertices_then_keys(monkeypatch):
    schema = LeftoverSchema({
        "propertykeys": [{"name": "name"}, {"name": "age"}],
        "vertexlabels": [{"name": "person"}, {"name": "judge"}],
        "edgelabels": [{"name": "presided_over"}],
        "indexlabels": [],
    })
    monkeypatch.setattr(app, "get_hg_client", lambda: SimpleNamespace(schema=lambda: schema))
    sleeps = []

    app.remove_leftover_schema(sleep=sleeps.append)

    assert schema.removed == [("edgelabels", "presided_over"), ("vertexlabels", "person"),
                              ("vertexlabels", "judge"), ("propertykeys", "name"), ("propertykeys", "age")]
    assert schema.getSchema() == {"propertykeys": [], "vertexlabels": [], "edgelabels": [], "indexlabels": []}
    assert sleeps


def test_remove_leftover_schema_does_nothing_when_the_clear_removed_it(monkeypatch):
    schema = LeftoverSchema({"propertykeys": [], "vertexlabels": [], "edgelabels": [], "indexlabels": []})
    monkeypatch.setattr(app, "get_hg_client", lambda: SimpleNamespace(schema=lambda: schema))
    sleeps = []

    app.remove_leftover_schema(sleep=sleeps.append)

    assert schema.removed == [] and sleeps == []


def test_remove_leftover_schema_gives_up_when_labels_stay(monkeypatch):
    schema = LeftoverSchema({"propertykeys": [], "vertexlabels": [], "edgelabels": [{"name": "x"}],
                             "indexlabels": []}, delay=10**9)
    monkeypatch.setattr(app, "get_hg_client", lambda: SimpleNamespace(schema=lambda: schema))
    monkeypatch.setattr(app, "SCHEMA_REMOVE_WAIT_SECS", 3)

    with pytest.raises(RuntimeError, match="still has edgelabels"):
        app.remove_leftover_schema(sleep=lambda _: None)


def test_pack_chunks_never_cuts_a_block_and_packs_up_to_limit():
    msg = "Title: Case 1\nDate: 2026-01-01\nOffice: USAO\nSource: x"  # one multi-line record, no blank lines inside
    text = "\n\n".join([msg] * 5)

    chunks = app.pack_chunks(text, limit=len(msg) * 2 + 2)

    assert chunks == [f"{msg}\n\n{msg}", f"{msg}\n\n{msg}", msg]


def test_pack_chunks_keeps_oversized_record_whole_but_splits_huge_prose():
    record = "x" * 150
    assert app.pack_chunks(record, limit=100) == [record]  # up to 3x the limit stays intact

    prose = "\n".join(["line"] * 200)  # ~1000 chars without blank lines
    chunks = app.pack_chunks(prose, limit=100)
    assert all(len(c) <= 100 for c in chunks)
    assert "".join(chunks).count("line") == 200


def test_extraction_prompt_appends_hints_only_when_given():
    plain = app.extraction_prompt("")
    assert plain == app.extraction_prompt("  ")
    assert plain.startswith(app.prompt.extract_graph_prompt)
    assert "## Domain Hints" not in plain

    built = app.extraction_prompt("a/k/a names are the same person")

    assert built.startswith(plain)
    assert built.rstrip().endswith("a/k/a names are the same person")


def test_hints_reach_schema_inference_and_extraction(tmp_path):
    doc = tmp_path / "a.txt"
    doc.write_text("Edward Low, a/k/a Eddie Low", encoding="utf-8")
    scheduler = FakeScheduler(
        {
            FlowName.BUILD_SCHEMA: SCHEMA_TEXT,
            FlowName.GRAPH_EXTRACT: json.dumps({"vertices": [], "edges": []}),
        }
    )

    app.infer_schema(str(doc), hints="a/k/a names are the same person", scheduler=scheduler)
    list(app.ingest_files([str(doc)], SCHEMA_TEXT, hints="a/k/a names are the same person", scheduler=scheduler))

    schema_text_sent = scheduler.calls[0][1][0][0]
    assert schema_text_sent.startswith("Domain notes (not data):\na/k/a names are the same person")
    assert schema_text_sent.endswith("Edward Low, a/k/a Eddie Low")
    extract_prompt_sent = scheduler.calls[1][1][2]
    assert "## Domain Hints" in extract_prompt_sent


def test_extraction_prompt_always_requires_edge_properties():
    for hints in ("", "a/k/a names are the same person"):
        assert '"properties": {}' in app.extraction_prompt(hints)


def raw_graph(vertices, edges):
    return "```json\n" + json.dumps({"vertices": vertices, "edges": edges}) + "\n```"


ALICE = {"id": "1:Alice", "label": "person", "properties": {"name": "Alice"}}
ACME = {"id": "2:Acme", "label": "company", "properties": {"name": "Acme"}}


def test_audit_extraction_counts_everything_the_schema_drops():
    raw = [
        raw_graph(
            [
                {**ALICE, "properties": {"name": "Alice", "email": "a@x"}},
                ACME,
                {"id": "9:acct", "label": "account", "properties": {"number": "1"}},
            ],
            [
                {"label": "works_at", "outV": "1:Alice", "outVLabel": "person", "inV": "2:Acme", "inVLabel": "company", "properties": {}},
                {"label": "works_at", "outV": "2:Acme", "outVLabel": "company", "inV": "1:Alice", "inVLabel": "person", "properties": {}},
                {"label": "works_at", "outV": "1:Alice", "outVLabel": "person", "inV": "2:Ghost", "inVLabel": "company", "properties": {}},
                {"label": "holds", "outV": "1:Alice", "outVLabel": "person", "inV": "9:acct", "inVLabel": "account", "properties": {}},
                {"label": "works_at", "outV": "1:Alice", "outVLabel": "person", "inV": "2:Acme", "inVLabel": "company"},
            ],
        ),
        "Sorry, I cannot help with that.",
    ]

    drops = app.audit_extraction(raw, SCHEMA)

    assert drops["labels"] == {"vertex 'account'": 1, "edge 'holds'": 1}
    assert drops["properties"] == {"person.email": 1}
    assert drops["no_properties"] == {"works_at": 1}
    assert drops["endpoints"] == {"works_at company->person": 1}
    assert (drops["unresolved"], drops["unparsed"], drops["malformed"]) == (1, 1, 0)
    assert app.dropped_item_count(drops) == 5
    note = app.format_drops(drops)
    assert "labels not in schema: vertex 'account' x1, edge 'holds' x1" in note
    assert "properties not in schema: person.email x1" in note
    assert "edges without a properties key: works_at x1" in note
    assert "1 chunk(s) without valid graph JSON" in note


def test_audit_extraction_is_silent_when_everything_fits():
    edge = {"label": "works_at", "outV": "1:Alice", "outVLabel": "person", "inV": "2:Acme", "inVLabel": "company", "properties": {}}
    drops = app.audit_extraction([raw_graph([ALICE, ACME], [edge])], SCHEMA)
    assert app.dropped_item_count(drops) == 0
    assert app.format_drops(drops) == ""


def test_ingest_files_reports_what_the_schema_dropped_per_file(tmp_path):
    fits, off = tmp_path / "fits.txt", tmp_path / "off.txt"
    fits.write_text("Alice works at Acme.", encoding="utf-8")
    off.write_text("Account 1 belongs to a patient.", encoding="utf-8")

    def extract(schema, texts, *args, **kwargs):
        # stands in for hugegraph-llm: record the raw LLM reply, return what survives its filtering
        if "Alice" in texts[0]:
            app.record_extraction(
                raw_graph([{**ALICE, "properties": {"name": "Alice", "email": "a@x"}}, ACME], []), texts[0]
            )
            return json.dumps({"vertices": [ALICE, ACME], "edges": []})
        app.record_extraction(raw_graph([{"id": "9:p", "label": "patient", "properties": {"name": "P"}}], []), texts[0])
        return json.dumps({"vertices": [], "edges": []})

    scheduler = FakeScheduler({FlowName.GRAPH_EXTRACT: extract, FlowName.IMPORT_GRAPH_DATA: "{}"})
    final = list(
        app.ingest_files(
            [str(fits), str(off)],
            SCHEMA_TEXT,
            scheduler=scheduler,
            existing_ids=lambda ids: set(),
            existing_values=lambda label, key: [],
        )
    )[-1]

    assert "`fits.txt`: imported 2 vertices, 0 edges\n  - dropped by the schema: properties not in schema: person.email x1" in final
    assert "`off.txt`: nothing extracted (does the document match the schema?)\n  - dropped by the schema: labels not in schema: vertex 'patient' x1" in final
    assert "**Warning:** 1 extracted vertex/edge item(s) did not fit the schema" in final
    assert not app._EXTRACTION_CAPTURES  # capture is released after each file


def test_extraction_recorder_captures_raw_llm_reply_once_installed():
    app.install_extraction_recorder()
    app.install_extraction_recorder()  # idempotent
    extractor = app.PropertyGraphExtract(FakeLLM(["RAW REPLY"]), example_prompt=None)
    with app.capture_extractions(["chunk"]) as captured:
        assert extractor.extract_property_graph_by_llm("{}", "chunk") == "RAW REPLY"
    assert captured == ["RAW REPLY"]


def test_last_gremlin_block_runs_the_query_the_model_corrected(monkeypatch):
    monkeypatch.setattr(
        app.GremlinGenerateSynthesize, "_extract_response", app.GremlinGenerateSynthesize._extract_response
    )
    app.install_last_gremlin_block()
    app.install_last_gremlin_block()  # idempotent
    reply = (
        "```gremlin\ng.V().hasLabel('person').in('prosecuted_by').count()\n```\n\n"
        "Wait - the edge goes from the defendant to the prosecutor. Corrected query:\n\n"
        "```gremlin\ng.V().hasLabel('person').where(__.in('prosecuted_by')).count()\n```"
    )
    extract = app.GremlinGenerateSynthesize._extract_response

    assert extract(None, reply) == "g.V().hasLabel('person').where(__.in('prosecuted_by')).count()"
    assert extract(None, "```gremlin\ng.V().count()\n```") == "g.V().count()"
    assert extract(None, "g.V().count()") == "g.V().count()"


class RecordingLiteLLM(LiteLLMClient):
    def __init__(self, model_name):
        super().__init__(api_key="k", api_base="http://gateway", model_name=model_name)
        self.calls = []

    def generate(self, messages=None, prompt=None):
        self.calls.append({"messages": messages, "prompt": prompt})
        return '{"vertices": [], "edges": []}'


@pytest.fixture
def prompt_caching(monkeypatch):
    """Install prompt caching over the stock extraction call, and record warm-up calls instead of sending them."""
    monkeypatch.setattr(
        app.PropertyGraphExtract, "extract_property_graph_by_llm", app.PropertyGraphExtract.extract_property_graph_by_llm
    )
    monkeypatch.setattr(app, "_CACHE_WARMED", {})
    warm_ups = []
    monkeypatch.setattr(app.litellm, "completion", lambda **kwargs: warm_ups.append(kwargs))
    app.install_prompt_caching()
    app.install_prompt_caching()  # idempotent
    return warm_ups


def test_prompt_caching_sends_the_instructions_as_a_cached_block_ahead_of_the_document(prompt_caching):
    llm = RecordingLiteLLM("anthropic/claude-sonnet")
    extractor = app.PropertyGraphExtract(llm, example_prompt="INSTRUCTIONS")

    extractor.extract_property_graph_by_llm('{"vertexlabels": []}', "Alice works at Acme.")
    extractor.extract_property_graph_by_llm('{"vertexlabels": []}', "Bob works at Initech.")

    first, second = (call["messages"][0]["content"] for call in llm.calls)
    assert first[0] == {"type": "text", "text": "INSTRUCTIONS", "cache_control": {"type": "ephemeral"}}
    assert first[1]["text"] == app.generate_extract_property_graph_prompt("Alice works at Acme.", '{"vertexlabels": []}')
    assert first[1]["text"].index("Alice") < first[1]["text"].index("vertexlabels")  # prompt order is unchanged
    assert second[0] == first[0]
    assert len(prompt_caching) == 1  # one warm-up, then both calls read the cache
    warm_up = prompt_caching[0]
    assert warm_up["messages"][0]["content"][0] == first[0]
    assert warm_up["max_tokens"] == 16 and warm_up["base_url"] == "http://gateway"


def test_prompt_caching_re_warms_after_the_cache_would_have_expired(prompt_caching, monkeypatch):
    llm = RecordingLiteLLM("anthropic/claude-sonnet")
    extractor = app.PropertyGraphExtract(llm, example_prompt="INSTRUCTIONS")
    clock = iter([0.0, 1000.0, 1000.0])  # first warm-up, then the next call finds it stale
    monkeypatch.setattr(app.time, "monotonic", lambda: next(clock))

    extractor.extract_property_graph_by_llm("{}", "Alice works at Acme.")
    extractor.extract_property_graph_by_llm("{}", "Bob works at Initech.")

    assert len(prompt_caching) == 2


def test_prompt_caching_leaves_other_providers_unchanged(prompt_caching):
    llm = RecordingLiteLLM("openai/gpt-x")
    extractor = app.PropertyGraphExtract(llm, example_prompt="INSTRUCTIONS")

    extractor.extract_property_graph_by_llm("{}", "Alice works at Acme.")

    assert llm.calls[0]["messages"] is None
    assert llm.calls[0]["prompt"] == "INSTRUCTIONS" + app.generate_extract_property_graph_prompt("Alice works at Acme.", "{}")
    assert prompt_caching == []


def test_prompt_caching_extracts_even_when_the_warm_up_fails(prompt_caching, monkeypatch):
    def failing_completion(**kwargs):
        raise RuntimeError("gateway down")

    monkeypatch.setattr(app.litellm, "completion", failing_completion)
    llm = RecordingLiteLLM("anthropic/claude-sonnet")

    reply = app.PropertyGraphExtract(llm, example_prompt="INSTRUCTIONS").extract_property_graph_by_llm("{}", "Alice")

    assert app.parse_llm_graph(reply) == {"vertices": [], "edges": []}
    assert len(llm.calls) == 1


def test_split_retry_re_extracts_a_truncated_chunk_in_pieces_and_combines_them(monkeypatch):
    calls = []

    def fake_llm_extract(self, schema, chunk):
        calls.append(chunk)
        if len(chunk) > 3000:
            return '{"vertices": [{"label": "person", "properties": {"name": "Brit'  # cut off at the token limit
        name = "Britany Brown" if chunk.startswith("A") else "Andrew Hooper"
        return json.dumps({"vertices": [{"label": "person", "properties": {"name": name}}], "edges": []})

    monkeypatch.setattr(app.PropertyGraphExtract, "extract_property_graph_by_llm", fake_llm_extract)
    app.install_split_retry()
    app.install_split_retry()  # idempotent
    chunk = "A" * 2000 + "\n\n" + "B" * 2000

    reply = app.PropertyGraphExtract.extract_property_graph_by_llm(None, "{}", chunk)

    assert calls == [chunk, "A" * 2000, "B" * 2000]
    graph = app.parse_llm_graph(reply)
    assert [v["properties"]["name"] for v in graph["vertices"]] == ["Britany Brown", "Andrew Hooper"]


def test_audit_sees_the_combined_reply_when_split_retry_sits_inside_the_recorder(monkeypatch):
    def fake_llm_extract(self, schema, chunk):
        if len(chunk) > 3000:
            return '{"vertices": [{"label": "person"'
        return json.dumps({"vertices": [{"label": "person", "properties": {"name": chunk[0]}}], "edges": []})

    monkeypatch.setattr(app.PropertyGraphExtract, "extract_property_graph_by_llm", fake_llm_extract)
    app.install_split_retry()
    app.install_extraction_recorder()
    chunk = "A" * 2000 + "\n\n" + "B" * 2000

    with app.capture_extractions([chunk]) as captured:
        app.PropertyGraphExtract.extract_property_graph_by_llm(None, "{}", chunk)

    assert len(captured) == 1
    assert app.audit_extraction(captured, PEOPLE_SCHEMA)["unparsed"] == 0


def test_split_retry_leaves_valid_and_small_chunks_alone(monkeypatch):
    calls = []

    def fake_llm_extract(self, schema, chunk):
        calls.append(chunk)
        return "not json" if chunk == "small" else '{"vertices": [], "edges": []}'

    monkeypatch.setattr(app.PropertyGraphExtract, "extract_property_graph_by_llm", fake_llm_extract)
    app.install_split_retry()

    assert app.PropertyGraphExtract.extract_property_graph_by_llm(None, "{}", "small") == "not json"
    big = "x" * 2000 + "\n\n" + "y" * 2000
    assert app.PropertyGraphExtract.extract_property_graph_by_llm(None, "{}", big) == '{"vertices": [], "edges": []}'
    assert calls == ["small", big]


def test_overlapping_captures_each_get_only_their_own_chunks_replies():
    with app.capture_extractions(["a1", "a2"]) as first, app.capture_extractions(["b1", "a1"]) as second:
        for chunk in ("b1", "a1", "a2", "a1", "stray"):
            app.record_extraction(f"reply to {chunk}", chunk)
    assert first == ["reply to a1", "reply to a2"]
    assert second == ["reply to b1", "reply to a1"]  # the shared chunk goes to whichever capture still needs it
    assert not app._EXTRACTION_CAPTURES


def test_ingest_files_extracts_files_concurrently_but_imports_them_in_upload_order(tmp_path, monkeypatch):
    import threading

    monkeypatch.setenv("EXTRACT_WORKERS", "3")
    paths = []
    for name in ("Alice", "Bob", "Carol"):
        doc = tmp_path / f"{name}.txt"
        doc.write_text(f"{name} works at Acme.", encoding="utf-8")
        paths.append(str(doc))
    barrier = threading.Barrier(3)  # only released once all three files are being extracted at the same time

    def extract(schema, texts, *args, **kwargs):
        barrier.wait(timeout=5)
        name = texts[0].split()[0]
        return json.dumps({"vertices": [{"id": f"1:{name}", "label": "person"}], "edges": []})

    scheduler = FakeScheduler({FlowName.GRAPH_EXTRACT: extract, FlowName.IMPORT_GRAPH_DATA: "{}"})
    final = list(
        app.ingest_files(
            paths, SCHEMA_TEXT, scheduler=scheduler, existing_ids=lambda ids: set(), existing_values=lambda l, k: []
        )
    )[-1]

    imports = [c[1][0]["vertices"][0]["id"] for c in scheduler.calls if c[0] == FlowName.IMPORT_GRAPH_DATA]
    assert imports == ["1:Alice", "1:Bob", "1:Carol"]
    assert "3 vertices and 0 edges imported" in final


def test_poll_agent_log_skips_render_until_log_changes():
    app.agent_log.LOG.clear()
    text, version = app.clear_agent_log(list(app.agent_log.KINDS))
    assert text == "(no activity yet)"
    assert app.poll_agent_log(list(app.agent_log.KINDS), True, version)[1] == version  # unchanged -> skip

    app.agent_log.LOG.add("LLM", "OllamaClient qwen", "hi", "hello", 0.5)
    text, new_version = app.poll_agent_log(list(app.agent_log.KINDS), True, version)
    assert new_version != version and ">>> request\nhi" in text
    assert app.poll_agent_log(["HugeGraph"], False, version)[1] == version  # auto-refresh off
    app.agent_log.LOG.clear()


def test_showcase_gremlin_prompt_removes_the_give_up_branch(monkeypatch):
    base = (
        "intro\n## Complex Query Detection:\nIf complex, return\n```gremlin\ng.V().limit(0)\n```\n"
        "## Gremlin Query Generation (Executed only if the query is not complex):\nGraph Schema:\n{schema}"
    )
    monkeypatch.setattr(app.prompt, "gremlin_generate_prompt", base)
    built = app.showcase_gremlin_prompt()
    assert "Complex Query Detection" not in built and "only if the query is not complex" not in built
    assert "Never answer with g.V().limit(0)" in built
    assert "TextP.containing" in built


def test_answer_question_explains_a_declined_query():
    res = {"template_gremlin": "", "raw_gremlin": "g.V().limit(0)", "raw_execution_result": '{"data": []}'}
    scheduler = FakeScheduler({FlowName.TEXT2GREMLIN: res})
    answer, gremlin, raw = app.answer_question("show all frauds in New York", scheduler=scheduler, schema=SCHEMA)
    assert "did not write a query" in answer and gremlin == "g.V().limit(0)" and raw is None


def test_showcase_gremlin_prompt_adds_generic_rules_and_keeps_format_placeholders():
    built = app.showcase_gremlin_prompt()
    assert built.index("# HugeGraph Rules") < built.index("Graph Schema:")
    filled = built.format(query="Q?", schema="S", example="E", vertices="V", properties="P")
    assert "hasLabel('LABEL')" in filled and "Q?" in filled
    assert "'person'" not in filled and "'location'" not in filled  # nothing dataset-specific




class TokenGeneratorLLM:
    """Streams like hugegraph-llm's Ollama/OpenAI clients: yields tokens and passes each to the callback."""

    def __init__(self, tokens):
        self.tokens = tokens

    def generate_streaming(self, messages=None, prompt=None, on_token_callback=None):
        for token in self.tokens:
            if on_token_callback:
                on_token_callback(token)
            yield token


class CallbackOnlyLLM:
    """Streams like hugegraph-llm's LiteLLM client: blocks, sends chunk objects to the callback, returns the text."""

    def __init__(self, tokens, fail=False):
        self.tokens, self.fail = tokens, fail

    def generate_streaming(self, messages=None, prompt=None, on_token_callback=None):
        for token in self.tokens:
            on_token_callback(SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=token))]))
        if self.fail:
            raise RuntimeError("stream dropped")
        return "".join(self.tokens)


@pytest.mark.parametrize("llm_cls", [TokenGeneratorLLM, CallbackOnlyLLM])
def test_stream_llm_text_yields_growing_text_for_both_client_styles(llm_cls):
    steps = list(app.stream_llm_text(llm_cls(["Al", "ice", " is", " here."]), "p"))
    assert steps[-1] == "Alice is here."
    assert all(steps[-1].startswith(step) for step in steps)  # each step extends the previous one


def test_stream_llm_text_falls_back_to_generate_and_reports_errors_and_empty_replies():
    assert list(app.stream_llm_text(FakeLLM(["whole answer"]), "p")) == ["whole answer"]
    with pytest.raises(RuntimeError, match="stream dropped"):
        list(app.stream_llm_text(CallbackOnlyLLM(["partial"], fail=True), "p"))
    assert list(app.stream_llm_text(TokenGeneratorLLM([]), "p")) == ["(The model returned an empty summary.)"]


def test_stream_answer_shows_query_first_then_streams_the_summary():
    data = [{"id": "1:Alice", "label": "person"}]
    scheduler = FakeScheduler(
        {FlowName.TEXT2GREMLIN: {"template_gremlin": "g.V()", "template_execution_result": _exec(data)}}
    )
    steps = list(
        app.stream_answer(
            "Who?", scheduler=scheduler, llm=TokenGeneratorLLM(["Alice", " is a person."]), schema=SCHEMA,
        )
    )
    assert steps[0] == ("_Found 1 result(s); summarizing..._", "g.V()", data)
    assert steps[-1] == ("Alice is a person.", "g.V()", data)
    assert all(step[1:] == ("g.V()", data) for step in steps)


def test_stream_answer_reports_a_summary_that_fails_mid_stream():
    data = [{"id": "1:Alice"}]
    scheduler = FakeScheduler(
        {FlowName.TEXT2GREMLIN: {"template_gremlin": "g.V()", "template_execution_result": _exec(data)}}
    )
    final = app.answer_question(
        "Who?", scheduler=scheduler, llm=CallbackOnlyLLM(["Al"], fail=True), schema=SCHEMA,
    )
    assert final[0].startswith("(Summarization failed: stream dropped)") and final[2] == data


class ChunkLLM:
    """Replies with one person vertex named after the chunk; waits on a barrier to prove calls overlap."""

    def __init__(self, barrier=None):
        self.barrier = barrier

    def generate(self, prompt=None, **kwargs):
        if self.barrier:
            self.barrier.wait(timeout=5)
        name = prompt.rsplit("CHUNK:", 1)[1].split()[0]
        return json.dumps({"vertices": [{"label": "person", "type": "vertex", "properties": {"name": name}}], "edges": []})


def test_parallel_extraction_overlaps_llm_calls_and_matches_sequential_result(monkeypatch):
    import threading

    sequential = app.PropertyGraphExtract.run
    monkeypatch.setattr(app.PropertyGraphExtract, "run", sequential)  # restored after the test
    monkeypatch.setenv("EXTRACT_WORKERS", "3")
    chunks = [f"CHUNK:{name}" for name in ("alice", "bob", "carol")]
    schema = SCHEMA
    expected = sequential(app.PropertyGraphExtract(ChunkLLM(), example_prompt=None), {"schema": schema, "chunks": chunks})

    app.install_parallel_extraction()
    app.install_parallel_extraction()  # idempotent
    # ExtractNode passes in the chat LLM; the patched run must use the extract-role LLM instead.
    monkeypatch.setattr(app, "get_extract_llm", lambda settings: ChunkLLM(threading.Barrier(3)))
    extractor = app.PropertyGraphExtract(FakeLLM([]), example_prompt=None)
    result = app.PropertyGraphExtract.run(extractor, {"schema": schema, "chunks": chunks})

    assert [v["properties"]["name"] for v in result["vertices"]] == ["alice", "bob", "carol"]
    assert result == expected


def test_extract_workers_defaults_to_sequential_for_ollama(monkeypatch):
    monkeypatch.delenv("EXTRACT_WORKERS", raising=False)
    monkeypatch.setattr(llm_settings, "chat_llm_type", "ollama/local")
    assert app.extract_workers() == 1
    monkeypatch.setattr(llm_settings, "chat_llm_type", "litellm")
    assert app.extract_workers() == app.HOSTED_EXTRACT_WORKERS
    monkeypatch.setenv("EXTRACT_WORKERS", "8")
    assert app.extract_workers() == 8


def test_ask_with_history_streams_then_moves_the_answer_into_a_new_card():
    def fake_answer(question):
        yield "_Found 1 result(s); summarizing..._", "g.V()", [1]
        yield f"Answer to {question}", "g.V()", [1]

    earlier = [{"question": "Q1", "answer": "A1", "gremlin": "g.E()", "raw": []}]

    steps = list(app.ask_with_history("Who? ", earlier, use_context=False, answer_fn=fake_answer))

    assert steps[0][:3] == ("_Found 1 result(s); summarizing..._", "g.V()", [1])
    assert steps[1][:3] == ("Answer to Who?", "g.V()", [1])
    assert all(isinstance(value, type(app.gr.skip())) for step in steps[:2] for value in step[3:])
    live_answer, live_gremlin, live_raw, history, question_box = steps[-1]
    assert (live_answer, live_gremlin, live_raw, question_box) == ("", "", None, "")
    assert history == earlier + [{"question": "Who?", "answer": "Answer to Who?", "gremlin": "g.V()", "raw": [1]}]
    assert len(earlier) == 1  # the previous history list is not mutated


def test_ask_with_history_passes_errors_through_without_adding_a_card():
    def failing_answer(question):
        raise app.gr.Error("Please type a question.")
        yield  # pragma: no cover

    with pytest.raises(app.gr.Error):
        list(app.ask_with_history("", [], answer_fn=failing_answer))


@pytest.mark.parametrize(
    ("gremlin", "relaxed"),
    [
        (
            "g.V().hasLabel('bank').has('name', 'TD Bank').in('employed_by').values('name')",
            "g.V().hasLabel('bank').has('name', TextP.regex('(?i).*\\\\QTD Bank\\\\E.*')).in('employed_by').values('name')",
        ),
        (
            "g.V().hasLabel('offense').has('name', TextP.containing('Bank Fraud'))",
            "g.V().hasLabel('offense').has('name', TextP.regex('(?i).*\\\\QBank Fraud\\\\E.*'))",
        ),
        ('g.V().hasLabel("p").has("plea_status","guilty")', 'g.V().hasLabel("p").has("plea_status", TextP.regex("(?i).*\\\\Qguilty\\\\E.*"))'),
        (  # the bracket closing where(...) stays
            "g.V().hasLabel('person').where(outE('charged_with').has('plea_status', 'pending')).values('name')",
            "g.V().hasLabel('person').where(outE('charged_with').has('plea_status', "
            "TextP.regex('(?i).*\\\\Qpending\\\\E.*'))).values('name')",
        ),
        (
            "g.V().where(out('x').has('name', TextP.containing('Acme'))).count()",
            "g.V().where(out('x').has('name', TextP.regex('(?i).*\\\\QAcme\\\\E.*'))).count()",
        ),
        ("g.V().hasLabel('person').has('age', gt(50))", None),  # not a text literal
        ("g.V().has('bank', 'name', 'TD Bank')", None),  # three-argument form
        ("g.V().hasLabel('x').has('name', TextP.regex('(?i)td'))", None),  # already a regex
    ],
)
def test_relax_exact_matches_turns_text_filters_into_partial_matches(gremlin, relaxed):
    assert app.relax_exact_matches(gremlin) == relaxed


def test_stream_answer_retries_an_empty_result_with_partial_name_matches():
    exact = "g.V().hasLabel('bank').has('name', 'TD Bank').in('employed_by').values('name')"
    scheduler = FakeScheduler(
        {FlowName.TEXT2GREMLIN: {"template_gremlin": exact, "template_execution_result": _exec([])}}
    )
    queries = []

    def run_query(query):
        queries.append(query)
        return ["Cheungkin Lam", "Edward Low"]

    summary, gremlin, raw = app.answer_question(
        "Who worked at TD Bank?", scheduler=scheduler, llm=FakeLLM(["Cheungkin Lam and Edward Low."]), schema={},
        run_query=run_query,
    )

    assert queries == [
        "g.V().hasLabel('bank').has('name', TextP.regex('(?i).*\\\\QTD Bank\\\\E.*')).in('employed_by').values('name')"
        ".limit(100)"
    ]
    assert gremlin == app.relax_exact_matches(exact)
    assert raw == ["Cheungkin Lam", "Edward Low"]
    assert summary.startswith("_No exact match for a name in the question, so it was matched partially")
    assert summary.endswith("Cheungkin Lam and Edward Low.")


def test_stream_answer_reports_no_results_when_the_partial_retry_finds_nothing_too():
    scheduler = FakeScheduler(
        {FlowName.TEXT2GREMLIN: {"template_gremlin": "g.V().has('name', 'Nobody')", "template_execution_result": _exec([])}}
    )

    summary, _, raw = app.answer_question(
        "Who is Nobody?", scheduler=scheduler, llm=FakeLLM([]), schema={},
        run_query=lambda query: [],
    )

    assert "returned no results" in summary and raw == []


def test_gremlin_rules_ask_for_partial_matches_on_organization_names():
    assert "Match a person's full name exactly" in app.GREMLIN_RULES
    assert "'Acme Corporation, Inc.' does not equal 'Acme'" in app.GREMLIN_RULES


LUNEKE_TURN = {
    "question": "Which bank did Aaron T. Luneke defraud?",
    "answer": "Aaron T. Luneke defrauded Bank of the Valley.",
    "gremlin": "g.V()",
    "raw": ["Bank of the Valley"],
}


def test_standalone_question_sends_recent_turns_and_returns_the_rewrite():
    llm = FakeLLM(["  `Which judge tried Aaron T. Luneke?`\n"])
    history = [{"question": f"old {i}", "answer": "a", "raw": [i]} for i in range(4)] + [LUNEKE_TURN]

    assert app.standalone_question("Which judge tried him?", history, llm=llm) == "Which judge tried Aaron T. Luneke?"

    prompt = llm.calls[0]["prompt"]
    assert "Follow-up question: Which judge tried him?" in prompt
    assert "A3: Aaron T. Luneke defrauded Bank of the Valley." in prompt
    assert 'Results3: ["Bank of the Valley"]' in prompt
    assert "old 1" not in prompt and "old 2" in prompt  # only the last CONTEXT_TURNS turns


def test_standalone_question_truncates_long_results_and_prefers_earlier_rewrites():
    turn = {"question": "And him?", "standalone": "Who is Bob?", "answer": "Bob.", "raw": ["x" * 5000]}
    llm = FakeLLM(["Q"])

    app.standalone_question("Next?", [turn], llm=llm)

    prompt = llm.calls[0]["prompt"]
    assert "Q1: Who is Bob?" in prompt and "And him?" not in prompt
    assert "...(truncated)" in prompt and "x" * (app.CONTEXT_RESULT_CHARS + 1) not in prompt


def test_standalone_question_falls_back_to_the_question_without_history_or_on_errors():
    class BrokenLLM:
        def generate(self, **kwargs):
            raise RuntimeError("gateway down")

    assert app.standalone_question("Who?", [], llm=BrokenLLM()) == "Who?"
    assert app.standalone_question("Who?", [LUNEKE_TURN], llm=BrokenLLM()) == "Who?"
    assert app.standalone_question("Who?", [LUNEKE_TURN], llm=FakeLLM(["  \n"])) == "Who?"


def test_ask_with_history_answers_the_rewritten_follow_up_and_says_so():
    asked = []

    def fake_answer(question):
        asked.append(question)
        yield "Brian C. Buescher.", "g.V().hasLabel('defendant')", ["Brian C. Buescher"]

    steps = list(
        app.ask_with_history(
            "Which judge tried him?", [LUNEKE_TURN], answer_fn=fake_answer,
            rewrite_fn=lambda question, history: "Which judge tried Aaron T. Luneke?",
        )
    )

    assert asked == ["Which judge tried Aaron T. Luneke?"]
    assert steps[0][0].startswith("_Reading the follow-up")
    assert steps[1][0] == "_Interpreted as: Which judge tried Aaron T. Luneke?_\n\nBrian C. Buescher."
    card = steps[-1][3][-1]
    assert card["question"] == "Which judge tried him?"
    assert card["standalone"] == "Which judge tried Aaron T. Luneke?"
    assert card["answer"].startswith("_Interpreted as:")


def test_ask_with_history_skips_the_rewrite_when_context_is_off_or_unchanged():
    def fake_answer(question):
        yield f"Answer to {question}", "g.V()", []

    def must_not_rewrite(question, history):
        raise AssertionError("rewrite must not run")

    off = list(app.ask_with_history("Him?", [LUNEKE_TURN], use_context=False, answer_fn=fake_answer,
                                    rewrite_fn=must_not_rewrite))
    unchanged = list(app.ask_with_history("Who is Bob?", [LUNEKE_TURN], answer_fn=fake_answer,
                                          rewrite_fn=lambda question, history: question))

    assert off[-1][3][-1] == {"question": "Him?", "answer": "Answer to Him?", "gremlin": "g.V()", "raw": []}
    assert "standalone" not in unchanged[-1][3][-1]
    assert unchanged[-1][3][-1]["answer"] == "Answer to Who is Bob?"


TYPED_SCHEMA = {
    "vertexlabels": [
        {"name": "person", "primary_keys": ["name"], "properties": ["name", "age", "fraud_amount", "residence"]},
        {"name": "charge", "primary_keys": ["name"], "properties": ["name", "max_fine", "max_prison_years"]},
    ],
    "edgelabels": [
        {"name": "tried_before", "source_label": "person", "target_label": "charge", "properties": ["sentencing_date"]}
    ],
    "propertykeys": [
        {"name": "name", "data_type": "TEXT", "cardinality": "SINGLE"},
        {"name": "age", "data_type": "INT", "cardinality": "SINGLE"},
        {"name": "fraud_amount", "data_type": "DOUBLE", "cardinality": "SINGLE"},
        {"name": "max_fine", "data_type": "DOUBLE", "cardinality": "SINGLE"},
        {"name": "max_prison_years", "data_type": "INT", "cardinality": "SINGLE"},
        {"name": "residence", "data_type": "TEXT", "cardinality": "SINGLE"},
        {"name": "sentencing_date", "data_type": "DATE", "cardinality": "SINGLE"},
    ],
}


def test_coerce_property_types_fixes_values_the_importer_would_reject():
    # Shaped like the Mercedes Henry and charge vertices that hugegraph-llm skipped as "not correct".
    extracted = {
        "vertices": [
            {"id": "1:Mercedes Henry", "label": "person",
             "properties": {"name": "Mercedes Henry", "age": 35.0, "fraud_amount": 931500, "residence": 30303}},
            {"id": "3:bank fraud", "label": "charge",
             "properties": {"name": "bank fraud", "max_fine": "$1,000,000", "max_prison_years": "30"}},
            {"id": "3:wire fraud", "label": "charge", "properties": {"name": "wire fraud", "max_fine": "4.3 million"}},
        ],
        "edges": [{"label": "tried_before", "outV": "1:Mercedes Henry", "inV": "3:bank fraud",
                   "properties": {"sentencing_date": "2026-07-14 00:00:00"}}],
    }

    converted, dropped = app.coerce_property_types(extracted, TYPED_SCHEMA)

    henry, bank_fraud, wire_fraud = (v["properties"] for v in extracted["vertices"])
    assert henry == {"name": "Mercedes Henry", "age": 35, "fraud_amount": 931500.0, "residence": "30303"}
    assert type(henry["fraud_amount"]) is float and type(henry["age"]) is int
    assert bank_fraud == {"name": "bank fraud", "max_fine": 1000000.0, "max_prison_years": 30}
    assert wire_fraud["max_fine"] == 4300000.0
    assert extracted["edges"][0]["properties"] == {"sentencing_date": "2026-07-14"}
    assert converted == 7 and not dropped


def test_coerce_property_types_drops_only_the_unconvertible_property():
    extracted = {
        "vertices": [{"id": "1:Al", "label": "person",
                      "properties": {"name": "Al", "age": "about forty", "fraud_amount": 1.5}}],
        "edges": [{"label": "tried_before", "properties": {"sentencing_date": "next spring"}}],
    }

    converted, dropped = app.coerce_property_types(extracted, TYPED_SCHEMA)

    assert extracted["vertices"][0]["properties"] == {"name": "Al", "fraud_amount": 1.5}
    assert extracted["edges"][0]["properties"] == {}
    assert converted == 0 and dropped == Counter({"person.age": 1, "tried_before.sentencing_date": 1})


def test_ingest_files_converts_types_before_import_and_reports_dropped_values(tmp_path):
    doc = tmp_path / "a.txt"
    doc.write_text("Mercedes Henry ...", encoding="utf-8")
    extracted = {
        "vertices": [{"id": "1:Mercedes Henry", "label": "person",
                      "properties": {"name": "Mercedes Henry", "fraud_amount": 931500, "age": "unknown"}}],
        "edges": [],
    }
    scheduler = FakeScheduler({FlowName.GRAPH_EXTRACT: json.dumps(extracted), FlowName.IMPORT_GRAPH_DATA: "{}"})

    final = list(
        app.ingest_files([str(doc)], json.dumps(TYPED_SCHEMA), scheduler=scheduler,
                         existing_ids=lambda ids: set(), existing_values=lambda label, key: [])
    )[-1]

    imported = next(c for c in scheduler.calls if c[0] == FlowName.IMPORT_GRAPH_DATA)[1][0]
    assert imported["vertices"][0]["properties"] == {"name": "Mercedes Henry", "fraud_amount": 931500.0}
    assert "dropped 1 property value(s) of the wrong type (person.age x1)" in final




def test_gremlin_rules_ask_for_the_values_a_question_filters_on():
    assert "return that value with each name: .project('name', 'KEY')" in app.GREMLIN_RULES
    assert "return only the top item with its count or value" in app.GREMLIN_RULES
    assert "Return a ranking only when the question asks for several items" in app.GREMLIN_RULES
    assert "{" not in app.GREMLIN_RULES  # the rules are passed through str.format


def test_gremlin_rules_keep_rows_whose_returned_property_is_missing():
    assert "by(coalesce(values('KEY'), constant('')))" in app.GREMLIN_RULES


@pytest.mark.parametrize(
    ("data_type", "value", "expected"),
    [
        ("DOUBLE", "more than $500,000", 500000.0),
        ("DOUBLE", "approximately $373,000", 373000.0),
        ("DOUBLE", "US$54.5 billion", 54.5e9),
        ("INT", "one count", 1),
        ("INT", "8 counts", 8),
        ("INT", "30 years", 30),
        ("DATE", "July 14, 2026", "2026-07-14"),
        ("DATE", "Sept. 9, 2026", "2026-09-09"),
        ("DATE", "Aug. 18, 2026", "2026-08-18"),
        ("DATE", "07/14/2026", "2026-07-14"),
    ],
)
def test_coerce_reads_numbers_and_dates_as_written_in_documents(data_type, value, expected):
    assert app._coerce(data_type, value) == expected


@pytest.mark.parametrize(
    ("data_type", "value"),
    [("INT", "about forty"), ("DATE", "next spring"), ("DATE", "Mayday 3, 2026"), ("DATE", "13/40/2026")],
)
def test_coerce_rejects_values_it_cannot_read(data_type, value):
    assert app._coerce(data_type, value) is app._INVALID


def test_gremlin_rules_compare_dates_as_ranges():
    assert "Compare DATE properties as ranges of 'yyyy-MM-dd' values, never with TextP" in app.GREMLIN_RULES












AGE_SCHEMA = {
    "vertexlabels": [{"name": "person", "primary_keys": ["name"], "properties": ["name", "age"]}],
    "edgelabels": [],
    "propertykeys": [{"name": "name", "data_type": "TEXT"}, {"name": "age", "data_type": "INT"}],
}
UNTYPED_AGE_SCHEMA = {"vertexlabels": AGE_SCHEMA["vertexlabels"], "edgelabels": []}  # no propertykeys


def test_all_text_properties_flags_a_schema_without_property_types():
    assert app.all_text_properties(app.parse_schema(json.dumps(UNTYPED_AGE_SCHEMA)))
    assert not app.all_text_properties(app.parse_schema(json.dumps(AGE_SCHEMA)))
    assert not app.all_text_properties(app.parse_schema(SCHEMA_TEXT))  # only key properties, nothing to type


def test_infer_schema_infers_again_when_every_property_is_text(tmp_path):
    sample = tmp_path / "sample.txt"
    sample.write_text("Alice, 30, works at Acme.", encoding="utf-8")
    replies = iter([json.dumps(UNTYPED_AGE_SCHEMA), json.dumps(AGE_SCHEMA)])
    scheduler = FakeScheduler({FlowName.BUILD_SCHEMA: lambda *args: next(replies)})

    schema_text, summary = app.infer_schema(str(sample), scheduler=scheduler)

    assert len(scheduler.calls) == 2
    assert {p["name"]: p["data_type"] for p in json.loads(schema_text)["propertykeys"]}["age"] == "INT"
    assert not summary.startswith("**Warning:**")


def test_infer_schema_warns_when_every_attempt_is_text(tmp_path):
    sample = tmp_path / "sample.txt"
    sample.write_text("Alice, 30, works at Acme.", encoding="utf-8")
    scheduler = FakeScheduler({FlowName.BUILD_SCHEMA: json.dumps(UNTYPED_AGE_SCHEMA)})

    _, summary = app.infer_schema(str(sample), scheduler=scheduler)
    assert len(scheduler.calls) == app.SCHEMA_ATTEMPTS
    assert summary.startswith("**Warning:** every property is TEXT")


def _timeout():
    return app.litellm.Timeout(message="timed out", model="m", llm_provider="anthropic")


@pytest.fixture
def llm_retries(monkeypatch):
    """Install the timeout and retry over a scripted LiteLLMClient.generate; returns a function taking the replies."""
    monkeypatch.setattr(LiteLLMClient, "generate", LiteLLMClient.generate)
    monkeypatch.setattr(app.litellm, "request_timeout", app.litellm.request_timeout)
    monkeypatch.setattr(app.time, "sleep", lambda secs: None)
    monkeypatch.delenv("LLM_TIMEOUT", raising=False)
    monkeypatch.delenv("LLM_RETRIES", raising=False)
    calls = []

    def install(replies):
        def scripted(self, messages=None, prompt=None):
            calls.append(prompt)
            reply = replies.pop(0)
            if isinstance(reply, Exception):
                raise reply
            return reply

        monkeypatch.setattr(LiteLLMClient, "generate", scripted)
        app.install_llm_timeouts()
        app.install_llm_timeouts()  # idempotent
        return calls

    return install


def test_llm_calls_get_a_timeout(llm_retries, monkeypatch):
    llm_retries([])
    assert app.litellm.request_timeout == app.LLM_TIMEOUT_SECS
    monkeypatch.setenv("LLM_TIMEOUT", "60")
    app.install_llm_timeouts()
    assert app.litellm.request_timeout == 60.0


def test_llm_calls_are_retried_after_a_timeout_or_dropped_connection(llm_retries):
    dropped = app.litellm.InternalServerError(message="Server disconnected", llm_provider="anthropic", model="m")
    calls = llm_retries([_timeout(), dropped, "ok"])

    reply = LiteLLMClient(model_name="anthropic/claude-sonnet").generate(prompt="p")

    assert reply == "ok" and calls == ["p", "p", "p"]


def test_llm_calls_give_up_after_the_retries_and_never_retry_a_bad_request(llm_retries, monkeypatch):
    llm = LiteLLMClient(model_name="anthropic/claude-sonnet")
    calls = llm_retries([_timeout(), _timeout(), _timeout()])
    with pytest.raises(app.litellm.Timeout):
        llm.generate(prompt="p")
    assert len(calls) == app.LLM_RETRIES + 1

    monkeypatch.setenv("LLM_RETRIES", "0")
    calls = llm_retries([_timeout()])
    with pytest.raises(app.litellm.Timeout):
        llm.generate(prompt="p")
    assert len(calls) == app.LLM_RETRIES + 2  # one more call, no retry

    bad = app.litellm.BadRequestError(message="bad", model="m", llm_provider="anthropic")
    calls = llm_retries([bad])
    with pytest.raises(app.litellm.BadRequestError):
        llm.generate(prompt="p")
    assert len(calls) == app.LLM_RETRIES + 3


def test_settings_come_from_the_environment_then_connection_properties(monkeypatch, tmp_path):
    props = tmp_path / "connection.properties"
    props.write_text(
        "# remote cluster\n! another comment\n\nHUGEGRAPH_URL = ccycloud-1.example.site:8080\n"
        "HUGEGRAPH_GRAPHSPACE=ANALYTICS\nHUGEGRAPH_PASSWORD=s3cr=t\nnot a setting\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(app, "CONNECTION_PROPERTIES", props)
    for name in ("HUGEGRAPH_URL", "HUGEGRAPH_GRAPHSPACE", "HUGEGRAPH_PASSWORD", "HUGEGRAPH_GRAPH"):
        monkeypatch.delenv(name, raising=False)

    cfg = app.default_config()
    assert (cfg["graph_url"], cfg["graph_space"], cfg["graph_pwd"]) == ("ccycloud-1.example.site:8080", "ANALYTICS", "s3cr=t")
    assert cfg["graph_name"] == "hugegraph"  # not in the file: the default

    monkeypatch.setenv("HUGEGRAPH_URL", "10.0.0.9:8080")
    assert app.default_config()["graph_url"] == "10.0.0.9:8080"  # the environment wins
    assert app.read_properties(tmp_path / "missing.properties") == {}


def test_connection_properties_example_is_never_read(monkeypatch, tmp_path):
    (tmp_path / "connection.properties.example").write_text("HUGEGRAPH_URL=example-host:8080\n", encoding="utf-8")
    props = tmp_path / "connection.properties"
    monkeypatch.setattr(app, "CONNECTION_PROPERTIES", props)
    for name in ("HUGEGRAPH_URL", "HUGEGRAPH_PASSWORD"):
        monkeypatch.delenv(name, raising=False)

    assert app.default_config()["graph_url"] == "127.0.0.1:8080"  # no connection.properties: the built-in default

    props.write_text("HUGEGRAPH_PASSWORD=s3cret\n", encoding="utf-8")  # only the overridden key
    cfg = app.default_config()
    assert (cfg["graph_url"], cfg["graph_pwd"]) == ("127.0.0.1:8080", "s3cret")


def test_apply_config_sets_the_graph_space(monkeypatch):
    args = ("ollama", "qwen2.5", "nomic-embed-text", "", "", "127.0.0.1", 11434, "hg:8080", "kg", "u", "p")
    status = app.apply_config(*args, graph_space="ANALYTICS")
    assert huge_settings.graph_space == "ANALYTICS" and "HugeGraph `ANALYTICS/kg` at hg:8080" in status

    app.apply_config(*args, graph_space=" ")
    assert huge_settings.graph_space is None  # blank: the server's default

    monkeypatch.setenv("HUGEGRAPH_GRAPHSPACE", "FROM_ENV")
    app.apply_config(*args)  # not given: the configured value
    assert huge_settings.graph_space == "FROM_ENV"


class FakeResponse:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code, self.payload, self.text = status_code, payload, text
        self.ok = status_code < 400

    def json(self):
        return self.payload


@pytest.fixture
def graph_server(monkeypatch):
    """Point the app at a fake HugeGraph REST API; returns the recorded POSTs and a setter for the graph list."""
    huge_settings.graph_url, huge_settings.graph_name = "hg.example:8080", "kg"
    huge_settings.graph_user, huge_settings.graph_pwd, huge_settings.graph_space = "admin", "pw", None
    state = {"list": FakeResponse(200, {"graphs": []}), "create": FakeResponse(201, {}), "posts": []}

    def get(url, auth, timeout):
        assert url == "http://hg.example:8080/graphspaces/DEFAULT/graphs" and auth == ("admin", "pw")
        return state["list"]

    def post(url, params, json, auth, timeout):
        state["posts"].append((url, params, json))
        return state["create"]

    monkeypatch.setattr(app.requests, "get", get)
    monkeypatch.setattr(app.requests, "post", post)
    monkeypatch.delenv("HUGEGRAPH_BACKEND", raising=False)
    return state


def test_ensure_graph_exists_does_nothing_when_the_graph_is_there(graph_server):
    graph_server["list"] = FakeResponse(200, {"graphs": ["kg", "other"]})
    assert ENSURE_GRAPH_EXISTS() == "" and graph_server["posts"] == []


def test_ensure_graph_exists_copies_another_graph_config_or_uses_the_backend(graph_server, monkeypatch):
    graph_server["list"] = FakeResponse(200, {"graphs": ["sales", "hr"]})
    note = ENSURE_GRAPH_EXISTS()
    assert graph_server["posts"][-1] == (
        "http://hg.example:8080/graphspaces/DEFAULT/graphs/kg", {"clone_graph_name": "hr"}, {}
    )
    assert note == "Created graph `kg` in graph space `DEFAULT` with the configuration of `hr`."

    graph_server["list"] = FakeResponse(200, {"graphs": []})
    ENSURE_GRAPH_EXISTS()
    assert graph_server["posts"][-1][1:] == ({}, {
        "gremlin.graph": "org.apache.hugegraph.HugeFactory", "backend": "rocksdb", "serializer": "binary", "store": "kg"
    })

    monkeypatch.setenv("HUGEGRAPH_BACKEND", "hstore")
    graph_server["list"] = FakeResponse(200, {"graphs": ["sales"]})
    assert "with backend `hstore`" in ENSURE_GRAPH_EXISTS()  # an explicit backend wins over copying


def test_ensure_graph_exists_leaves_old_servers_alone_and_reports_failures(graph_server):
    graph_server["list"] = FakeResponse(404, text="no graph spaces")
    assert ENSURE_GRAPH_EXISTS() == "" and graph_server["posts"] == []

    graph_server["list"] = FakeResponse(200, {"graphs": []})
    graph_server["create"] = FakeResponse(403, text="Permission denied: space role required")
    with pytest.raises(RuntimeError, match="does not exist in graph space `DEFAULT` and could not be created: 403"):
        ENSURE_GRAPH_EXISTS()


def test_create_graph_schema_reports_a_created_graph(monkeypatch):
    monkeypatch.setattr(app, "ensure_graph_exists", lambda: "Created graph `kg` in graph space `DEFAULT` with backend `rocksdb`.")
    committer, committed = _committer(None)

    status = app.create_graph_schema(SCHEMA_TEXT, committer=committer)

    assert status.startswith("Created graph `kg` in graph space `DEFAULT`") and len(committed) == 1


def test_create_graph_schema_asks_for_a_schema_when_the_box_is_empty():
    for text in ("", "   \n", None):
        with pytest.raises(gr.Error, match="The schema box is empty. Click \\*Infer schema\\*"):
            app.create_graph_schema(text, committer=_committer(None)[0])


NEEDS_TEXT_FILTER_REWRITE = app.needs_text_filter_rewrite


@pytest.mark.parametrize(
    ("gremlin", "rewritten"),
    [
        (
            "g.V().hasLabel('bank').has('name', TextP.containing('TD Bank')).in('defrauded').values('name')",
            "g.V().hasLabel('bank').filter(__.values('name').is(TextP.containing('TD Bank'))).in('defrauded').values('name')",
        ),
        (  # inside where(...), with a regex
            "g.V().where(outE('charged_with').has('plea_status', TextP.regex('(?i).*\\\\Qpending\\\\E.*'))).count()",
            "g.V().where(outE('charged_with').filter(__.values('plea_status')"
            ".is(TextP.regex('(?i).*\\\\Qpending\\\\E.*')))).count()",
        ),
        (  # an anonymous traversal that starts with has(...)
            "g.V().where(has(\"name\", TextP.startingWith(\"Acme\"))).count()",
            "g.V().where(filter(__.values(\"name\").is(TextP.startingWith(\"Acme\")))).count()",
        ),
        ("g.V().hasLabel('person').has('age', P.gt(50)).values('name')", None),  # not a text predicate
        ("g.V().hasLabel('person').has('name', 'Alice')", None),  # exact match: supported everywhere
    ],
)
def test_rewrite_text_filters_moves_textp_out_of_has(gremlin, rewritten):
    assert app.rewrite_text_filters(gremlin) == (rewritten or gremlin)


def test_text_filters_are_rewritten_only_for_servers_older_than_1_8(monkeypatch):
    monkeypatch.setattr(app, "needs_text_filter_rewrite", NEEDS_TEXT_FILTER_REWRITE)
    monkeypatch.setattr(app, "_SERVER_VERSIONS", {})
    replies = {"new:8080": {"versions": {"core": "1.8.0"}}, "old:8080": {"versions": {"core": "1.7.0.0"}}}
    calls = []

    def get(url, timeout):
        calls.append(url)
        host = url.split("//")[1].split("/")[0]
        if host not in replies:
            raise ConnectionError("unreachable")
        return FakeResponse(200, replies[host])

    monkeypatch.setattr(app.requests, "get", get)
    for url, expected in (("new:8080", False), ("old:8080", True), ("down:8080", True), ("new:8080", False)):
        huge_settings.graph_url = url
        assert app.needs_text_filter_rewrite() is expected
    assert calls == ["http://new:8080/versions", "http://old:8080/versions", "http://down:8080/versions"]  # cached


def test_generated_queries_and_partial_retries_are_rewritten_for_old_servers(monkeypatch):
    monkeypatch.setattr(
        app.GremlinGenerateSynthesize, "_extract_response", app.GremlinGenerateSynthesize._extract_response
    )
    monkeypatch.setattr(app, "needs_text_filter_rewrite", lambda: True)
    app.install_text_filter_rewrite()
    app.install_text_filter_rewrite()  # idempotent
    reply = "```gremlin\ng.V().hasLabel('bank').has('name', TextP.containing('TD Bank')).values('name')\n```"
    assert app.GremlinGenerateSynthesize._extract_response(None, reply) == (
        "g.V().hasLabel('bank').filter(__.values('name').is(TextP.containing('TD Bank'))).values('name')"
    )

    queries = []
    relaxed, data = app.retry_with_partial_matches(
        "g.V().hasLabel('bank').has('name', 'TD Bank').values('name')",
        run_query=lambda query: queries.append(query) or ["Cheungkin Lam"],
    )
    # Servers before 1.8 have no TextP.regex, so the retry tries letter-case variants with TextP.containing.
    assert queries == [
        "g.V().hasLabel('bank').filter(__.values('name').is(TextP.containing('TD Bank')"
        ".or(TextP.containing('td bank').or(TextP.containing('TD BANK').or(TextP.containing('Td Bank')"
        ".or(TextP.containing('Td bank'))))))).values('name').limit(100)"
    ]
    assert data == ["Cheungkin Lam"]


def test_partial_retry_relaxes_an_already_rewritten_containing_filter():
    assert app.relax_exact_matches(
        "g.V().hasLabel('bank').filter(__.values('name').is(TextP.containing('TD Bank'))).values('name')"
    ) == "g.V().hasLabel('bank').filter(__.values('name').is(TextP.regex('(?i).*\\\\QTD Bank\\\\E.*'))).values('name')"


def test_case_variants_relax_both_filter_forms_without_regex():
    relaxed = app.relax_exact_matches(
        "g.V().hasLabel('person').filter(__.values('name').is(TextP.containing('James Keating')))"
        ".where(outE('charged_with').has('plea_status', 'pending')).values('name')",
        case_variants=True,
    )
    assert "regex" not in relaxed
    assert relaxed == (
        "g.V().hasLabel('person').filter(__.values('name').is(TextP.containing('James Keating')"
        ".or(TextP.containing('james keating').or(TextP.containing('JAMES KEATING').or(TextP.containing('James keating'))))))"
        ".where(outE('charged_with').filter(__.values('plea_status').is(TextP.containing('pending')"
        ".or(TextP.containing('PENDING').or(TextP.containing('Pending')))))).values('name')"
    )


def test_partial_retry_uses_regex_on_new_servers(monkeypatch):
    monkeypatch.setattr(app, "needs_text_filter_rewrite", lambda: False)
    queries = []
    app.retry_with_partial_matches(
        "g.V().hasLabel('bank').has('name', 'TD Bank').values('name')",
        run_query=lambda query: queries.append(query) or ["Cheungkin Lam"],
    )
    assert queries == ["g.V().hasLabel('bank').has('name', TextP.regex('(?i).*\\\\QTD Bank\\\\E.*')).values('name').limit(100)"]
