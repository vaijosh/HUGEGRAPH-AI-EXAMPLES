import asyncio

import pytest

import agent_log
from agent_log import AgentLog, clip, describe_graph, describe_llm, wrap_method


class FakeClient:
    model = "fake-model"

    def generate(self, messages=None, prompt=None):
        if prompt == "boom":
            raise RuntimeError("model down")
        return f"echo: {prompt}"

    async def agenerate(self, messages=None, prompt=None):
        return f"async echo: {prompt}"

    def outer(self, prompt=None):
        return self.generate(prompt=prompt)


@pytest.fixture
def log_and_client():
    log = AgentLog()

    class Client(FakeClient):
        pass

    Client.generate = FakeClient.generate  # wrap_method only replaces methods defined on the class itself
    Client.agenerate = FakeClient.agenerate
    for name in ("generate", "agenerate"):
        assert wrap_method(Client, name, agent_log.LLM, describe_llm, log=log)
    return log, Client()


def test_wrap_method_records_request_response_and_errors(log_and_client):
    log, client = log_and_client
    assert client.generate(prompt="hello") == "echo: hello"
    with pytest.raises(RuntimeError, match="model down"):
        client.generate(prompt="boom")

    ok, failed = log.entries()
    assert (ok.kind, ok.action, ok.request, ok.response, ok.error) == (
        "LLM", "Client fake-model", "hello", "echo: hello", ""
    )
    assert failed.error == "RuntimeError: model down" and failed.response == ""
    assert wrap_method(type(client), "generate", agent_log.LLM, describe_llm, log=log) is False  # already wrapped


def test_wrap_method_handles_async_methods(log_and_client):
    log, client = log_and_client
    assert asyncio.run(client.agenerate(prompt="hi")) == "async echo: hi"
    assert log.entries()[-1].response == "async echo: hi"


def test_nested_wrapped_calls_are_logged_once():
    log = AgentLog()

    class Client(FakeClient):
        pass

    Client.generate = FakeClient.generate
    Client.outer = FakeClient.outer
    wrap_method(Client, "generate", agent_log.LLM, describe_llm, log=log)
    wrap_method(Client, "outer", agent_log.LLM, describe_llm, log=log)
    Client().outer(prompt="x")
    assert len(log.entries()) == 1


def test_describe_llm_formats_chat_messages():
    action, text = describe_llm(FakeClient(), messages=[{"role": "user", "content": "q"}])
    assert (action, text) == ("FakeClient fake-model", "[user] q")


def test_describe_graph_shows_gremlin_query_text():
    assert describe_graph(None, "gremlin", "POST", json={"gremlin": "g.V().count()"}) == (
        "POST gremlin",
        "g.V().count()",
    )
    assert describe_graph(None, "graphs/hugegraph/schema") == ("GET graphs/hugegraph/schema", "")


def test_clip_truncates_long_text_and_dumps_objects():
    assert clip({"a": 1}) == '{"a": 1}'
    assert clip("abcdefghij", limit=4) == "a\n... (6 chars omitted) ...\nhij"


def test_render_is_newest_first_and_filters_by_kind():
    log = AgentLog(maxlen=2)
    log.add("LLM", "a", "r1", "s1", 0.1)
    log.add("HugeGraph", "b", "r2", "s2", 0.2)
    log.add("LLM", "c", "r3", "s3", 0.3)  # evicts the first entry
    assert [e.action for e in log.entries()] == ["b", "c"]
    text = log.render()
    assert text.index("| c |") < text.index("| b |")
    assert "| b |" not in log.render(["LLM"])
    assert log.render(["Embedding"]) == "(no activity yet)"


def test_install_wraps_real_clients_once():
    first = agent_log.install(AgentLog())
    assert "HGraphSession.request" in first or first == []  # empty when an earlier test already installed
    assert agent_log.install(AgentLog()) == []


def test_streaming_generators_are_logged_after_they_are_consumed():
    log = AgentLog()

    class Streamer:
        model = "m"

        def generate_streaming(self, messages=None, prompt=None, on_token_callback=None):
            yield "Hel"
            yield "lo"

    wrap_method(Streamer, "generate_streaming", agent_log.LLM, describe_llm, log=log)
    stream = Streamer().generate_streaming(prompt="hi")
    assert log.entries() == []  # nothing logged until the stream is read
    assert list(stream) == ["Hel", "lo"]
    (entry,) = log.entries()
    assert (entry.request, entry.response, entry.error) == ("hi", "Hello", "")
