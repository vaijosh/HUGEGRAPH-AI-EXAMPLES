"""Agent log: an in-memory record of the app's background calls to the LLM, embedding model and HugeGraph.

install() wraps the hugegraph-llm / pyhugegraph client methods so every request and response (or error)
is recorded, whichever flow makes the call. Entries live only in this process and are cleared on restart.
"""

import functools
import inspect
import itertools
import json
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional, Tuple

LLM, EMBEDDING, GRAPH = "LLM", "Embedding", "HugeGraph"
KINDS = (LLM, EMBEDDING, GRAPH)
MAX_ENTRIES = 300
MAX_TEXT = 4000  # per request/response; extraction prompts with instructions, schema and chunk run past 6000 chars


@dataclass
class Entry:
    seq: int
    at: float
    kind: str
    action: str
    request: str
    response: str
    seconds: float
    error: str = ""


def clip(value: Any, limit: int = MAX_TEXT) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        try:
            value = json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            value = repr(value)
    if len(value) <= limit:
        return value
    # Keep both ends: an LLM prompt opens with fixed instructions and ends with the document chunk and schema.
    head = limit // 4
    tail = limit - head
    return f"{value[:head]}\n... ({len(value) - limit} chars omitted) ...\n{value[-tail:]}"


class AgentLog:
    def __init__(self, maxlen: int = MAX_ENTRIES):
        self._entries: deque = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._seq = itertools.count(1)
        self.version = 0  # bumps on every change so the UI can skip re-rendering an unchanged log

    def add(self, kind: str, action: str, request: Any, response: Any, seconds: float, error: str = "") -> None:
        with self._lock:
            self._entries.append(
                Entry(next(self._seq), time.time(), kind, action, clip(request), clip(response), seconds, error)
            )
            self.version += 1

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self.version += 1

    def entries(self) -> List[Entry]:
        with self._lock:
            return list(self._entries)

    def render(self, kinds: Optional[Iterable[str]] = None, limit: int = 50) -> str:
        """Newest entries first, as plain text."""
        wanted = set(kinds) if kinds is not None else set(KINDS)
        shown = [e for e in reversed(self.entries()) if e.kind in wanted][:limit]
        if not shown:
            return "(no activity yet)"
        blocks = []
        for e in shown:
            stamp = time.strftime("%H:%M:%S", time.localtime(e.at)) + f".{int(e.at % 1 * 1000):03d}"
            status = f"FAILED: {e.error}" if e.error else "ok"
            blocks.append(
                f"#{e.seq} [{stamp}] {e.kind} | {e.action} | {e.seconds:.2f}s | {status}\n"
                f">>> request\n{e.request}\n<<< response\n{e.response or '(none)'}"
            )
        return ("\n\n" + "-" * 80 + "\n\n").join(blocks)


LOG = AgentLog()
_local = threading.local()  # suppresses nested entries, e.g. get_texts_embeddings calling get_text_embedding


def _record(log: AgentLog, kind: str, action: str, request: Any, call: Callable[[], Any], describe_response):
    depth = getattr(_local, "depth", 0)
    if depth:
        return call()
    _local.depth = depth + 1
    start = time.perf_counter()
    try:
        result = call()
    except Exception as exc:
        log.add(kind, action, request, None, time.perf_counter() - start, error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        _local.depth = depth
    if inspect.isgenerator(result):  # streaming call: log once the caller has consumed it
        return _logged_stream(log, kind, action, request, result, start)
    log.add(kind, action, request, describe_response(result), time.perf_counter() - start)
    return result


def _logged_stream(log: AgentLog, kind: str, action: str, request: Any, stream, start: float):
    pieces = []
    try:
        for piece in stream:
            if isinstance(piece, str):
                pieces.append(piece)
            yield piece
    except Exception as exc:
        log.add(kind, action, request, "".join(pieces), time.perf_counter() - start, error=f"{type(exc).__name__}: {exc}")
        raise
    log.add(kind, action, request, "".join(pieces), time.perf_counter() - start)


def wrap_method(
    cls: type,
    name: str,
    kind: str,
    describe: Callable[..., Tuple[str, Any]],
    describe_response: Callable[[Any], Any] = lambda result: result,
    log: AgentLog = LOG,
) -> bool:
    """Replace cls.name with a recording wrapper. describe(self, *args, **kwargs) -> (action, request)."""
    original = cls.__dict__.get(name)
    if original is None or getattr(original, "_agent_log_wrapped", False):
        return False

    if inspect.iscoroutinefunction(original):

        @functools.wraps(original)
        async def wrapper(self, *args, **kwargs):
            action, request = describe(self, *args, **kwargs)
            start = time.perf_counter()
            try:
                result = await original(self, *args, **kwargs)
            except Exception as exc:
                log.add(kind, action, request, None, time.perf_counter() - start, error=f"{type(exc).__name__}: {exc}")
                raise
            log.add(kind, action, request, describe_response(result), time.perf_counter() - start)
            return result

    else:

        @functools.wraps(original)
        def wrapper(self, *args, **kwargs):
            action, request = describe(self, *args, **kwargs)
            return _record(log, kind, action, request, lambda: original(self, *args, **kwargs), describe_response)

    wrapper._agent_log_wrapped = True  # type: ignore[attr-defined]
    setattr(cls, name, wrapper)
    return True


# ---------------------------------------------------------------------------
# What each call looks like in the log
# ---------------------------------------------------------------------------
def describe_llm(self, messages=None, prompt=None, *_, **__) -> Tuple[str, str]:
    model = getattr(self, "model", None) or getattr(self, "model_name", "?")
    if messages:
        text = "\n\n".join(f"[{m.get('role', '?')}] {m.get('content', '')}" for m in messages)
    else:
        text = prompt or ""
    return f"{type(self).__name__} {model}", text


def describe_embedding(self, texts=None, *_, **__) -> Tuple[str, str]:
    model = getattr(self, "model", None) or getattr(self, "model_name", "?")
    texts = [texts] if isinstance(texts, str) else list(texts or [])
    preview = "\n".join(f"- {t[:200]}" for t in texts[:5])
    more = f"\n... and {len(texts) - 5} more" if len(texts) > 5 else ""
    return f"{type(self).__name__} {model}", f"{len(texts)} text(s):\n{preview}{more}"


def describe_vectors(result: Any) -> str:
    if result and isinstance(result[0], (list, tuple)):
        return f"{len(result)} vector(s) of dimension {len(result[0])}"
    return f"1 vector of dimension {len(result or [])}"


def describe_graph(self, path, method="GET", *_, **kwargs) -> Tuple[str, Any]:
    body = kwargs.get("json", kwargs.get("data"))
    if isinstance(body, (bytes, bytearray)):
        body = body.decode("utf-8", "replace")
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except ValueError:
            pass
    # Gremlin requests are easiest to read as the query itself
    if isinstance(body, dict) and "gremlin" in body:
        body = body["gremlin"]
    return f"{method.upper()} {path}", body if body is not None else ""


def install(log: AgentLog = LOG) -> List[str]:
    """Wrap the LLM, embedding and HugeGraph clients; safe to call more than once."""
    # pylint: disable=import-outside-toplevel
    from hugegraph_llm.models.embeddings.litellm import LiteLLMEmbedding
    from hugegraph_llm.models.embeddings.ollama import OllamaEmbedding
    from hugegraph_llm.models.embeddings.openai import OpenAIEmbedding
    from hugegraph_llm.models.llms.litellm import LiteLLMClient
    from hugegraph_llm.models.llms.ollama import OllamaClient
    from hugegraph_llm.models.llms.openai import OpenAIClient
    from pyhugegraph.utils.huge_requests import HGraphSession

    wrapped = []
    for cls in (OllamaClient, LiteLLMClient, OpenAIClient):
        for name in ("generate", "agenerate", "generate_streaming"):
            if wrap_method(cls, name, LLM, describe_llm, log=log):
                wrapped.append(f"{cls.__name__}.{name}")
    for cls in (OllamaEmbedding, LiteLLMEmbedding, OpenAIEmbedding):
        for name in ("get_text_embedding", "get_texts_embeddings"):
            if wrap_method(cls, name, EMBEDDING, describe_embedding, describe_vectors, log=log):
                wrapped.append(f"{cls.__name__}.{name}")
    if wrap_method(HGraphSession, "request", GRAPH, describe_graph, log=log):
        wrapped.append("HGraphSession.request")
    return wrapped
