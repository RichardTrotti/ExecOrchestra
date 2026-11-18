"""Check streaming boundaries and failure behavior without live services."""

from collections import Counter
from pathlib import Path
from urllib.error import HTTPError, URLError

import numpy as np
import pytest


@pytest.fixture
def fake_retrieval(monkeypatch):
    template = Path(__file__).resolve().parents[1] / "configs" / "retrieval_policy.example.json"
    monkeypatch.setenv("SGAR_RETRIEVAL_POLICY_PATH", str(template))
    monkeypatch.setenv("SGAR_SEALED_LOCAL_VALIDATION", "1")
    import retrieve

    generated = []
    embedded = []

    def generate(query, *, cost_ledger):
        generated.append((query, cost_ledger))
        return {
            "capability_text": f"cap: {query}",
            "constraint_text": f"con: {query}",
            "metadata": {"status": "generated"},
        }

    def encode(texts, *, role):
        embedded.append((role, tuple(texts)))
        return np.array([(len(text), sum(map(ord, text))) for text in texts], dtype="float32").reshape(-1, 2)

    monkeypatch.setattr(retrieve._engine, "ensure_loaded", lambda: None)
    monkeypatch.setattr(retrieve, "_generate_hyde_result", generate)
    monkeypatch.setattr(retrieve, "_encode_many", encode)
    return retrieve, generated, embedded


@pytest.mark.parametrize("use_hyde", [False, True])
def test_batches_are_lazy_bounded_and_match_legacy(fake_retrieval, use_hyde):
    retrieve, generated, embedded = fake_retrieval
    queries = ["Return JSON", "Deliver code", "repeat", "repeat", "Last"]
    ledger = object()
    legacy = retrieve.encode_query_profiles(queries, use_hyde, cost_ledger=ledger)
    assert isinstance(legacy, list)
    legacy_encoding = embedded[:]
    generated.clear()
    embedded.clear()
    consumed = []

    def source():
        for query in queries:
            consumed.append(query)
            yield query

    stream = retrieve.iter_query_profile_batches(
        source(), chunk_size=2, use_hyde=use_hyde, cost_ledger=ledger,
    )
    assert consumed == generated == embedded == []
    first = next(stream)
    assert (first.start_index, first.stop_index) == (0, 2)
    assert consumed == queries[:2]
    if use_hyde:
        assert Counter(query for query, _ in generated) == Counter(queries[:2])
    assert all(len(texts) == 2 for _, texts in embedded)

    rest = list(stream)
    assert [(b.start_index, b.stop_index) for b in rest] == [(2, 4), (4, 5)]
    assert [p for b in (first, *rest) for p in b.profiles] == legacy
    assert all(1 <= len(texts) <= 2 for _, texts in embedded)
    assert {role for role, _ in embedded} == {role for role, _ in legacy_encoding}
    expected = Counter(queries) if use_hyde else Counter()
    assert Counter(query for query, _ in generated) == expected
    assert all(seen_ledger is ledger for _, seen_ledger in generated)


@pytest.mark.parametrize("chunk_size", [0, -1, True, 1.5])
def test_invalid_chunk_size_does_not_consume_input(fake_retrieval, chunk_size):
    retrieve, generated, embedded = fake_retrieval

    def source():
        pytest.fail("invalid batch size consumed the input")
        yield "unreachable"

    with pytest.raises(ValueError, match="positive integer"):
        next(retrieve.iter_query_profile_batches(source(), chunk_size=chunk_size))
    assert not generated and not embedded


def test_empty_input_does_not_call_either_service(fake_retrieval):
    retrieve, generated, embedded = fake_retrieval
    assert list(retrieve.iter_query_profile_batches(iter(()), chunk_size=3)) == []
    assert not generated and not embedded


def test_opt_in_embedding_retry_reuses_generated_texts(fake_retrieval, monkeypatch):
    retrieve, generated, embedded = fake_retrieval
    original = retrieve._encode_many
    failed = False

    def encode(texts, *, role):
        nonlocal failed
        vectors = original(texts, role=role)
        if role == "constraint" and not failed:
            failed = True
            raise retrieve.RetrievalLifecycleError("embedding_api_request_failed") from URLError(TimeoutError())
        return vectors

    monkeypatch.setattr(retrieve, "_encode_many", encode)
    batch, = retrieve.iter_query_profile_batches(
        iter(("first", "second")), chunk_size=2, retry_embedding=True,
    )
    assert [p.raw_query_text for p in batch.profiles] == ["first", "second"]
    assert Counter(query for query, _ in generated) == {"first": 1, "second": 1}
    assert [role for role, _ in embedded] == ["raw", "capability", "constraint"] * 2
    assert embedded[:3] == embedded[3:]


@pytest.mark.parametrize(("retry_embedding", "kind", "attempts"), [
    (False, "timeout", 1),
    (True, "timeout", 2),
    (True, "http", 1),
    (True, "response", 1),
])
def test_embedding_failure_stops_without_consuming_another_batch(
    fake_retrieval, monkeypatch, retry_embedding, kind, attempts,
):
    retrieve, generated, embedded = fake_retrieval
    original = retrieve._encode_many

    def encode(texts, *, role):
        vectors = original(texts, role=role)
        if list(texts) == ["fail"]:
            if kind == "response":
                raise retrieve.RetrievalLifecycleError("embedding_api_response_invalid")
            cause = (
                HTTPError("https://example.invalid", 401, "Unauthorized", None, None)
                if kind == "http" else TimeoutError()
            )
            raise retrieve.RetrievalLifecycleError("embedding_api_request_failed") from cause
        return vectors

    monkeypatch.setattr(retrieve, "_encode_many", encode)
    stream = retrieve.iter_query_profile_batches(
        iter(("ok", "fail", "untouched")), chunk_size=1, retry_embedding=retry_embedding,
    )
    assert [p.raw_query_text for p in next(stream).profiles] == ["ok"]
    with pytest.raises(retrieve.QueryProfileBatchError) as failure:
        next(stream)
    error = failure.value
    assert (error.start_index, error.stop_index, error.stage) == (1, 2, "embedding")
    assert isinstance(error.__cause__, retrieve.RetrievalLifecycleError)
    assert [query for query, _ in generated] == ["ok", "fail"]
    assert embedded.count(("raw", ("fail",))) == attempts
    assert list(stream) == []


def test_generation_failure_stops_without_retrying_or_encoding(
    fake_retrieval, monkeypatch,
):
    retrieve, generated, embedded = fake_retrieval
    terminal = retrieve.HyDEGenerationError(
        "budget_exhausted", responsibility="budget", response_received=False,
    )
    seen = []

    def generate(query, *, cost_ledger):
        seen.append(query)
        raise terminal

    monkeypatch.setattr(retrieve, "_generate_hyde_result", generate)
    stream = retrieve.iter_query_profile_batches(
        iter(("fail", "untouched")), chunk_size=1, retry_embedding=True,
    )
    with pytest.raises(retrieve.QueryProfileBatchError) as failure:
        next(stream)
    error = failure.value
    assert (error.start_index, error.stop_index, error.stage) == (0, 1, "generation")
    assert error.__cause__ is terminal
    assert seen == ["fail"]
    assert not generated and not embedded


def test_legacy_interface_preserves_original_failure(fake_retrieval, monkeypatch):
    retrieve, _, _ = fake_retrieval
    terminal = retrieve.RetrievalLifecycleError("embedding_api_response_invalid")

    def encode(texts, *, role):
        raise terminal

    monkeypatch.setattr(retrieve, "_encode_many", encode)
    with pytest.raises(retrieve.RetrievalLifecycleError) as failure:
        retrieve.encode_query_profiles(["query"], use_hyde=False)
    assert failure.value is terminal
