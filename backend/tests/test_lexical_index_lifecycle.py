"""Lifecycle tests for the process-local, read-only passage lexical index."""
from __future__ import annotations

from threading import Event, Thread

from backend.app.rag.http_store import (
    ChromaHttpEvidenceStore,
    clear_process_lexical_index_cache,
)


class NoEmbedding:
    def embed(self, _texts):  # pragma: no cover - must not be called here
        raise AssertionError("lexical index construction must not invoke embeddings")


class Corpus:
    def __init__(self, corpus_id: str = "roman-republic-v2", *, fail: bool = False):
        self.id = corpus_id
        self.fail = fail
        self.get_calls = 0

    def get(self, **_kwargs):
        self.get_calls += 1
        if self.fail:
            raise RuntimeError("simulated corpus read failure")
        return {
            "ids": ["a", "b"],
            "documents": [
                "Hannibal crossed the Alps and entered Italy with his army.",
                "At Cannae the Roman army suffered a severe defeat.",
            ],
            "metadatas": [
                {"document_id": "polybius", "author": "Polybius", "work": "Histories"},
                {"document_id": "livy", "author": "Livy", "work": "History of Rome"},
            ],
        }


def store(corpus: Corpus) -> ChromaHttpEvidenceStore:
    return ChromaHttpEvidenceStore(corpus, NoEmbedding())


def test_first_access_builds_once_and_second_store_reuses_same_process_index():
    clear_process_lexical_index_cache()
    try:
        first_corpus = Corpus()
        first = store(first_corpus).lexical_candidates("Hannibal Alps", 5)
        second_corpus = Corpus()
        second = store(second_corpus).lexical_candidates("Hannibal Alps", 5)

        assert first_corpus.get_calls == 1
        assert second_corpus.get_calls == 0
        assert [item.id for item in first] == [item.id for item in second]
        assert all(item.metadata["source_chunk_id"] == "a" for item in first)
    finally:
        clear_process_lexical_index_cache()


def test_failed_initialization_is_not_cached_and_a_later_attempt_can_build():
    clear_process_lexical_index_cache()
    try:
        failed = Corpus(fail=True)
        try:
            store(failed).lexical_candidates("Hannibal", 5)
        except RuntimeError as exc:
            assert str(exc) == "simulated corpus read failure"
        else:  # pragma: no cover
            raise AssertionError("corpus read failure must propagate")

        healthy = Corpus()
        result = store(healthy).lexical_candidates("Hannibal", 5)
        assert failed.get_calls == 1
        assert healthy.get_calls == 1
        assert result
    finally:
        clear_process_lexical_index_cache()


def test_cache_is_process_local_read_only_and_can_be_explicitly_reinitialized():
    clear_process_lexical_index_cache()
    try:
        corpus = Corpus()
        evidence_store = store(corpus)
        evidence_store.lexical_candidates("Cannae", 5)
        evidence_store.lexical_candidates("Cannae", 5)
        assert corpus.get_calls == 1

        clear_process_lexical_index_cache()
        fresh_store = store(corpus)
        fresh_store.lexical_candidates("Cannae", 5)
        assert corpus.get_calls == 2
    finally:
        clear_process_lexical_index_cache()


def test_concurrent_first_access_is_bounded_to_one_corpus_build():
    class BlockingCorpus(Corpus):
        def __init__(self):
            super().__init__("concurrent-roman-republic-v2")
            self.build_started = Event()
            self.release_build = Event()

        def get(self, **kwargs):
            self.build_started.set()
            assert self.release_build.wait(timeout=1)
            return super().get(**kwargs)

    clear_process_lexical_index_cache()
    try:
        corpus = BlockingCorpus()
        results: list[list] = []
        errors: list[BaseException] = []

        def retrieve() -> None:
            try:
                results.append(store(corpus).lexical_candidates("Hannibal", 5))
            except BaseException as exc:  # pragma: no cover - assertion below reports it
                errors.append(exc)

        first = Thread(target=retrieve)
        second = Thread(target=retrieve)
        first.start()
        assert corpus.build_started.wait(timeout=1)
        second.start()
        corpus.release_build.set()
        first.join(timeout=1)
        second.join(timeout=1)

        assert not errors
        assert len(results) == 2
        assert corpus.get_calls == 1
    finally:
        clear_process_lexical_index_cache()
