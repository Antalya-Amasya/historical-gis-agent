from backend.app.rag.http_store import ChromaHttpEvidenceStore, canonical_e5_query
from backend.app.rag.retriever import ChromaHistoricalRetriever


class Embedding:
    def __init__(self): self.values=[]
    def embed(self, values): self.values.extend(values); return [[0.1, 0.2] for _ in values]


class Collection:
    def __init__(self): self.calls=[]
    def query(self, **kwargs):
        self.calls.append(kwargs)
        return {"ids":[["chunk-1"]], "documents":[["Caesar crossed the Rubicon."]], "metadatas":[[{"document_id":"caesar_gallic_civil_wars","author":"Julius Caesar","work":"Gallic War + Civil War","source_file":"caesar.epub","source_type":"primary_source","navigation_path_json":"[]"}]], "distances":[[0.25]]}


def test_e5_query_prefix_is_once_and_vectors_are_forwarded():
    embedding, collection = Embedding(), Collection()
    store = ChromaHttpEvidenceStore(collection, embedding)
    store.query("query: Caesar and Pompey", 3)
    assert embedding.values == ["query: Caesar and Pompey"]
    assert collection.calls[0]["query_embeddings"] == [[0.1, 0.2]]
    assert collection.calls[0]["n_results"] == 3


def test_http_result_maps_v2_optional_citation_fields_without_fabrication():
    evidence = ChromaHistoricalRetriever(ChromaHttpEvidenceStore(Collection(), Embedding())).retrieve("Caesar", 1)[0]
    assert evidence.metadata["document_id"] == "caesar_gallic_civil_wars"
    assert evidence.page_start is None and evidence.chapter is None
    assert evidence.metadata["distance"] == 0.25 and evidence.metadata["rank"] == 1


def test_canonical_query_prefix():
    assert canonical_e5_query("Caesar") == "query: Caesar"
    assert canonical_e5_query("query: Caesar") == "query: Caesar"
