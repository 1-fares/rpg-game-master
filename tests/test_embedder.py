from rpg_gm.ingestion.chunker import Chunk
from rpg_gm.ingestion.embedder import embed_chunks, query_chunks


SAMPLE_CHUNKS = [
    Chunk(index=0, text="Socrates taught philosophy in ancient Athens. He questioned everything.", page=1),
    Chunk(index=1, text="The Colosseum in Rome hosted gladiatorial combat for centuries.", page=2),
    Chunk(index=2, text="Plato founded the Academy in Athens, advancing Greek philosophy.", page=3),
    Chunk(index=3, text="Roman aqueducts supplied water to cities across the empire.", page=4),
]


def test_embed_chunks(tmp_path):
    collection = embed_chunks(SAMPLE_CHUNKS, "test-world", persist_dir=tmp_path)
    assert collection.count() == 4


def test_query_returns_relevant_results(tmp_path):
    embed_chunks(SAMPLE_CHUNKS, "test-world", persist_dir=tmp_path)
    results = query_chunks("philosophy in Athens", "test-world", persist_dir=tmp_path, n_results=4)
    top_ids = {r["id"] for r in results[:2]}
    assert "chunk-0" in top_ids or "chunk-2" in top_ids, (
        f"Expected Athens/philosophy chunks in top 2, got {top_ids}"
    )


def test_query_includes_metadata(tmp_path):
    embed_chunks(SAMPLE_CHUNKS, "test-world", persist_dir=tmp_path)
    results = query_chunks("gladiators", "test-world", persist_dir=tmp_path, n_results=1)
    assert len(results) == 1
    r = results[0]
    assert "page" in r
    assert "chunk_index" in r
    assert r["page"] is not None
    assert r["chunk_index"] is not None
