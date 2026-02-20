from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from rpg_gm.ingestion.chunker import Chunk

DEFAULT_MODEL = "all-MiniLM-L6-v2"


def _get_client(persist_dir: Path | None = None) -> chromadb.ClientAPI:
    if persist_dir:
        return chromadb.PersistentClient(path=str(persist_dir))
    return chromadb.PersistentClient(path="worlds/_chroma")


def _get_embedding_fn() -> SentenceTransformerEmbeddingFunction:
    return SentenceTransformerEmbeddingFunction(model_name=DEFAULT_MODEL)


def embed_chunks(
    chunks: list[Chunk],
    world_name: str,
    persist_dir: Path | None = None,
) -> chromadb.Collection:
    """Embed chunks into a ChromaDB collection."""
    client = _get_client(persist_dir)
    ef = _get_embedding_fn()
    collection = client.get_or_create_collection(
        name=world_name.replace(" ", "-"),
        embedding_function=ef,
    )
    ids = [f"chunk-{c.index}" for c in chunks]
    documents = [c.text for c in chunks]
    metadatas = [
        {"chunk_index": c.index, "page": c.page or 0, "preview": c.text[:100]}
        for c in chunks
    ]
    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    return collection


def query_chunks(
    query: str,
    world_name: str,
    persist_dir: Path | None = None,
    n_results: int = 5,
) -> list[dict]:
    """Query ChromaDB and return relevant chunks."""
    client = _get_client(persist_dir)
    ef = _get_embedding_fn()
    collection = client.get_collection(
        name=world_name.replace(" ", "-"),
        embedding_function=ef,
    )
    results = collection.query(query_texts=[query], n_results=n_results)
    output = []
    for i in range(len(results["ids"][0])):
        output.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "page": results["metadatas"][0][i].get("page"),
            "chunk_index": results["metadatas"][0][i].get("chunk_index"),
            "distance": results["distances"][0][i] if results.get("distances") else None,
        })
    return output
