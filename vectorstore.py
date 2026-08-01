"""
vectorstore.py
Qdrant collections: one for the churn report chunks (RAG), one for
past-turn conversation memory. Note: v1.18.0 renamed .search() to
.query_points(), with results accessed via .results.points -> actually
.points on the returned object (see query_points calls below).

Embeddings come from the proxy's gemini-embedding model (via
config.get_embedder()) rather than a local SentenceTransformer — call
init_embedder() once before anything else in this module is used.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

DOCS_COLLECTION = "churn_docs"
MEMORY_COLLECTION = "past_turns"

qdrant = QdrantClient(":memory:")
embedder = None  # set by init_embedder()
_memory_counter = 0


def init_embedder():
    """Must be called once (after the Gemini key is loaded) before
    index_report / add_to_memory / get_relevant_memory are used."""
    global embedder
    from config import get_embedder
    embedder = get_embedder()
    return embedder


def index_report(report_text: str, chunk_size: int = 500, chunk_overlap: int = 100):
    """Splits the churn report into chunks and stores embeddings in Qdrant."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(report_text)

    vectors = embedder.embed_documents(chunks)  # list[list[float]]
    dim = len(vectors[0])

    qdrant.recreate_collection(
        collection_name=DOCS_COLLECTION,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    points = [PointStruct(id=i, vector=vectors[i], payload={"text": chunks[i]}) for i in range(len(chunks))]
    qdrant.upsert(collection_name=DOCS_COLLECTION, points=points)

    print("Stored", len(points), "chunks in Qdrant")
    return dim  # embedding dim, needed to size the memory collection


def init_memory_collection(embedding_dim: int):
    qdrant.recreate_collection(
        collection_name=MEMORY_COLLECTION,
        vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
    )


def add_to_memory(question: str, answer: str):
    global _memory_counter
    text = f"Q: {question}\nA: {answer}"
    vec = embedder.embed_query(text)
    qdrant.upsert(
        collection_name=MEMORY_COLLECTION,
        points=[PointStruct(id=_memory_counter, vector=vec, payload={"text": text})],
    )
    _memory_counter += 1


def get_relevant_memory(question: str, k: int = 3) -> list[str]:
    vec = embedder.embed_query(question)
    try:
        results = qdrant.query_points(collection_name=MEMORY_COLLECTION, query=vec, limit=k)
        return [r.payload["text"] for r in results.points]
    except Exception:
        return []  # no memory yet
