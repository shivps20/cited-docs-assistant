"""Qdrant collection definition and helpers shared by the init script, indexer and retrieval."""

from qdrant_client import QdrantClient, models

from kb.core.config import get_settings

DENSE = "dense"    # bge-m3 dense embedding
SPARSE = "sparse"  # bge-m3 learned lexical weights
DENSE_DIM = 1024

# Fields used in query filters. Without an index, filtering scans every point.
PAYLOAD_INDEXES: dict[str, models.PayloadSchemaType] = {
    "doc_id": models.PayloadSchemaType.KEYWORD,           # re-ingest deletes, per-doc lookups
    "section_id": models.PayloadSchemaType.KEYWORD,       # parent-section expansion
    "chunk_index": models.PayloadSchemaType.INTEGER,      # child +/-1 neighbour window
    "allowed_groups": models.PayloadSchemaType.KEYWORD,   # ACL filter (list of groups)
    "release_min": models.PayloadSchemaType.INTEGER,      # first applicable release year, R2015x -> 2015
    "release_max": models.PayloadSchemaType.INTEGER,      # last applicable release year (9999 = open-ended)
    "is_latest": models.PayloadSchemaType.BOOL,           # latest-revision filter
    "external_ok": models.PayloadSchemaType.BOOL,         # may chunks go to OpenAI
    "category": models.PayloadSchemaType.KEYWORD,         # agentic search_kb(category=)
    "doc_type": models.PayloadSchemaType.KEYWORD,         # pdf / pptx / docx
}


def get_client() -> QdrantClient:
    """Qdrant client for QDRANT_URL."""
    return QdrantClient(url=get_settings().qdrant_url)


def create_collection(client: QdrantClient, name: str) -> None:
    """Create the collection with the named dense (cosine) and sparse vectors."""
    client.create_collection(
        collection_name=name,
        vectors_config={DENSE: models.VectorParams(size=DENSE_DIM, distance=models.Distance.COSINE)},
        # No IDF modifier: bge-m3 sparse weights are already learned term importances.
        sparse_vectors_config={SPARSE: models.SparseVectorParams(index=models.SparseIndexParams(on_disk=False))},
    )


def vector_config_problems(client: QdrantClient, name: str) -> list[str]:
    """Differences between an existing collection's vectors and what the code expects (empty = OK)."""
    params = client.get_collection(name).config.params
    dense = params.vectors.get(DENSE) if isinstance(params.vectors, dict) else None
    sparse = (params.sparse_vectors or {}).get(SPARSE)
    problems = []
    if dense is None:
        problems.append(f"missing dense vector '{DENSE}'")
    else:
        if dense.size != DENSE_DIM:
            problems.append(f"dense size {dense.size} != {DENSE_DIM}")
        if dense.distance != models.Distance.COSINE:
            problems.append(f"dense distance {dense.distance} != Cosine")
    if sparse is None:
        problems.append(f"missing sparse vector '{SPARSE}'")
    return problems


def ensure_payload_indexes(client: QdrantClient, name: str) -> dict[str, str]:
    """Create missing payload indexes. Returns field -> 'created' | 'ok' | 'mismatch: <type>'."""
    existing = client.get_collection(name).payload_schema
    result = {}
    for field, schema in PAYLOAD_INDEXES.items():
        current = existing.get(field)
        if current is None:
            client.create_payload_index(name, field_name=field, field_schema=schema, wait=True)
            result[field] = "created"
        elif current.data_type.value != schema.value:
            result[field] = f"mismatch: {current.data_type.value}"
        else:
            result[field] = "ok"
    return result


def doc_filter(doc_id: str) -> models.Filter:
    """Filter matching every point of one document (used to delete or count its chunks)."""
    return models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))])
