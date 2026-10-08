"""Create the kb_chunks collection and its payload indexes in Qdrant.

Idempotent: safe to re-run. Creates the collection if missing, verifies the
vector config if it exists, and adds any payload indexes that are missing.
The collection definition itself lives in kb.store.vectorstore.

    uv run python scripts/init_qdrant.py
    uv run python scripts/init_qdrant.py --recreate     # drops all points!
"""

import argparse
import sys

from kb.core.config import get_settings
from kb.store.vectorstore import (
    create_collection,
    ensure_payload_indexes,
    get_client,
    vector_config_problems,
)

QDRANT_URL = get_settings().qdrant_url
COLLECTION = get_settings().qdrant_collection


def main() -> int:
    """Create the collection (or check an existing one) and ensure its payload indexes; --recreate drops it
    first.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recreate", action="store_true", help="delete and recreate the collection (drops all points)")
    parser.add_argument("--yes", action="store_true", help="skip the --recreate confirmation prompt")
    args = parser.parse_args()

    client = get_client()
    print(f"Qdrant {QDRANT_URL} | collection '{COLLECTION}'")

    exists = client.collection_exists(COLLECTION)
    if exists and args.recreate:
        points = client.count(COLLECTION, exact=True).count
        if not args.yes and input(f"Delete '{COLLECTION}' with {points} points? Type 'yes': ").strip() != "yes":
            print("aborted")
            return 1
        client.delete_collection(COLLECTION)
        print(f"deleted collection '{COLLECTION}' ({points} points)")
        exists = False

    if exists:
        print(f"collection '{COLLECTION}' already exists")
        problems = vector_config_problems(client, COLLECTION)
        for p in problems:
            print(f"  MISMATCH: {p}")
        if problems:
            print("vector config differs from expected; re-run with --recreate to fix (drops all points)")
            return 1
    else:
        create_collection(client, COLLECTION)
        print(f"created collection '{COLLECTION}'")

    for field, state in ensure_payload_indexes(client, COLLECTION).items():
        marker = {"created": "+", "ok": "="}.get(state, "!")
        print(f"  {marker} index {field} ({state})")

    info = client.get_collection(COLLECTION)
    print(f"ok: status={info.status.value} points={info.points_count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
