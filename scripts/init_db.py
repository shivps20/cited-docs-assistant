"""Create the SQLite database (data/kb.db) and apply schema migrations.

Idempotent: safe to re-run. The schema itself lives in kb.core.db.MIGRATIONS.

    uv run python scripts/init_db.py
    uv run python scripts/init_db.py --reset     # deletes the database file!
"""

import argparse
import sys
from pathlib import Path

from kb.core.config import get_settings
from kb.core.db import SCHEMA_VERSION, connect, migrate


def main() -> int:
    """Create data/kb.db or apply pending migrations; with --reset, delete and recreate it first."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reset", action="store_true", help="delete the database file and recreate it")
    parser.add_argument("--yes", action="store_true", help="skip the --reset confirmation prompt")
    args = parser.parse_args()

    db_path = get_settings().db_path
    print(f"database {db_path}")
    if args.reset and db_path.exists():
        if not args.yes and input(f"Delete {db_path}? Type 'yes': ").strip() != "yes":
            print("aborted")
            return 1
        for suffix in ("", "-wal", "-shm"):
            Path(f"{db_path}{suffix}").unlink(missing_ok=True)
        print("deleted")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = connect(db_path, check_schema=False)
    try:
        applied = migrate(conn)
        for version in applied:
            print(f"applied migration v{version - 1} -> v{version}")
        if not applied:
            print(f"schema up to date (v{SCHEMA_VERSION})")
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )]
        print(f"ok: journal_mode={mode} tables={', '.join(tables)}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
