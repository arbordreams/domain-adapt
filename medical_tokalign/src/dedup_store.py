from __future__ import annotations

import os
import sqlite3
from collections import deque
from typing import Deque, Optional, Set


class SeenStore:
    """SQLite-backed exact dedup store.

    Stores content digests as TEXT primary keys. Thread-safe enough for our single-process builder.
    """

    def __init__(self, db_path: str, recent_cache_size: int = 200_000) -> None:
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute("PRAGMA temp_store=MEMORY;")
        self._conn.execute("PRAGMA mmap_size=268435456;")
        self._conn.execute("CREATE TABLE IF NOT EXISTS seen (h TEXT PRIMARY KEY);")
        self._conn.commit()
        self._recent_q: Deque[str] = deque(maxlen=int(recent_cache_size))
        self._recent_set: Set[str] = set()

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    def _recent_add(self, h: str) -> None:
        if h in self._recent_set:
            return
        if len(self._recent_q) == self._recent_q.maxlen:
            old = self._recent_q.popleft()
            self._recent_set.discard(old)
        self._recent_q.append(h)
        self._recent_set.add(h)

    def exists(self, h: str) -> bool:
        if h in self._recent_set:
            return True
        cur = self._conn.execute("SELECT 1 FROM seen WHERE h=? LIMIT 1;", (h,))
        row = cur.fetchone()
        if row:
            self._recent_add(h)
            return True
        return False

    def add(self, h: str) -> None:
        try:
            self._conn.execute("INSERT OR IGNORE INTO seen(h) VALUES (?);", (h,))
            self._conn.commit()
            self._recent_add(h)
        except Exception:
            # Best-effort: ignore transient failures
            pass

    def backfill_from_jsonl(self, jsonl_path: str, hash_fn) -> int:
        """Populate store from an existing corpus JSONL file. Returns number of inserted rows."""
        inserted = 0
        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                batch = []
                for line in f:
                    try:
                        import json as _json
                        rec = _json.loads(line)
                        t = str(rec.get("text", "")).strip()
                        if not t:
                            continue
                        h = hash_fn(t)
                        batch.append((h,))
                        if len(batch) >= 1000:
                            cur = self._conn.executemany("INSERT OR IGNORE INTO seen(h) VALUES (?);", batch)
                            self._conn.commit()
                            inserted += cur.rowcount or 0
                            batch.clear()
                    except Exception:
                        continue
                if batch:
                    cur = self._conn.executemany("INSERT OR IGNORE INTO seen(h) VALUES (?);", batch)
                    self._conn.commit()
                    inserted += cur.rowcount or 0
        except Exception:
            pass
        return int(inserted)


