import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime


class GameState:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path, timeout=30.0)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS state (
                key TEXT PRIMARY KEY,
                value TEXT
            );
            CREATE TABLE IF NOT EXISTS journal (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                text TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS discovered (
                category TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                PRIMARY KEY (category, entity_id)
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL
            );
        """)
        self.conn.commit()

    @contextmanager
    def _transaction(self):
        """Context manager for atomic multi-statement transactions."""
        try:
            yield self.conn
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def init(self, world_name: str, start_location: str):
        self._set("world_name", world_name)
        if not self._get("location"):
            self._set("location", start_location)
            self._add_visited(start_location)

    def _set(self, key: str, value: str):
        self.conn.execute(
            "INSERT OR REPLACE INTO state (key, value) VALUES (?, ?)",
            (key, value),
        )
        self.conn.commit()

    def _get(self, key: str) -> str | None:
        row = self.conn.execute(
            "SELECT value FROM state WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else None

    def get_location(self) -> str:
        return self._get("location") or ""

    def set_location(self, location_id: str):
        """Move to a new location -- atomic update of location + visited."""
        with self._transaction():
            self.conn.execute(
                "INSERT OR REPLACE INTO state (key, value) VALUES (?, ?)",
                ("location", location_id),
            )
            visited = self.get_visited()
            visited.add(location_id)
            self.conn.execute(
                "INSERT OR REPLACE INTO state (key, value) VALUES (?, ?)",
                ("visited", json.dumps(sorted(visited))),
            )

    def _add_visited(self, location_id: str):
        visited = self.get_visited()
        visited.add(location_id)
        self._set("visited", json.dumps(sorted(visited)))

    def get_visited(self) -> set[str]:
        raw = self._get("visited")
        if not raw:
            return set()
        return set(json.loads(raw))

    def add_journal_entry(self, text: str):
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.conn.execute(
            "INSERT INTO journal (timestamp, text) VALUES (?, ?)",
            (now, text),
        )
        self.conn.commit()

    def get_journal(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT timestamp, text FROM journal ORDER BY id"
        ).fetchall()
        return [{"time": r["timestamp"], "text": r["text"]} for r in rows]

    def discover(self, category: str, entity_id: str):
        now = datetime.now().isoformat()
        self.conn.execute(
            "INSERT OR IGNORE INTO discovered (category, entity_id, timestamp) VALUES (?, ?, ?)",
            (category, entity_id, now),
        )
        self.conn.commit()

    def get_discovered(self) -> dict[str, set[str]]:
        rows = self.conn.execute("SELECT category, entity_id FROM discovered").fetchall()
        result: dict[str, set[str]] = {
            "locations": set(),
            "npcs": set(),
            "events": set(),
            "factions": set(),
            "lore": set(),
        }
        for r in rows:
            cat = r["category"]
            if cat not in result:
                result[cat] = set()
            result[cat].add(r["entity_id"])
        return result

    def add_message(self, role: str, content: str):
        now = datetime.now().isoformat()
        self.conn.execute(
            "INSERT INTO messages (role, content, timestamp) VALUES (?, ?, ?)",
            (role, content, now),
        )
        self.conn.commit()

    def get_recent_messages(self, limit: int = 20) -> list[dict]:
        rows = self.conn.execute(
            "SELECT role, content FROM messages ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    def close(self):
        self.conn.close()
