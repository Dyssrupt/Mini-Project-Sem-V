"""store.py -- SQLite incident store (prototype; the schema is one JSON document per incident, so it ports to Postgres JSONB / MongoDB)."""
import json, sqlite3, threading
from datetime import datetime, timezone


class Store:
    def __init__(self, path):
        self.db = sqlite3.connect(str(path), check_same_thread=False); self.lock = threading.Lock()
        self.db.execute("CREATE TABLE IF NOT EXISTS incidents(id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, status TEXT, source TEXT, data TEXT)")
        self.db.commit()

    def _row(self, r):
        return {"id": r[0], "created_at": r[1], "status": r[2], "source": r[3], **json.loads(r[4])}

    def create(self, source, data):
        with self.lock:
            cur = self.db.execute("INSERT INTO incidents(created_at,status,source,data) VALUES(?,?,?,?)",
                                  (datetime.now(timezone.utc).isoformat(), "new", source, json.dumps(data))); self.db.commit()
            return cur.lastrowid

    def get(self, iid):
        with self.lock:
            r = self.db.execute("SELECT id,created_at,status,source,data FROM incidents WHERE id=?", (iid,)).fetchone()
        return self._row(r) if r else None

    def list(self, limit=200):
        with self.lock:
            rs = self.db.execute("SELECT id,created_at,status,source,data FROM incidents ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [self._row(r) for r in rs]

    def update(self, iid, patch, status=None):
        with self.lock:
            r = self.db.execute("SELECT data FROM incidents WHERE id=?", (iid,)).fetchone()
            if not r: return None
            data = {**json.loads(r[0]), **patch}
            self.db.execute("UPDATE incidents SET data=?, status=COALESCE(?,status) WHERE id=?", (json.dumps(data), status, iid)); self.db.commit()
        return self.get(iid)
