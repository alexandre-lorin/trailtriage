import json
import sqlite3
from pathlib import Path


class Database:
    def __init__(self, path: Path):
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS images (path TEXT PRIMARY KEY, size INTEGER, "
            "mtime_ns INTEGER, metadata TEXT NOT NULL, error TEXT, "
            "analysis_key TEXT, analysis TEXT, label TEXT)"
        )
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS runs (id INTEGER PRIMARY KEY, started TEXT "
            "DEFAULT CURRENT_TIMESTAMP, finished TEXT, status TEXT NOT NULL, input TEXT)"
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def get(self, path: Path) -> dict | None:
        row = self.connection.execute(
            "SELECT size,mtime_ns,metadata,error,analysis_key,analysis,label FROM images "
            "WHERE path=?",
            (str(path),),
        ).fetchone()
        if not row:
            return None
        return dict(
            zip(["size", "mtime_ns", "metadata", "error", "analysis_key", "analysis", "label"], row)
        )

    def unchanged(self, path: Path) -> bool:
        row = self.get(path)
        try:
            stat = path.stat()
        except OSError:
            return False
        return bool(row and row["size"] == stat.st_size and row["mtime_ns"] == stat.st_mtime_ns)

    def save(self, path: Path, metadata: dict, error: str | None = None) -> None:
        self.connection.execute(
            "INSERT INTO images(path,size,mtime_ns,metadata,error) VALUES(?,?,?,?,?) "
            "ON CONFLICT(path) DO UPDATE SET size=excluded.size,mtime_ns=excluded.mtime_ns,"
            "metadata=excluded.metadata,error=excluded.error,analysis_key=NULL,analysis=NULL",
            (
                str(path),
                metadata.get("file_size"),
                metadata.get("mtime_ns"),
                json.dumps(metadata),
                error,
            ),
        )
        self.connection.commit()

    def save_analysis(self, path: Path, key: str, analysis: dict) -> None:
        self.connection.execute(
            "UPDATE images SET analysis_key=?,analysis=? WHERE path=?",
            (key, json.dumps(analysis), str(path)),
        )
        self.connection.commit()

    def label(self, path: str, label: str) -> None:
        with self.connection:
            self.connection.execute("UPDATE images SET label=? WHERE path=?", (label, path))

    def rows(self) -> list[dict]:
        result = []
        for path, metadata, error, analysis, label in self.connection.execute(
            "SELECT path,metadata,error,analysis,label FROM images ORDER BY path"
        ):
            result.append(
                {
                    **json.loads(metadata),
                    **(json.loads(analysis) if analysis else {}),
                    "original_path": path,
                    "error": error,
                    "label": label,
                }
            )
        return result
