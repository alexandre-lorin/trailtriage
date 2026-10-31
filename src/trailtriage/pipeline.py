import fcntl
import logging
from pathlib import Path
from time import perf_counter

from PIL import Image
from tqdm import tqdm

from .config import Config
from .database import Database
from .metadata import extract
from .scanner import discover
from .utils import safe_target


def _scan(root: Path, output: Path, config: Config, metadata_only: bool = False) -> dict:
    begin = perf_counter()
    root, output = root.resolve(), output.resolve()
    if root == output or root.is_relative_to(output):
        raise ValueError("Output must be separate from, or nested inside, input")
    paths = discover(root, config, output)
    output.mkdir(parents=True, exist_ok=True)
    reports = output / "reports"
    safe_target(output, reports).mkdir(exist_ok=True)
    safe_target(output, reports / "run.log")
    safe_target(output, output / "index.sqlite")
    safe_target(output, output / "index.sqlite-wal")
    safe_target(output, output / "index.sqlite-shm")
    handler = logging.FileHandler(reports / "run.log")
    logger = logging.getLogger("trailtriage")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    db = Database(output / "index.sqlite")
    run_id = db.connection.execute(
        "INSERT INTO runs(status,input) VALUES('running',?)", (str(root),)
    ).lastrowid
    db.connection.commit()
    counts: dict = {"found": len(paths), "processed": 0, "skipped": 0, "errors": 0}
    metadata_start = perf_counter()
    try:
        for path in tqdm(paths, desc="Metadata", unit="image"):
            if db.unchanged(path):
                counts["skipped"] += 1
                row = db.get(path)
                assert row is not None
                counts["errors"] += bool(row["error"])
                continue
            try:
                metadata = extract(path)
                # Refuse a cache entry for a file changed during decoding.
                stat = path.stat()
                if (stat.st_size, stat.st_mtime_ns) != (
                    metadata["file_size"],
                    metadata["mtime_ns"],
                ):
                    raise OSError("File changed during processing; retry on next scan")
                db.save(path, metadata)
                counts["processed"] += 1
            except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
                try:
                    stat = path.stat()
                    failed: dict = {"file_size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
                except OSError:
                    failed = {}
                failed["filename"] = path.name
                db.save(path, failed, str(exc))
                logger.warning("%s: %s", path, exc)
                counts["errors"] += 1
        counts["metadata_seconds"] = round(perf_counter() - metadata_start, 3)
        if not metadata_only:
            from .analysis import analyze

            counts.update(analyze(paths, db, config, output))
        counts["total_seconds"] = round(perf_counter() - begin, 3)
        if not metadata_only:
            import json

            safe_target(output, reports / "summary.json").write_text(json.dumps(counts, indent=2))
        db.connection.execute(
            "UPDATE runs SET status='complete',finished=CURRENT_TIMESTAMP WHERE id=?", (run_id,)
        )
        db.connection.commit()
        logger.info("Summary: %s", counts)
        return counts
    except BaseException:
        db.connection.execute("UPDATE runs SET status='interrupted' WHERE id=?", (run_id,))
        db.connection.commit()
        raise
    finally:
        db.close()
        logger.removeHandler(handler)
        handler.close()


def scan(root: Path, output: Path, config: Config, metadata_only: bool = False) -> dict:
    root, output = root.resolve(), output.resolve()
    if root == output or root.is_relative_to(output):
        raise ValueError("Output must be separate from, or nested inside, input")
    if not root.is_dir():
        raise ValueError("Input directory does not exist")
    output.mkdir(parents=True, exist_ok=True)
    marker = safe_target(output, output / ".trailtriage")
    if not marker.exists() and any(output.iterdir()):
        raise ValueError("Choose an empty output directory; unrelated files must be preserved")
    if marker.exists() and marker.read_text() != str(root):
        raise ValueError("This output belongs to another input directory")
    with safe_target(output, output / ".scan.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another scan is running in this output") from exc
        marker.write_text(str(root))
        return _scan(root, output, config, metadata_only)
