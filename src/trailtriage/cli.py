import argparse
import json
import sqlite3
from pathlib import Path

from .config import load_config
from .database import Database
from .pipeline import scan
from .scanner import discover


def main() -> None:
    parser = argparse.ArgumentParser(description="Local, safe trail-camera photo triage")
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("scan")
    command.add_argument("input", type=Path)
    command.add_argument("--output", type=Path, default=Path("trailtriage-output"))
    command.add_argument("--config", type=Path)
    command.add_argument("--device")
    command.add_argument("--model")
    command.add_argument("--batch-size", type=int)
    command.add_argument("--no-dedup", action="store_true")
    command.add_argument("--metadata-only", action="store_true")
    command.add_argument("--dry-run", action="store_true")
    for name in ("stats", "gallery", "serve"):
        command = sub.add_parser(name)
        command.add_argument("output", type=Path)
        if name == "serve":
            command.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    try:
        if args.command == "scan":
            config = load_config(args.config)
            for key in ("device", "model", "batch_size"):
                value = getattr(args, key)
                if value is not None:
                    setattr(config.detection, key, value)
            if args.no_dedup:
                config.deduplication.enabled = False
            config = type(config).model_validate(config.model_dump())
            if args.dry_run:
                result = {
                    "found": len(discover(args.input.resolve(), config, args.output.resolve()))
                }
            else:
                result = scan(args.input, args.output, config, args.metadata_only)
            print(json.dumps(result, indent=2))
        elif args.command == "stats":
            if not (args.output / "index.sqlite").is_file():
                raise ValueError("No index.sqlite in output")
            db = Database(args.output / "index.sqlite")
            try:
                rows = db.rows()
                summary = args.output / "reports/summary.json"
                if summary.is_file():
                    print(summary.read_text())
                    return
                print(
                    json.dumps(
                        {"indexed": len(rows), "errors": sum(bool(row["error"]) for row in rows)},
                        indent=2,
                    )
                )
            finally:
                db.close()
        else:
            from .gallery import build, serve

            if args.command == "gallery":
                build(args.output)
            else:
                serve(args.output, args.port)
    except KeyboardInterrupt:
        parser.exit(130, "Interrupted; restart the same command to resume.\n")
    except (ValueError, OSError, RuntimeError, sqlite3.Error) as exc:
        parser.exit(1, f"Error: {exc}\n")
