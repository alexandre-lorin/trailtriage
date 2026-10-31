import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from uuid import uuid4

from .config import Output
from .utils import safe_target


def export(rows: list[dict], output: Path, config: Output) -> dict:
    # Preserve earlier output trees instead of deleting anything.
    history = output / ".history" / uuid4().hex
    for name in ("review", "boring", "duplicates", "gallery"):
        path = safe_target(output, output / name)
        if path.exists():
            safe_target(output, history).mkdir(parents=True, exist_ok=True)
            path.rename(history / name)
    for name in (
        "review/humans",
        "review/vehicles",
        "review/animals",
        "review/mixed",
        "review/uncertain",
        "boring/empty",
        "duplicates",
        "reports",
    ):
        safe_target(output, output / name).mkdir(parents=True, exist_ok=True)
    for row in rows:
        if row.get("error") or "representative" not in row:
            continue
        source = Path(row["original_path"])
        if not source.is_file():
            raise OSError(f"Source unavailable: {source}")
        digest = hashlib.sha256(str(source).encode()).hexdigest()[:24]
        name = digest + source.suffix.lower()
        if not row["representative"]:
            folder = "duplicates"
        elif row["primary_category"] == "empty":
            folder = "boring/empty"
        else:
            folder = "review/" + row["primary_category"]
        destination = safe_target(output, output / folder / name)
        if config.mode == "symlink":
            destination.symlink_to(source)
        else:
            shutil.copyfile(source, destination)
        row["output_path"] = str(destination.relative_to(output))
    review = sum(
        bool(r.get("representative")) and r.get("primary_category") != "empty" for r in rows
    )
    summary = {
        "input_images": len(rows),
        "failed_images": sum(bool(r.get("error")) for r in rows),
        "images_with_detections": sum(r.get("detection_count", 0) > 0 for r in rows),
        "review_images": review,
        "duplicates": sum(r.get("representative") is False for r in rows),
        "clusters": len({r["cluster_id"] for r in rows if "cluster_id" in r}),
        "reduction_percent": round(100 * (1 - review / len(rows)), 2) if rows else 0,
        "categories": dict(Counter(r.get("primary_category", "error") for r in rows)),
        "warning": (
            "Empty means no detection, not proof of absence. Inspect errors and sample empties."
        ),
    }
    reports = output / "reports"
    safe_target(output, reports / "summary.json").write_text(json.dumps(summary, indent=2))
    with safe_target(output, reports / "results.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    fields = [
        "original_path",
        "filename",
        "timestamp",
        "timestamp_source",
        "width",
        "height",
        "file_size",
        "phash",
        "sharpness_score",
        "detection_count",
        "detected_classes",
        "max_confidence",
        "interest_score",
        "score_band",
        "primary_category",
        "burst_id",
        "cluster_id",
        "representative",
        "processing_time_ms",
        "error",
        "label",
        "output_path",
    ]
    with safe_target(output, reports / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with safe_target(output, reports / "errors.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["original_path", "error"], extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(r for r in rows if r.get("error"))
    return summary
