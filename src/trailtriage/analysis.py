import hashlib
import json
import logging
from pathlib import Path
from time import perf_counter

from PIL import Image, ImageOps
from tqdm import tqdm

from .bursts import group_bursts
from .clustering import cluster
from .config import Config
from .database import Database
from .detector import Detector, model_identity
from .exporter import export
from .hashing import features
from .scoring import score


def analyze(paths: list[Path], db: Database, config: Config, output: Path) -> dict:
    identity = model_identity(config.detection)
    key = hashlib.sha256(
        json.dumps(
            {
                "version": 2,
                "model": identity,
                "confidence": config.detection.confidence_threshold,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    pending = []
    for path in paths:
        row = db.get(path)
        if row and not row["error"] and row["analysis_key"] != key:
            pending.append(path)
    detector = Detector(config.detection) if pending else None
    count = 0
    inference_start = perf_counter()
    kept = 0
    if detector:
        progress = tqdm(range(0, len(pending), detector.batch_size), desc="YOLO", unit="batch")
        for start in progress:
            batch_paths, images = [], []
            try:
                for path in pending[start : start + detector.batch_size]:
                    try:
                        with Image.open(path) as source:
                            image = ImageOps.exif_transpose(source).convert("RGB")
                            image.thumbnail((1280, 1280))
                            images.append(image)
                            batch_paths.append(path)
                    except (OSError, ValueError) as exc:
                        row = db.get(path)
                        assert row is not None
                        db.save(path, json.loads(row["metadata"]), str(exc))
                        logging.getLogger("trailtriage").warning("%s: %s", path, exc)
                if not images:
                    continue
                begin = perf_counter()
                predictions = detector.predict(images)
                if len(predictions) != len(batch_paths):
                    raise RuntimeError("Detector returned an incomplete batch")
                elapsed = (perf_counter() - begin) * 1000 / len(images)
                for path, detections in zip(batch_paths, predictions):
                    if not db.unchanged(path):
                        raise RuntimeError(f"Input changed during inference: {path}; restart scan")
                    analysis = score(detections, config.interest)
                    analysis["processing_time_ms"] = elapsed
                    db.save_analysis(path, key, analysis)
                    count += 1
                    kept += analysis["primary_category"] != "empty"
                progress.set_postfix(inferred=count, review_candidates=kept)
            finally:
                for image in images:
                    image.close()
    inference_seconds = perf_counter() - inference_start
    hashing_start = perf_counter()
    active = {str(path) for path in paths}
    rows = [row for row in db.rows() if row["original_path"] in active]
    good = []
    for row in tqdm(rows, desc="Hashing", unit="image"):
        if row.get("error"):
            continue
        path = Path(row["original_path"])
        try:
            row.update(score(row["detections"], config.interest))
            if "phash" not in row:
                row.update(features(path))
            cached = db.get(path)
            assert cached is not None
            analysis = json.loads(cached["analysis"])
            updated = {
                **analysis,
                **score(row["detections"], config.interest),
                **{k: row[k] for k in ("phash", "sharpness_score", "contrast")},
            }
            if updated != analysis:
                db.save_analysis(path, key, updated)
            good.append(row)
        except (OSError, ValueError) as exc:
            logging.getLogger("trailtriage").warning("%s: %s", path, exc)
            row["error"] = str(exc)
            cached = db.get(path)
            assert cached is not None
            db.save(path, json.loads(cached["metadata"]), str(exc))
    hashing_seconds = perf_counter() - hashing_start
    clustering_start = perf_counter()
    clustered = cluster(group_bursts(good, config.bursts), config.deduplication)
    for row in clustered:
        cached = db.get(Path(row["original_path"]))
        assert cached is not None
        analysis = json.loads(cached["analysis"])
        updated = {**analysis, **{k: row[k] for k in ("burst_id", "cluster_id", "representative")}}
        if updated != analysis:
            db.save_analysis(Path(row["original_path"]), key, updated)
    clustering_seconds = perf_counter() - clustering_start
    export_start = perf_counter()
    summary = export(rows, output, config.output)
    export_seconds = perf_counter() - export_start
    gallery_start = perf_counter()
    if config.output.generate_gallery:
        from .gallery import build

        build(output)
    timings = {
        "yolo_seconds": round(inference_seconds, 3),
        "hashing_seconds": round(hashing_seconds, 3),
        "clustering_seconds": round(clustering_seconds, 3),
        "export_seconds": round(export_seconds, 3),
        "gallery_seconds": round(perf_counter() - gallery_start, 3),
    }
    summary.update(timings)
    (output / "reports/summary.json").write_text(json.dumps(summary, indent=2))
    return {"inferred": count, **summary}
