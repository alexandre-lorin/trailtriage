# Plan TrailTriage

- [x] Phase 1: CLI, validated configuration, discovery, EXIF, SQLite, recovery, logs.
- [x] Phase 2: local YOLO, bounded batches, CPU/CUDA, scores and detection cache.
- [x] Phase 3: temporal grouping by camera/folder.
- [x] Phase 4: pHash, bounded comparison, conservative selection.
- [x] Phase 5: safe links, no writing in the originals.
- [x] Phase 6: CSV, JSONL, errors, summary and metrics.
- [x] Phase 7: filterable local gallery, persistent labels.

## Decisions
- Isolated Python 3.12: System Python 3.14 should not impose compatibility on GPU libraries.
- Optional extra detection: discovery and recovery can still be used without PyTorch.
- Exclusively local YOLO weights: explicit initial download, no connection required for scanning.
- An “empty” result means no detection, not guaranteed absence of event.
- Low detections remain under review; the proposed thresholds must not obscure the animals.
- No guarantee of 500–2 000 images without calibration on a real sample.
- No facial recognition, VLM or training in V1.

- Conservative deduplication: same folder, same burst, compatible detected content and fixed anchor; comparisons limited to 16 groups. Flat scenes remain distinct.

## Validation status

- 23 tests: configuration, EXIF, recovery, interruption, links, categories, pHash,
  bounded comparison and server/labels.
- Ruff and Mypy type checking on the package.
- Real YOLO11n on CPU: 31 valid images + 1 corrupted, resumed without inference.
- Offline test: any outgoing connection is refused and counted; 0 attempt.
- A public Ultralytics example valid people + bus; synthetic designs
  are for development, not for measuring precision/recall.

## Simplifications justified

- SQLite stores detections and clusters as JSON per image: no joins
  unnecessary, the labels are separated and persist during reanalysis.
- Static gallery + small standard Python server: avoid FastAPI for three routes.
  Pagination of 120 images to keep the browser responsive.
- Uncertain threshold used for score band, but any weak detection remains
  reviewed even below this threshold (priority to recall).
- Camera groups inferred from parent folder. Put each camera in its folder.
- Modified score/deduplication configurations: recalculation without new inference.
- Previous releases moved to `.history/`, never cleared automatically.
- No VLM, facial recognition or training included.

## To be validated in the field

- [ ] Working NVIDIA driver and CUDA version of PyTorch: this installation
  uses PyTorch CPU and `nvidia-smi` does not communicate with the driver.
- [ ] GPU benchmark and real sample (100–500 images): precision, recall,
  non-COCO animals, quality of timestamps and camera settings.
- [ ] Full scan of 50 000 photos after sample validation.
- [ ] Objective 500–2 000 images to measure, without promising a universal reduction.
- [ ] Assisted calibration, detection boxes and shortcuts: future improvements.
