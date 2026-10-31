"""Benchmark a small local sample; output caches are reused on subsequent runs."""

import argparse
import json
import resource
from pathlib import Path
from time import perf_counter

from trailtriage.config import Config
from trailtriage.database import Database
from trailtriage.pipeline import scan

parser = argparse.ArgumentParser()
parser.add_argument("input", type=Path)
parser.add_argument("--model", default="models/yolo11n.pt")
parser.add_argument("--output", type=Path, default=Path("output/benchmark"))
parser.add_argument("--device", default="auto")
parser.add_argument("--batch-size", type=int, default=4)
args = parser.parse_args()
config = Config()
config.detection.model = args.model
config.detection.device = args.device
config.detection.batch_size = args.batch_size
import torch  # noqa: E402

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
start = perf_counter()
summary = scan(args.input, args.output, config)
elapsed = perf_counter() - start
db = Database(args.output / "index.sqlite")
rows = [r for r in db.rows() if not r["error"]]
db.close()
summary.update(
    {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "model": args.model,
        "batch_size": args.batch_size,
        "runtime_seconds": elapsed,
        "images_per_second": len(rows) / elapsed,
        "average_resolution": [
            sum(r[k] for r in rows) / len(rows) if rows else 0 for k in ("width", "height")
        ],
        "peak_cpu_ram_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "peak_vram_mb": torch.cuda.max_memory_allocated() / 1024**2
        if torch.cuda.is_available()
        else 0,
        "cache_note": "Use a fresh output directory to measure a cold run",
    }
)
print(json.dumps(summary, indent=2))
