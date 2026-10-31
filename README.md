# TrailTriage

Local trail-camera photo triage. TrailTriage detects people, vehicles and animals, groups near-duplicate images, and creates a smaller set for human review.

**Images stay on your machine. Originals are not deleted or modified.**

## Status

The seven pipeline stages are implemented. CPU behavior has been tested; CUDA and a real trail-camera dataset still need validation. The proposed reduction from 50,000 images to 500–2,000 depends on scenes and settings and is not guaranteed.

## Installation

Python 3.12 is recommended. Use an isolated environment:

```sh
git clone https://github.com/alexandre-lorin/trailtriage.git
cd trailtriage
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[detection]'
```

Use `pip install -e .` for discovery without detection dependencies. Detection requires a local `.pt` model. Obtain one explicitly before scanning; see the [full setup reference](docs/README-REFERENCE.md). Scans do not automatically download models or images.

## First scan

Start with a small sample:

```sh
trailtriage --help
trailtriage scan ~/Pictures/camera --dry-run
trailtriage scan ~/Pictures/camera --output ~/Pictures/camera-review
trailtriage serve ~/Pictures/camera-review
```

Open `http://127.0.0.1:8765`. Review decisions are stored in SQLite. Stop the server before rescanning into the same output directory.

Use `--device cpu` to force CPU; a requested CUDA device fails explicitly if unavailable. `auto` uses CUDA only when PyTorch exposes it. See the reference for model, batch, cache and configuration options.

## Outputs and privacy

Review folders use symlinks by default; choose `output.mode: copy` for independent copies. Earlier exports are retained under `.history/`. Copy mode can consume additional disk space on every rebuild.

The review server binds to loopback and re-encodes served images without EXIF/GPS. The static gallery opens original images, which retain their metadata. Reports contain local paths and dates: do not publish output folders, photos or model files.

**An empty classification means no detection, not proof that nothing happened.** Inspect `errors.csv` and sample the empty group to assess false negatives. Small or nighttime subjects and animals outside model classes may be missed.

## Development

```sh
pip install -e '.[dev,detection]'
pytest
ruff check .
ruff format --check .
```

See the [implementation plan](TODO.md) and [detailed reference](docs/README-REFERENCE.md) for pipeline behavior, safety rules and remaining work.
