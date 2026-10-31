# TrailTriage

TrailTriage scans trail camera photo folders locally, detects
people, vehicles and animals, combines almost identical images and produces
a smaller selection to check out. **Images never leave the machine.**
**Originals are never deleted or modified.**

The seven phases of the pipeline are implemented. CPU operation is tested;
CUDA validation and a real hunting camera game remain to be done.
The lens 50 000 → 500–2 000 depends on the scene and settings; it is not guaranteed.

## Linux/Arch installation

Python 3.12 recommended, `uv` or `pip`, isolated environment:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e '.[dev,detection]'
```

Alternative:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[detection]'
```

Discovery alone only requires `pip install -e .`.
To explicitly download the initial model (once, before scanning):

```bash
mkdir -p ~/.local/share/trailtriage/models
curl -fL https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt \
  -o ~/.local/share/trailtriage/models/yolo11s.pt
```

The installation carried out on this machine includes the YOLO11s model and a launcher
`~/.local/bin/trailtriage` ; the command works without activating the venv.
The YOLO11n test model remains in `models/` (ignored by Git).
The scan does not download any models or images. Telemetry and automatic installation
Ultralytics disabled. The detection package is an optional extra.

## CUDA

```bash
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Choice `auto` uses CUDA if PyTorch makes it available, CPU otherwise.
This validation installation uses PyTorch CPU: when the NVIDIA driver is
functional, install the CUDA wheels suitable for this driver in `.venv` according to
[the official PyTorch ](https://pytorch.org/get-started/locally/) selector, then redo
verification. No system drivers have been modified by TrailTriage.
`--device cpu` CPU strength; `--device 0` requests CUDA and clearly fails if unavailable.
Auto batch: 8 GPU images, 2 CPU. A lack of VRAM splits the batch recursively.

## First scan

Start with a sample, then use the same output folder to resume:

```bash
trailtriage --help
trailtriage scan ~/Pictures/camera --dry-run
trailtriage scan ~/Pictures/camera --output ~/Pictures/camera-review
trailtriage serve ~/Pictures/camera-review
```

Open `http://127.0.0.1:8765`. Decisions are recorded in SQLite.
Stop the server before restarting a scan in the same output folder.
Filters, sorting by score/date and labels are available; boxes and
Keyboard shortcuts are not included. The static gallery `gallery/index.html`
also works without a server, but without recording decisions.

```bash
trailtriage scan photos --output output --metadata-only
trailtriage scan photos --output output --model /chemin/modele.pt --batch-size 4
trailtriage scan photos --output output --no-dedup --device cpu
trailtriage scan photos --output output --config config.example.yaml
trailtriage stats output
trailtriage gallery output
```

Default output: `trailtriage-output` in the current directory.
An existing output folder must be empty or belong to this same scan.
Strict YAML configuration: unknown key causes explicit error.
A modified file is reanalyzed; unchanged images return to their cache.
Changing scores or deduplication does not restart YOLO. Change model or
its confidence threshold restarts the inference. Manual labels are retained.

## Exits and safety

```text
output/
├── index.sqlite
├── review/{humans,vehicles,animals,mixed,uncertain}/
├── boring/empty/
├── duplicates/
├── reports/{results.csv,results.jsonl,summary.json,errors.csv,run.log}
├── gallery/{index.html,thumbnails/}
└── .history/     # précédentes générations conservées
```

Review files are symbolic links to the originals; choose
`output.mode: copy` for copies. No physical link likely to allow
the modification of the originals is not created. Do not move sources without redoing
the scan. Output folders are never rescanned and input links
are ignored. Suspicious links in report paths are rejected.
No delete command is offered.
The old outputs are kept under `.history/`: in copy mode they
consume additional space each time they rebuild.

The server only binds to `127.0.0.1`. The thumbnails and images served are
re-encoded without EXIF/GPS. The static gallery directly opens the originals,
who retain their EXIF. Reports contain local paths and dates:
do not publish these files. Images and templates are ignored by Git.

## Limits and settings

**“Empty” means no detection, not certain absence of event.** Check
`errors.csv` and a sample of `boring/empty` to evaluate false negatives.
The small silhouettes, nocturnal scenes and animals absent from the model's classes
may be missed. We can provide a compatible local model and modify
`interest.category_groups`, with no COCO assumption in the rest of the pipeline.
Any weak or unknown detection is kept in `uncertain`, even below the threshold
uncertain. The threshold defines the score band, not a permission to hide.

Put each camera in a separate folder. Bursts use local EXIF,
then mtime, then the name to decide between. EXIF dates without a time zone are interpreted
as local. Deduplication compares to most recent 16 groups in the same
burst, to a fixed anchor, with compatible detected content. The scenes almost
uniforms remain distinct. For long clusters to be reviewed, also keep
the first and the last. All duplicates remain viewable in `duplicates`.

## Tests and benchmarks

```bash
pytest
ruff check .
ruff format --check .
mypy src/trailtriage
python scripts/make_sample.py
python scripts/offline_smoke.py
python scripts/benchmark.py /chemin/echantillon --model models/yolo11n.pt
```

The synthetic game contains drawn scenes, bursts and a corrupted file.
The drawings do not measure the accuracy of YOLO. The actual test also used
[Ultralytics public bus image](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/assets/bus.jpg).
Offline testing blocks and counts outgoing connections. The benchmark measures
steps, throughput, RAM and peak VRAM (if available). Use a new outlet to
measure a scan without matte. See `TODO.md` for remaining field validations.
