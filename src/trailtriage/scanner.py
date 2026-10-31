from pathlib import Path

from .config import Config


def discover(root: Path, config: Config, excluded: Path | None = None) -> list[Path]:
    if not root.is_dir():
        raise ValueError(f"Input directory does not exist: {root}")
    extensions = {"." + e.lower().lstrip(".") for e in config.input.extensions}
    paths = root.rglob("*") if config.input.recursive else root.iterdir()
    return sorted(
        p
        for p in paths
        if p.is_file()
        and not p.is_symlink()
        and p.suffix.lower() in extensions
        and (excluded is None or not p.resolve().is_relative_to(excluded))
    )
