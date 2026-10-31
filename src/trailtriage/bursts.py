from pathlib import Path

from .config import Bursts


def group_bursts(rows: list[dict], config: Bursts) -> list[dict]:
    # Parent directory is the conservative default camera identity.
    ordered = sorted(
        rows, key=lambda r: (str(Path(r["original_path"]).parent), r["timestamp"], r["filename"])
    )
    burst = 0
    previous = None
    for row in ordered:
        camera = str(Path(row["original_path"]).parent)
        if (
            not config.enabled
            or previous is None
            or camera != previous[0]
            or row["timestamp"] - previous[1] > config.max_gap_seconds
        ):
            burst += 1
        row["burst_id"] = burst
        previous = (camera, row["timestamp"])
    return ordered
