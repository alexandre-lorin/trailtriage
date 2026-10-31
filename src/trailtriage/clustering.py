from collections import deque

from .config import Deduplication
from .hashing import distance
from .selection import select


def compatible(first: dict, second: dict) -> bool:
    # Never merge across different detected content. Unknown objects stay distinct.
    if first["primary_category"] != second["primary_category"]:
        return False
    if first["detected_classes"] != second["detected_classes"]:
        return False
    if first["detection_count"] != second["detection_count"]:
        return False
    if first.get("contrast", 0) < 5 or second.get("contrast", 0) < 5:
        return False  # Flat images have unreliable perceptual hashes.
    for a, b in zip(first.get("detections", []), second.get("detections", [])):
        if a["class"] != b["class"]:
            return False
        # Boxes use the same bounded inference coordinate system.
        if max(abs(x - y) for x, y in zip(a["box"], b["box"])) > 30:
            return False
    return True


def cluster(rows: list[dict], config: Deduplication) -> list[dict]:
    groups: list[list[dict]] = []
    recent: deque[int] = deque(maxlen=config.comparison_window)
    previous_burst = None
    for row in rows:
        if previous_burst != row["burst_id"]:
            recent.clear()
        previous_burst = row["burst_id"]
        matched = None
        if config.enabled:
            for index in reversed(recent):
                anchor = groups[index][0]
                if compatible(anchor, row) and distance(anchor["phash"], row["phash"]) <= (
                    config.phash_distance
                ):
                    matched = index
                    break
        if matched is None:
            matched = len(groups)
            groups.append([])
            recent.append(matched)
        groups[matched].append(row)
        row["cluster_id"] = matched + 1
    for group in groups:
        selected = select(group, config.representatives_per_cluster)
        for row in group:
            row["representative"] = row["original_path"] in selected
    return rows
