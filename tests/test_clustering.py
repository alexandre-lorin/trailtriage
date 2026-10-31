from PIL import Image

from trailtriage.clustering import cluster
from trailtriage.config import Deduplication
from trailtriage.hashing import distance, features


def row(i, **changes):
    return {
        "original_path": str(i),
        "burst_id": 1,
        "phash": "0000000000000000",
        "primary_category": "humans",
        "detected_classes": ["person"],
        "detection_count": 1,
        "detections": [],
        "contrast": 40,
        "interest_score": 100,
        "max_confidence": 0.9,
        "sharpness_score": i,
        **changes,
    }


def test_best_and_content_safety():
    rows = [row(1), row(2), row(3, detected_classes=["dog"]), row(4, burst_id=2)]
    cluster(rows, Deduplication())
    assert [r["representative"] for r in rows] == [False, True, True, True]
    assert rows[0]["cluster_id"] == rows[1]["cluster_id"]


def test_no_transitive_drift():
    rows = [row(1), row(2, phash="000000000000000f"), row(3, phash="00000000000000ff")]
    cluster(rows, Deduplication(phash_distance=4))
    assert rows[0]["cluster_id"] != rows[2]["cluster_id"]


def test_large_burst_endpoints():
    rows = [row(i) for i in range(20)]
    cluster(rows, Deduplication())
    assert [r["original_path"] for r in rows if r["representative"]] == ["0", "19"]


def test_hashing(tmp_path):
    import numpy as np

    rng = np.random.default_rng(42)
    array = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    first, second = tmp_path / "a.png", tmp_path / "b.png"
    Image.fromarray(array).save(first)
    Image.fromarray(255 - array).save(second)
    a, b = features(first), features(second)
    assert distance(a["phash"], features(first)["phash"]) == 0
    assert distance(a["phash"], b["phash"]) > 20
    assert a["sharpness_score"] > 0


def test_near_identical_hash(tmp_path):
    import numpy as np
    from PIL import ImageEnhance

    rng = np.random.default_rng(27)
    image = Image.fromarray(rng.integers(0, 255, (128, 128, 3), dtype=np.uint8))
    first, second = tmp_path / "original.png", tmp_path / "brighter.png"
    image.save(first)
    ImageEnhance.Brightness(image).enhance(1.03).save(second)
    assert distance(features(first)["phash"], features(second)["phash"]) <= 6


def test_comparisons_are_bounded(monkeypatch):
    import trailtriage.clustering as module

    comparisons = 0

    def counted_distance(first, second):
        nonlocal comparisons
        comparisons += 1
        return 64

    monkeypatch.setattr(module, "distance", counted_distance)
    rows = [row(i) for i in range(2000)]
    cluster(rows, Deduplication(comparison_window=16))
    assert comparisons <= len(rows) * 16
