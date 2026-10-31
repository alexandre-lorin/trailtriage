from trailtriage.bursts import group_bursts
from trailtriage.config import Bursts


def test_temporal_groups_and_camera_isolation():
    rows = [
        {"original_path": f"/camera/a/{i}.jpg", "filename": f"{i}.jpg", "timestamp": t}
        for i, t in enumerate([0, 2, 6, 120])
    ]
    rows.append({"original_path": "/camera/b/1.jpg", "filename": "1.jpg", "timestamp": 2})
    grouped = group_bursts(rows[::-1], Bursts())
    assert [r["burst_id"] for r in grouped] == [1, 1, 1, 2, 3]
    assert len({r["burst_id"] for r in group_bursts(rows, Bursts(enabled=False))}) == 5
