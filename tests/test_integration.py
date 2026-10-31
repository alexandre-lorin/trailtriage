import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest
from PIL import Image, ImageDraw

from trailtriage.config import Config
from trailtriage.database import Database
from trailtriage.gallery import make_handler
from trailtriage.pipeline import scan


class FakeDetector:
    calls = 0
    batch_size = 4

    def __init__(self, config):
        pass

    def predict(self, images):
        FakeDetector.calls += len(images)
        return [[{"class": "person", "confidence": 0.9, "box": [20, 20, 80, 100]}] for _ in images]


@pytest.fixture
def completed(tmp_path, monkeypatch):
    from trailtriage import analysis

    monkeypatch.setattr(analysis, "Detector", FakeDetector)
    source = tmp_path / "input"
    source.mkdir()
    for i in range(24):
        image = Image.new("RGB", (160, 120), "white")
        ImageDraw.Draw(image).rectangle((20, 20, 80, 100), fill="black")
        image.save(source / f"{i:02}.jpg")
    (source / "corrupt.jpg").write_bytes(b"broken")
    model = tmp_path / "model.pt"
    model.write_bytes(b"test model")
    config = Config()
    config.detection.model = str(model)
    output = tmp_path / "output"
    summary = scan(source, output, config)
    return source, output, config, summary


def test_pipeline_resume_reports_and_safety(completed):
    source, output, config, summary = completed
    assert summary["input_images"] == 25
    assert summary["review_images"] <= 3
    assert summary["failed_images"] == 1
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in source.iterdir()}
    second = scan(source, output, config)
    assert second["inferred"] == 0 and second["skipped"] == 25
    assert {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in source.iterdir()} == before
    for name in ("results.csv", "results.jsonl", "summary.json", "errors.csv", "run.log"):
        assert (output / "reports" / name).is_file()
    assert (output / "gallery/index.html").is_file()
    assert len((output / "reports/results.jsonl").read_text().splitlines()) == 25
    db = Database(output / "index.sqlite")
    db.label(str(source / "00.jpg"), "known_person")
    config.interest.human_weight = 101
    assert scan(source, output, config)["inferred"] == 0
    assert json.loads(db.get(source / "00.jpg")["analysis"])["interest_score"] == 101
    assert db.get(source / "00.jpg")["label"] == "known_person"
    db.close()


def test_local_gallery_and_labels(completed):
    _, output, _, _ = completed
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(output))
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(base) as response:
            html = response.read().decode()
        token = html.split('token="')[1].split('"')[0]
        with urllib.request.urlopen(base + "/thumbnails/0.jpg") as response:
            assert response.headers["Content-Type"] == "image/jpeg"
        with urllib.request.urlopen(base + "/image/0") as response:
            assert response.read().startswith(b"\xff\xd8")
        request = urllib.request.Request(
            base + "/label",
            method="POST",
            data=json.dumps({"id": 0, "label": "interesting"}).encode(),
            headers={"X-TrailTriage-Token": token, "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request) as response:
            assert json.loads(response.read())["ok"]
        request = urllib.request.Request(base + "/label", data=b"{}", method="POST")
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request)
        assert error.value.code == 403
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(base + "/../index.sqlite")
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def test_reject_unrelated_output(tmp_path):
    source, output = tmp_path / "input", tmp_path / "output"
    source.mkdir()
    output.mkdir()
    (output / "valuable.txt").write_text("keep")
    with pytest.raises(ValueError):
        scan(source, output, Config(), True)
    assert (output / "valuable.txt").read_text() == "keep"


def test_interrupted_detection_resumes(completed, monkeypatch):
    from trailtriage import analysis

    source, output, config, _ = completed
    config.detection.confidence_threshold = 0.2

    class InterruptDetector(FakeDetector):
        batches = 0

        def predict(self, images):
            self.batches += 1
            if self.batches == 2:
                raise KeyboardInterrupt
            return super().predict(images)

    monkeypatch.setattr(analysis, "Detector", InterruptDetector)
    with pytest.raises(KeyboardInterrupt):
        scan(source, output, config)
    db = Database(output / "index.sqlite")
    assert db.connection.execute("SELECT status FROM runs ORDER BY id DESC LIMIT 1").fetchone()[
        0
    ] == ("interrupted")
    db.close()
    monkeypatch.setattr(analysis, "Detector", FakeDetector)
    summary = scan(source, output, config)
    assert summary["inferred"] == 20
    assert summary["skipped"] == 25


def test_output_link_to_original_refused(completed):
    source, output, config, _ = completed
    target = output / "reports/results.csv"
    target.rename(output / "reports/results.previous.csv")
    target.symlink_to(source / "00.jpg")
    before = (source / "00.jpg").read_bytes()
    with pytest.raises(ValueError):
        scan(source, output, config)
    assert (source / "00.jpg").read_bytes() == before
