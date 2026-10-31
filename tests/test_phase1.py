import pytest
from PIL import Image

from trailtriage.config import Config, load_config
from trailtriage.database import Database
from trailtriage.metadata import extract
from trailtriage.pipeline import scan
from trailtriage.scanner import discover


def test_resume_and_safety(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    photo = source / "photo.JPG"
    Image.new("RGB", (32, 24), "red").save(photo)
    (source / "broken.jpg").write_bytes(b"not a jpeg")
    before = photo.read_bytes(), photo.stat().st_mtime_ns
    output = tmp_path / "output"
    first = scan(source, output, Config(), metadata_only=True)
    assert {k: first[k] for k in ("found", "processed", "skipped", "errors")} == {
        "found": 2,
        "processed": 1,
        "skipped": 0,
        "errors": 1,
    }
    assert scan(source, output, Config(), True)["skipped"] == 2
    assert (photo.read_bytes(), photo.stat().st_mtime_ns) == before
    Image.new("RGB", (40, 24), "blue").save(photo)
    assert scan(source, output, Config(), True)["processed"] == 1
    db = Database(output / "index.sqlite")
    assert db.get(photo)["error"] is None
    assert len(db.rows()) == 2
    db.close()


def test_discovery_exclusion(tmp_path):
    (tmp_path / "output").mkdir()
    for path in (tmp_path / "a.JPG", tmp_path / "output/b.jpg"):
        path.touch()
    (tmp_path / "alias.jpg").symlink_to(tmp_path / "a.JPG")
    assert discover(tmp_path, Config(), tmp_path / "output") == [tmp_path / "a.JPG"]


def test_config(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("detection:\n  batch_size: 0\n")
    with pytest.raises(ValueError):
        load_config(path)
    path.write_text("interest:\n  category_groups:\n    ../escape: [person]\n")
    with pytest.raises(ValueError):
        load_config(path)
    path.write_text("typo: true")
    with pytest.raises(ValueError):
        load_config(path)


def test_exif_orientation(tmp_path):
    path = tmp_path / "rotated.jpg"
    exif = Image.Exif()
    exif[274] = 6
    exif[36867] = "2026:01:02 12:00:00"
    Image.new("RGB", (40, 20)).save(path, exif=exif)
    result = extract(path)
    assert (result["width"], result["height"]) == (20, 40)
    assert result["timestamp_source"] == "exif"


def test_output_cannot_be_input(tmp_path):
    with pytest.raises(ValueError):
        scan(tmp_path, tmp_path, Config(), True)
