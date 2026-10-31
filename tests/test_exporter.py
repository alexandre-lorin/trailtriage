from PIL import Image

from trailtriage.config import Output
from trailtriage.exporter import export


def test_links_and_rebuild_preserve_files(tmp_path):
    source = tmp_path / "original.jpg"
    Image.new("RGB", (20, 20)).save(source)
    before = source.read_bytes(), source.stat().st_mtime_ns
    output = tmp_path / "output"
    output.mkdir()
    rows = [
        {
            "original_path": str(source),
            "primary_category": "humans",
            "representative": True,
            "cluster_id": 1,
        }
    ]
    assert export(rows, output, Output())["review_images"] == 1
    link = output / rows[0]["output_path"]
    assert link.is_symlink() and link.resolve() == source
    export(rows, output, Output())
    assert list((output / ".history").rglob("*.jpg"))
    assert (source.read_bytes(), source.stat().st_mtime_ns) == before


def test_refuse_output_symlink(tmp_path):
    import pytest

    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    (output / "review").symlink_to(source)
    with pytest.raises(ValueError):
        export([], output, Output())
    assert source.is_dir()
    from trailtriage.utils import safe_target

    with pytest.raises(ValueError):
        safe_target(output, output / ".." / "source" / "photo.jpg")
