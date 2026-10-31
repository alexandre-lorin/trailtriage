import pytest

from trailtriage.config import Detection
from trailtriage.detector import model_identity


def test_local_model_required():
    with pytest.raises(ValueError):
        model_identity(Detection(model="https://example.com/model.pt"))


def test_model_identity_invalidates_same_name(tmp_path):
    path = tmp_path / "model.pt"
    path.write_bytes(b"first")
    config = Detection(model=str(path))
    first = model_identity(config)
    path.write_bytes(b"second")
    assert model_identity(config) != first
