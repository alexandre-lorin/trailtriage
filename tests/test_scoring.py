from trailtriage.config import Interest
from trailtriage.scoring import score


def detection(name, confidence=0.9):
    return [{"class": name, "confidence": confidence, "box": [0, 0, 10, 10]}]


def test_order_and_retention():
    config = Interest()
    results = [score(detection(name), config) for name in ("person", "dog", "chair")]
    assert (
        results[0]["interest_score"] > results[1]["interest_score"] > results[2]["interest_score"]
    )
    assert score([], config)["interest_score"] == 0
    assert results[1]["primary_category"] == "animals"
    assert score(detection("person", 0.2), config)["primary_category"] == "uncertain"
    assert results[2]["primary_category"] == "uncertain"


def test_mixed():
    result = score(detection("person") + detection("car"), Interest())
    assert result["primary_category"] == "mixed"
