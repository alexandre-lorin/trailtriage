from .config import Interest


def score(detections: list[dict], config: Interest) -> dict:
    groups = set()
    value = 0.0
    max_confidence = max((d["confidence"] for d in detections), default=0.0)
    weights = {
        "humans": config.human_weight,
        "vehicles": config.vehicle_weight,
        "animals": config.animal_weight,
    }
    for detection in detections:
        matched = [g for g, names in config.category_groups.items() if detection["class"] in names]
        groups.update(matched)
        weight = max(
            (weights.get(g, config.unknown_weight) for g in matched), default=config.unknown_weight
        )
        value = max(value, weight if detection["confidence"] >= 0.35 else weight * 0.5)
    value += min(max(0, len(detections) - 1) * 2, 10)
    if not detections:
        category = "empty"
    elif not groups or value < config.interesting_threshold:
        category = "uncertain"
    else:
        category = next(iter(groups)) if len(groups) == 1 else "mixed"
    return {
        "detection_count": len(detections),
        "detected_classes": sorted({d["class"] for d in detections}),
        "detections": detections,
        "max_confidence": max_confidence,
        "interest_score": value,
        "score_band": (
            "interesting"
            if value >= config.interesting_threshold
            else "uncertain"
            if value >= config.uncertain_threshold
            else "low"
        ),
        "primary_category": category,
    }
