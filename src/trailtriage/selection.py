def select(rows: list[dict], count: int) -> set[str]:
    ranked = sorted(
        rows,
        key=lambda row: (row["interest_score"], row["max_confidence"], row["sharpness_score"]),
        reverse=True,
    )
    selected = {row["original_path"] for row in ranked[:count]}
    # Preserve event endpoints in large review clusters.
    if len(rows) >= 10 and any(row["primary_category"] != "empty" for row in rows):
        selected.update([rows[0]["original_path"], rows[-1]["original_path"]])
    return selected
