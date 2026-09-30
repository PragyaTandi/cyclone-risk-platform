CRITICALITY_WEIGHTS = {
    "hospital": 1.0,
    "shelter": 1.0,
    "power": 0.9,
    "road": 0.8,
    "other": 0.5
}


def calculate_priority(infrastructure):
    results = []

    for asset in infrastructure:
        risk_score = asset.get("risk_score", 0)

        asset_type = asset.get("type", "other").lower()
        criticality = CRITICALITY_WEIGHTS.get(asset_type, 0.5)

        priority_score = risk_score * criticality

        results.append({
            **asset,
            "criticality": criticality,
            "priority_score": round(priority_score, 3)
        })

    results.sort(
        key=lambda x: x["priority_score"],
        reverse=True
    )

    for index, asset in enumerate(results, start=1):
        asset["priority"] = index

    return results