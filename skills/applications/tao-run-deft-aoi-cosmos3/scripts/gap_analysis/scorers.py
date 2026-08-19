"""Record-level weakness scorers."""

from __future__ import annotations

import json
import math
from typing import Any


def _finite_unit(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{field} must be finite and in [0, 1]")
    return result


def score_rows(rows: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    scorer = config["scorer"]
    name = scorer["name"]
    output: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        sample_score = _finite_unit(row.get("sample_score"), field=f"{row.get('id')}.sample_score")
        deficit = 1.0 - sample_score
        uncertainty_value = row.get("uncertainty")
        uncertainty = (
            _finite_unit(uncertainty_value, field=f"{row.get('id')}.uncertainty")
            if uncertainty_value is not None
            else None
        )
        components: dict[str, float] = {"deficit": deficit}
        if name == "binary_error":
            score = 0.0 if bool(row.get("parse_ok")) and sample_score == 1.0 else 1.0
        elif name == "sample_metric_deficit":
            score = deficit
        elif name == "risk_weighted_deficit":
            weights = scorer.get("risk_weights", {})
            if not isinstance(weights, dict):
                raise ValueError("scorer.risk_weights must be an object")
            risk = weights.get(str(row.get("gap_type")), weights.get(str(row.get("task_type")), 1.0))
            risk = _finite_unit(risk, field="risk weight")
            components["risk"] = risk
            score = deficit * risk
        elif name == "uncertainty":
            if uncertainty is None:
                raise ValueError(f"candidate {row.get('id')!r} has no uncertainty field")
            components["uncertainty"] = uncertainty
            score = uncertainty
        elif name == "hybrid":
            if uncertainty is None:
                raise ValueError(f"candidate {row.get('id')!r} has no uncertainty field")
            weights = scorer.get("weights", {})
            if not isinstance(weights, dict) or not weights:
                raise ValueError("hybrid scorer requires non-empty weights")
            risk_weights = scorer.get("risk_weights", {})
            risk = risk_weights.get(str(row.get("gap_type")), risk_weights.get(str(row.get("task_type")), 1.0))
            risk = _finite_unit(risk, field="risk weight")
            components.update({"risk": risk, "uncertainty": uncertainty})
            total_weight = sum(float(weights.get(key, 0.0)) for key in components)
            if total_weight <= 0:
                raise ValueError("hybrid scorer weights must sum to a positive value")
            score = sum(float(weights.get(key, 0.0)) * value for key, value in components.items()) / total_weight
        else:  # validated before dispatch
            raise ValueError(f"unsupported scorer {name!r}")
        row["weakness_score"] = _finite_unit(score, field="weakness_score")
        row["selection_score"] = row["weakness_score"]
        row["scorer_components_json"] = json.dumps(components, sort_keys=True)
        output.append(row)
    return output
