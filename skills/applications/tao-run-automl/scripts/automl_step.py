#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Durable, skill-owned AutoML step engine.

This script owns optimizer state but never launches or polls compute. The
``tao-run-automl`` skill alternates its commands with the selected platform
skill's submit/status/logs/cancel verbs:

    init -> recommend -> submit -> bind-job -> status/logs -> report -> repeat

State writes are schema-validated, atomic, and protected by ``flock``. A lost
agent/process can rerun ``recommend`` safely: existing READY recommendations
are returned unchanged, and active jobs consume the configured concurrency
until their platform results are reconciled. All search policies live here;
ordinary numerical helpers are imported only by the policies that need them.
No TAO Python package or platform implementation is imported.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
import warnings
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
SKILLS_DIR = SCRIPT_PATH.parents[3]
ARTIFACT_REFS = SKILLS_DIR / "core" / "tao-artifacts" / "references"
EXPERIMENT_SCHEMA = ARTIFACT_REFS / "automl_experiment.schema.json"
METRIC_SCHEMA = ARTIFACT_REFS / "metric_record.schema.json"
BEST_REC_SCHEMA = ARTIFACT_REFS / "best_rec.schema.json"

ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
ALGORITHMS = (
    "bayesian",
    "bfbo",
    "hyperband",
    "bohb",
    "asha",
    "pbt",
    "dehb",
    "hyperband_es",
    "llm",
    "hybrid",
    "autoresearch",
)
BUDGETED_ALGORITHMS = frozenset(
    {"hyperband", "bohb", "asha", "pbt", "dehb", "hyperband_es"}
)
TERMINAL_REC_STATES = frozenset({"SUCCEEDED", "FAILED", "CANCELED"})
SENSITIVE_FEEDBACK_KEYS = frozenset(
    {
        "answer",
        "file",
        "filename",
        "gold",
        "ground_truth",
        "id",
        "image",
        "label",
        "media",
        "path",
        "target",
        "uri",
        "url",
        "video",
        "video_id",
    }
)
BUDGET_KEYS = (
    "num_epochs",
    "epoch",
    "epochs",
    "max_epochs",
    "n_epochs",
    "train.num_epochs",
    "train.epoch",
    "train.epochs",
    "train.max_epochs",
    "train.n_epochs",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _jsonschema_validate(instance: dict, schema_path: Path) -> None:
    try:
        import jsonschema
    except ImportError as exc:  # pragma: no cover - exercised in preflight, not CI
        raise SystemExit(
            "Missing required helper 'jsonschema'. Install it with: "
            "python -m pip install jsonschema"
        ) from exc
    schema = json.loads(schema_path.read_text())
    jsonschema.validate(instance, schema)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _read_document(path: Path) -> Any:
    text = path.read_text()
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - JSON is used in tests
        raise SystemExit(
            f"{path} is not JSON and PyYAML is unavailable. Install it with: "
            "python -m pip install PyYAML"
        ) from exc
    return yaml.safe_load(text)


@contextmanager
def _state_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    with open(lock_path, "w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_name, path)
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _write_state(path: Path, state: dict) -> None:
    state["updated_at"] = _now()
    _jsonschema_validate(state, EXPERIMENT_SCHEMA)
    _atomic_write(path, state)


def _load_state(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"AutoML state does not exist: {path}")
    state = json.loads(path.read_text())
    _jsonschema_validate(state, EXPERIMENT_SCHEMA)
    return state


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, allow_nan=False))


def _parse_objective(value: str) -> dict:
    parts = value.split(":")
    if len(parts) not in {2, 3, 4}:
        raise argparse.ArgumentTypeError(
            "objectives use NAME:DIRECTION[:WEIGHT[:SCALE]]"
        )
    name, direction = parts[:2]
    if not name or direction not in {"maximize", "minimize"}:
        raise argparse.ArgumentTypeError(
            "objective direction must be maximize or minimize"
        )
    try:
        weight = float(parts[2]) if len(parts) >= 3 else 1.0
        scale = float(parts[3]) if len(parts) == 4 else 1.0
    except ValueError as exc:
        raise argparse.ArgumentTypeError("objective weight/scale must be numeric") from exc
    if not math.isfinite(weight) or weight <= 0 or not math.isfinite(scale) or scale <= 0:
        raise argparse.ArgumentTypeError("objective weight/scale must be finite and positive")
    return {"name": name, "direction": direction, "weight": weight, "scale": scale}


def _parse_named_metric(value: str) -> tuple[str, float]:
    try:
        name, raw_value = value.split("=", 1)
        metric_value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("metric values use NAME=NUMBER") from exc
    if not name or not math.isfinite(metric_value):
        raise argparse.ArgumentTypeError(
            "metric values require a non-empty name and finite number"
        )
    return name, metric_value


def _sanitize_feedback(value: Any) -> Any:
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            normalized = str(key).lower().replace("-", "_")
            if normalized in SENSITIVE_FEEDBACK_KEYS or normalized.endswith("_path"):
                continue
            result[str(key)] = _sanitize_feedback(child)
        return result
    if isinstance(value, (list, tuple)):
        return [_sanitize_feedback(child) for child in value]
    return value


def _algorithm_config(args: argparse.Namespace) -> dict:
    research_program = None
    if args.research_program:
        research_program = _read_document(args.research_program.resolve())
        if not isinstance(research_program, dict):
            raise SystemExit("--research-program must contain a JSON/YAML object")
    return {
        "max_epochs": args.max_epochs,
        "reduction_factor": args.reduction_factor,
        "epoch_multiplier": args.epoch_multiplier,
        "population_size": args.population_size,
        "max_generations": args.max_generations,
        "eval_interval": args.eval_interval,
        "perturbation_factor": args.perturbation_factor,
        "mutation_factor": args.mutation_factor,
        "crossover_probability": args.crossover_probability,
        "early_stop_threshold": args.early_stop_threshold,
        "min_early_stop_epochs": args.min_early_stop_epochs,
        "kde_samples": args.kde_samples,
        "top_fraction": args.top_fraction,
        "min_points_in_model": args.min_points_in_model,
        "max_trials": args.max_trials,
        "min_top_configs": args.min_top_configs,
        "max_experiments": args.max_experiments,
        "research_program": research_program,
        "evolvable_text_parameters": list(args.evolvable_text_parameter or []),
        "llm_endpoint": args.llm_endpoint or os.getenv("AUTOML_LLM_ENDPOINT"),
        "llm_model": args.llm_model or os.getenv("AUTOML_LLM_MODEL"),
        "llm_temperature": args.llm_temperature,
        "llm_max_tokens": args.llm_max_tokens,
        "llm_timeout": args.llm_timeout,
        "llm_max_retries": args.llm_max_retries,
        "llm_retry_delay": args.llm_retry_delay,
    }


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return number


def _get_nested(source: dict, dotted: str) -> Any:
    current: Any = source
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _set_nested(target: dict, dotted: str, value: Any) -> None:
    current = target
    parts = dotted.split(".")
    for part in parts[:-1]:
        child = current.get(part)
        if child is None:
            child = {}
            current[part] = child
        if not isinstance(child, dict):
            raise ValueError(
                f"Cannot set {dotted!r}: {part!r} is not an object in the base spec"
            )
        current = child
    current[parts[-1]] = copy.deepcopy(value)


def _schema_node(schema: dict, dotted: str) -> dict:
    node: Any = schema
    for part in dotted.split("."):
        properties = node.get("properties") if isinstance(node, dict) else None
        if not isinstance(properties, dict) or part not in properties:
            raise ValueError(f"Search parameter {dotted!r} is absent from the schema")
        node = properties[part]
    if not isinstance(node, dict):
        raise ValueError(f"Schema node for {dotted!r} is not an object")
    return node


def _walk_enabled(node: dict, prefix: str = ""):
    for name, child in (node.get("properties") or {}).items():
        dotted = f"{prefix}.{name}" if prefix else name
        if not isinstance(child, dict):
            continue
        if child.get("automl_enabled") is True:
            yield dotted
        yield from _walk_enabled(child, dotted)


def _parameter_names(schema: dict, requested: list[str] | None) -> list[str]:
    names = list(requested or schema.get("automl_default_parameters") or [])
    if not names:
        names = list(_walk_enabled(schema))
    deduplicated = list(dict.fromkeys(str(name) for name in names if str(name)))
    if not deduplicated:
        raise ValueError("The schema declares no AutoML-enabled search parameters")
    return deduplicated


def _finite_bound(value: Any) -> float | None:
    try:
        return _finite_number(value, "parameter bound")
    except (TypeError, ValueError):
        return None


def _derive_numeric_bounds(
    name: str,
    kind: str,
    default: int | float,
    node: dict,
    override: dict,
) -> tuple[float, float, str]:
    minimum = _finite_bound(
        override.get(
            "minimum",
            override.get("valid_min", node.get("minimum", node.get("valid_min"))),
        )
    )
    maximum = _finite_bound(
        override.get(
            "maximum",
            override.get("valid_max", node.get("maximum", node.get("valid_max"))),
        )
    )
    default_number = _finite_number(default, f"default for {name}")

    if kind == "int":
        if minimum is None:
            minimum = max(0.0, math.floor(default_number / 2))
        if maximum is None:
            maximum = max(minimum + 1, math.ceil(default_number * 2), 1)
    else:
        if minimum is None:
            minimum = default_number / 10 if default_number > 0 else 0.0
        if maximum is None:
            maximum = default_number * 10 if default_number > 0 else 1.0

    name_tokens = set(re.split(r"[._-]", name.lower()))
    inferred_log = bool({"lr", "learningrate", "learning"} & name_tokens) and default_number > 0
    scale = str(override.get("scale") or ("log" if inferred_log else "linear"))
    if scale not in {"linear", "log"}:
        raise ValueError(f"{name}: scale must be 'linear' or 'log'")

    # Schemas often expose a mathematically valid but unhelpfully broad LR
    # range such as [0, 1] around a 2e-4 default. Keep the default two orders
    # of magnitude inside the initial Bayesian domain unless the user supplied
    # an explicit override.
    if scale == "log" and "minimum" not in override and "valid_min" not in override:
        minimum = max(minimum, default_number / 100, 1e-12)
    if scale == "log" and "maximum" not in override and "valid_max" not in override:
        maximum = min(maximum, default_number * 100)

    if scale == "log" and minimum <= 0:
        raise ValueError(f"{name}: logarithmic ranges require minimum > 0")
    if maximum <= minimum:
        raise ValueError(f"{name}: maximum must be greater than minimum")
    return minimum, maximum, scale


def _constraint_metadata(node: dict, override: dict) -> dict:
    metadata = {}
    for key in ("math_cond", "depends_on", "parent_param"):
        value = override.get(key, node.get(key))
        if value not in (None, "", False):
            metadata[key] = value
    return metadata


def _build_parameter(
    name: str,
    schema: dict,
    base_spec: dict,
    overrides: dict[str, Any],
) -> dict:
    node = _schema_node(schema, name)
    override = overrides.get(name) or {}
    if not isinstance(override, dict):
        raise ValueError(f"Search-space override for {name!r} must be an object")

    default = override.get("default", node.get("default", _get_nested(base_spec, name)))
    raw_options = override.get(
        "options",
        override.get("valid_options", node.get("enum", node.get("valid_options"))),
    )
    raw_type = str(node.get("type", "")).lower()

    if raw_type in {"bool", "boolean"} or isinstance(default, bool):
        if not isinstance(default, bool):
            raise ValueError(f"{name}: boolean parameter has no boolean default")
        return {
            "name": name,
            "kind": "bool",
            "default": default,
            "scale": "linear",
            **_constraint_metadata(node, override),
        }
    if raw_options not in (None, ""):
        options = list(raw_options) if isinstance(raw_options, (list, tuple)) else [raw_options]
        if not options:
            raise ValueError(f"{name}: categorical options cannot be empty")
        if default is None:
            default = options[0]
        if default not in options:
            raise ValueError(f"{name}: default {default!r} is not in options")
        return {
            "name": name,
            "kind": "categorical",
            "default": copy.deepcopy(default),
            "options": copy.deepcopy(options),
            "scale": "linear",
            **_constraint_metadata(node, override),
        }

    if raw_type in {"int", "integer", "ordered_int"} or (
        isinstance(default, int) and not isinstance(default, bool)
    ):
        if default is None:
            raise ValueError(f"{name}: integer parameter has no default")
        minimum, maximum, scale = _derive_numeric_bounds(
            name, "int", int(default), node, override
        )
        integer_minimum = math.ceil(minimum)
        integer_maximum = math.floor(maximum)
        if integer_maximum <= integer_minimum:
            raise ValueError(
                f"{name}: integer range must contain at least two values"
            )
        return {
            "name": name,
            "kind": "int",
            "default": int(default),
            "minimum": integer_minimum,
            "maximum": integer_maximum,
            "scale": scale,
            **_constraint_metadata(node, override),
        }

    if raw_type in {"float", "number"} or isinstance(default, float):
        if default is None:
            raise ValueError(f"{name}: float parameter has no default")
        minimum, maximum, scale = _derive_numeric_bounds(
            name, "float", float(default), node, override
        )
        return {
            "name": name,
            "kind": "float",
            "default": float(default),
            "minimum": minimum,
            "maximum": maximum,
            "scale": scale,
            **_constraint_metadata(node, override),
        }

    raise ValueError(
        f"{name}: unsupported parameter type {raw_type!r}; provide options or a numeric/bool type"
    )


def _value_from_unit(parameter: dict, unit: float) -> Any:
    unit = min(1.0, max(0.0, float(unit)))
    kind = parameter["kind"]
    if kind == "bool":
        return unit >= 0.5
    if kind == "categorical":
        options = parameter["options"]
        index = min(int(unit * len(options)), len(options) - 1)
        return copy.deepcopy(options[index])

    minimum = float(parameter["minimum"])
    maximum = float(parameter["maximum"])
    math_cond = str(parameter.get("math_cond", "")).strip()
    if kind == "int" and not parameter.get("depends_on"):
        parts = math_cond.split()
        if len(parts) == 2 and parts[1].isdigit():
            operator, factor_text = parts
            factor = int(factor_text)
            if operator == "^" and factor > 1:
                powers = []
                power = 1
                while factor**power <= maximum:
                    value = factor**power
                    if value >= minimum:
                        powers.append(value)
                    power += 1
                if not powers:
                    raise ValueError(
                        f"{parameter['name']}: no power of {factor} in its range"
                    )
                return powers[min(int(unit * len(powers)), len(powers) - 1)]
            if operator == "/" and factor > 0:
                first = math.ceil(minimum / factor) * factor
                last = math.floor(maximum / factor) * factor
                if first > last:
                    raise ValueError(
                        f"{parameter['name']}: no multiple of {factor} in its range"
                    )
                count = ((last - first) // factor) + 1
                return int(first + min(int(unit * count), count - 1) * factor)
    if parameter["scale"] == "log":
        value = math.exp(math.log(minimum) + unit * (math.log(maximum) - math.log(minimum)))
    else:
        value = minimum + unit * (maximum - minimum)
    if kind == "int":
        return min(int(maximum), max(int(minimum), int(round(value))))
    return float(value)


def _constrain_relation(
    parameter: dict, value: int | float, parent: int | float, operator: str
) -> int | float:
    minimum = parameter["minimum"]
    maximum = parameter["maximum"]
    integer = parameter["kind"] == "int"
    epsilon = 1 if integer else max(abs(float(parent)) * 1e-9, 1e-12)
    if operator == ">" and value <= parent:
        value = max(minimum, parent + epsilon)
    elif operator == ">=" and value < parent:
        value = max(minimum, parent)
    elif operator == "<" and value >= parent:
        value = min(maximum, parent - epsilon)
    elif operator == "<=" and value > parent:
        value = min(maximum, parent)
    satisfied = {
        ">": value > parent,
        ">=": value >= parent,
        "<": value < parent,
        "<=": value <= parent,
    }[operator]
    if not satisfied or value < minimum or value > maximum:
        raise ValueError(
            f"{parameter['name']}: cannot satisfy {operator} "
            f"{parameter['depends_on']} within its search range"
        )
    return int(value) if integer else float(value)


def _apply_parameter_constraints(
    parameters: list[dict], values: dict, base_spec: dict, network_arch: str
) -> dict:
    by_name = {parameter["name"]: parameter for parameter in parameters}
    for parameter in parameters:
        name = parameter["name"]
        depends_on = parameter.get("depends_on")
        math_cond = str(parameter.get("math_cond", "")).strip()
        if depends_on and math_cond and parameter["kind"] in {"int", "float"}:
            parent = values.get(depends_on, _get_nested(base_spec, depends_on))
            if isinstance(parent, bool) or not isinstance(parent, (int, float)):
                raise ValueError(
                    f"{name}: dependency {depends_on!r} is not numeric"
                )
            parts = math_cond.split()
            if len(parts) >= 2 and parts[1] == "depends_on":
                if parts[0] not in {">", ">=", "<", "<="}:
                    raise ValueError(f"{name}: unsupported relation {math_cond!r}")
                values[name] = _constrain_relation(
                    parameter, values[name], parent, parts[0]
                )
            elif len(parts) == 2 and parts[0] == "/" and parts[1].isdigit():
                factor = int(parts[1])
                if factor < 1:
                    raise ValueError(f"{name}: invalid dependency factor {factor}")
                calculated = math.floor(float(parent) / factor)
                values[name] = min(
                    int(parameter["maximum"]),
                    max(int(parameter["minimum"]), calculated),
                )

    normalized_network = re.sub(r"[^a-z0-9]", "", network_arch.lower())
    detr_networks = {
        "dino",
        "groundingdino",
        "deformabledetr",
        "rtdetr",
    }
    if normalized_network in detr_networks:
        for name, parameter in by_name.items():
            if not name.endswith(".num_select") or parameter["kind"] != "int":
                continue
            parent_name = parameter.get("depends_on") or name.replace(
                ".num_select", ".num_queries"
            )
            parent = values.get(parent_name, _get_nested(base_spec, parent_name))
            if isinstance(parent, int) and not isinstance(parent, bool):
                values[name] = min(values[name], parent)
    return values


def _parameter_values(
    parameters: list[dict],
    vector: list[float],
    *,
    base_spec: dict,
    network_arch: str,
) -> dict:
    values = {
        parameter["name"]: _value_from_unit(parameter, unit)
        for parameter, unit in zip(parameters, vector)
    }
    return _apply_parameter_constraints(parameters, values, base_spec, network_arch)


def _parameter_signature(values: dict) -> bytes:
    return _canonical_bytes(values)


def _bayesian_vector(state: dict) -> list[float]:
    try:
        import numpy as np
        from scipy.special import ndtr
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    except ImportError as exc:  # pragma: no cover - dependency preflight path
        raise SystemExit(
            "Bayesian recommendation requires numpy, scipy, and scikit-learn. "
            "Install them with: python -m pip install numpy scipy scikit-learn"
        ) from exc

    index = len(state["recommendations"])
    dimensions = len(state["search_parameters"])
    rng = np.random.default_rng(state["seed"] + index * 104729)
    candidates = rng.random((state["candidate_count"], dimensions))
    successful = [
        rec
        for rec in state["recommendations"]
        if rec["state"] == "SUCCEEDED"
        and (rec.get("objective_score") is not None or rec["metric"] is not None)
    ]

    if len(successful) >= 2:
        observed_x = np.asarray([rec["vector"] for rec in successful], dtype=float)
        observed_y = np.asarray(
            [
                rec["objective_score"]
                if rec.get("objective_score") is not None
                else (
                    rec["metric"]
                    if state["direction"] == "maximize"
                    else -rec["metric"]
                )
                for rec in successful
            ],
            dtype=float,
        )
        kernel = (
            ConstantKernel(1.0, constant_value_bounds="fixed")
            * Matern(
                length_scale=np.ones(dimensions),
                length_scale_bounds="fixed",
                nu=2.5,
            )
            + WhiteKernel(noise_level=1e-6, noise_level_bounds="fixed")
        )
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-8,
            optimizer=None,
            normalize_y=True,
            random_state=state["seed"],
        )
        model.fit(observed_x, observed_y)
        mean, stddev = model.predict(candidates, return_std=True)
        improvement = mean - float(np.max(observed_y)) - 0.01
        safe_stddev = np.maximum(stddev, 1e-12)
        z_score = improvement / safe_stddev
        expected_improvement = improvement * ndtr(z_score) + safe_stddev * (
            np.exp(-0.5 * z_score**2) / math.sqrt(2 * math.pi)
        )
        expected_improvement[stddev <= 1e-12] = 0.0
        candidate_order = np.argsort(-expected_improvement, kind="stable")
    else:
        candidate_order = np.arange(len(candidates))

    used = {
        _parameter_signature(rec["parameters"])
        for rec in state["recommendations"]
    }
    for candidate_index in candidate_order:
        vector = [float(value) for value in candidates[int(candidate_index)]]
        values = _parameter_values(
            state["search_parameters"],
            vector,
            base_spec=state["base_spec"],
            network_arch=state["network_arch"],
        )
        if _parameter_signature(values) not in used:
            return vector
    raise RuntimeError("Candidate pool exhausted without finding a unique recommendation")


def _active_recommendation(state: dict) -> dict | None:
    return next(
        (rec for rec in state["recommendations"] if rec["state"] in {"READY", "RUNNING"}),
        None,
    )


def _active_recommendations(state: dict) -> list[dict]:
    return [
        rec
        for rec in state["recommendations"]
        if rec["state"] in {"READY", "RUNNING"}
    ]


def _random_vector(state: dict, salt: int = 0) -> list[float]:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("AutoML requires numpy: python -m pip install numpy") from exc
    dimensions = len(state["search_parameters"])
    rng = np.random.default_rng(
        state["seed"] + (len(state["recommendations"]) + salt) * 104729
    )
    used = {
        _parameter_signature(rec["parameters"])
        for rec in state["recommendations"]
    }
    for vector_array in rng.random((state["candidate_count"], dimensions)):
        vector = [float(value) for value in vector_array]
        values = _parameter_values(
            state["search_parameters"],
            vector,
            base_spec=state["base_spec"],
            network_arch=state["network_arch"],
        )
        if _parameter_signature(values) not in used:
            return vector
    raise RuntimeError("Candidate pool exhausted without a unique random recommendation")


def _bfbo_vector(state: dict) -> list[float]:
    """Batch-friendly GP-UCB recommendation with local running-point penalties."""
    try:
        import numpy as np
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "BFBO requires numpy and scikit-learn. Install them with: "
            "python -m pip install numpy scikit-learn"
        ) from exc
    successful = [
        rec
        for rec in state["recommendations"]
        if rec["state"] == "SUCCEEDED" and rec.get("objective_score") is not None
    ]
    if len(successful) < 2:
        return _random_vector(state)
    dimensions = len(state["search_parameters"])
    rng = np.random.default_rng(state["seed"] + len(state["recommendations"]) * 65537)
    candidates = rng.random((state["candidate_count"], dimensions))
    observed_x = np.asarray([rec["vector"] for rec in successful], dtype=float)
    observed_y = np.asarray([rec["objective_score"] for rec in successful], dtype=float)
    model = GaussianProcessRegressor(
        kernel=(
            ConstantKernel(1.0, constant_value_bounds="fixed")
            * RBF(np.ones(dimensions), length_scale_bounds="fixed")
            + WhiteKernel(1e-6, noise_level_bounds="fixed")
        ),
        alpha=1e-8,
        optimizer=None,
        normalize_y=True,
        random_state=state["seed"],
    )
    model.fit(observed_x, observed_y)
    mean, stddev = model.predict(candidates, return_std=True)
    acquisition = mean + 2.0 * stddev
    running = [
        rec["vector"]
        for rec in state["recommendations"]
        if rec["state"] in {"READY", "RUNNING"}
    ]
    for vector in running:
        distance = np.linalg.norm(candidates - np.asarray(vector), axis=1)
        acquisition *= 1.0 - np.exp(-(distance**2) / 0.02)
    used = {
        _parameter_signature(rec["parameters"])
        for rec in state["recommendations"]
    }
    for candidate_index in np.argsort(-acquisition, kind="stable"):
        vector = [float(value) for value in candidates[int(candidate_index)]]
        values = _parameter_values(
            state["search_parameters"],
            vector,
            base_spec=state["base_spec"],
            network_arch=state["network_arch"],
        )
        if _parameter_signature(values) not in used:
            return vector
    return _random_vector(state, salt=1)


def _bohb_vector(state: dict) -> list[float]:
    """Sample by a TPE-style good/bad density ratio after a random warm-up."""
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("BOHB requires numpy: python -m pip install numpy") from exc
    successful = [
        rec
        for rec in state["recommendations"]
        if rec["state"] == "SUCCEEDED" and rec.get("objective_score") is not None
    ]
    minimum = int(state["algorithm_config"]["min_points_in_model"])
    if len(successful) < minimum:
        return _random_vector(state)
    ordered = sorted(successful, key=lambda rec: rec["objective_score"], reverse=True)
    good_count = max(1, int(math.ceil(len(ordered) * state["algorithm_config"]["top_fraction"])))
    good = np.asarray([rec["vector"] for rec in ordered[:good_count]], dtype=float)
    bad_rows = ordered[good_count:] or ordered[-1:]
    bad = np.asarray([rec["vector"] for rec in bad_rows], dtype=float)
    good_bandwidth = np.maximum(np.std(good, axis=0), 0.05)
    bad_bandwidth = np.maximum(np.std(bad, axis=0), 0.05)
    rng = np.random.default_rng(state["seed"] + len(state["recommendations"]) * 12289)
    sample_count = int(state["algorithm_config"]["kde_samples"])
    anchors = good[rng.integers(0, len(good), size=sample_count)]
    candidates = np.clip(
        anchors + rng.normal(size=anchors.shape) * good_bandwidth,
        0.0,
        1.0,
    )

    def log_density(points, observations, bandwidth):
        scaled = (points[:, None, :] - observations[None, :, :]) / bandwidth
        component = -0.5 * np.sum(scaled**2, axis=2) - np.sum(
            np.log(bandwidth)
        )
        maximum = np.max(component, axis=1)
        return maximum + np.log(
            np.mean(np.exp(component - maximum[:, None]), axis=1)
        )

    ratio = log_density(candidates, good, good_bandwidth) - log_density(
        candidates, bad, bad_bandwidth
    )
    used = {
        _parameter_signature(rec["parameters"])
        for rec in state["recommendations"]
    }
    for candidate_index in np.argsort(-ratio, kind="stable"):
        vector = [float(value) for value in candidates[int(candidate_index)]]
        values = _parameter_values(
            state["search_parameters"],
            vector,
            base_spec=state["base_spec"],
            network_arch=state["network_arch"],
        )
        if _parameter_signature(values) not in used:
            return vector
    return _random_vector(state, salt=3)


def _dehb_vector(state: dict) -> list[float]:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("DEHB requires numpy: python -m pip install numpy") from exc
    completed = [
        rec
        for rec in state["recommendations"]
        if rec["state"] == "SUCCEEDED" and rec.get("parent_rec_id") is None
    ]
    if len(completed) < 3:
        return _random_vector(state)
    rng = np.random.default_rng(state["seed"] + len(state["recommendations"]) * 8191)
    picked = rng.choice(len(completed), size=3, replace=False)
    a, b, c = [np.asarray(completed[int(index)]["vector"]) for index in picked]
    factor = state["algorithm_config"]["mutation_factor"]
    crossover = state["algorithm_config"]["crossover_probability"]
    mutant = np.clip(a + factor * (b - c), 0.0, 1.0)
    base = np.asarray(_random_vector(state, salt=2))
    mask = rng.random(len(base)) < crossover
    mask[int(rng.integers(0, len(base)))] = True
    return [float(value) for value in np.where(mask, mutant, base)]


def _objective_score(state: dict, values: dict[str, float]) -> float:
    score = 0.0
    for objective in state.get("objectives") or [
        {
            "name": state["metric_name"],
            "direction": state["direction"],
            "weight": 1.0,
            "scale": 1.0,
        }
    ]:
        if objective["name"] not in values:
            raise ValueError(f"Missing objective metric {objective['name']!r}")
        value = _finite_number(values[objective["name"]], objective["name"])
        contribution = objective["weight"] * value / objective["scale"]
        score += contribution if objective["direction"] == "maximize" else -contribution
    return float(score)


def _budget_key(state: dict) -> str | None:
    for key in BUDGET_KEYS:
        if _get_nested(state["base_spec"], key) is not None:
            return key
    return None


def _budget_ladder(state: dict) -> list[int]:
    config = state["algorithm_config"]
    base_maximum = int(config["max_epochs"])
    multiplier = int(config["epoch_multiplier"])
    reduction = int(config["reduction_factor"])
    exponent = (
        int(math.floor(math.log(base_maximum, reduction)))
        if base_maximum > 1
        else 0
    )
    maximum = base_maximum * multiplier
    value = max(1, int(base_maximum / (reduction**exponent))) * multiplier
    ladder = []
    while value < maximum:
        ladder.append(value)
        value = min(maximum, value * reduction)
    ladder.append(maximum)
    return list(dict.fromkeys(ladder))


def _hyperband_brackets(state: dict) -> list[dict]:
    maximum = int(state["algorithm_config"]["max_epochs"])
    multiplier = int(state["algorithm_config"]["epoch_multiplier"])
    reduction = int(state["algorithm_config"]["reduction_factor"])
    if maximum <= 1:
        return [{"index": 0, "initial_count": 1, "budgets": [maximum * multiplier]}]
    s_max = int(math.floor(math.log(maximum, reduction)))
    brackets = []
    for index, s_value in enumerate(range(s_max, 0, -1)):
        initial_count = int(
            math.ceil(((s_max + 1) / (s_value + 1)) * (reduction**s_value))
        )
        initial_budget = max(1, int(maximum / (reduction**s_value)))
        budgets = [
            min(maximum, initial_budget * (reduction**rung)) * multiplier
            for rung in range(s_value + 1)
        ]
        brackets.append(
            {
                "index": index,
                "initial_count": initial_count,
                "budgets": list(dict.fromkeys(budgets)),
            }
        )
    return brackets or [
        {"index": 0, "initial_count": 1, "budgets": [maximum * multiplier]}
    ]


def _ranked(state: dict, recs: list[dict]) -> list[dict]:
    return sorted(
        [rec for rec in recs if rec["state"] == "SUCCEEDED"],
        key=lambda rec: rec.get("objective_score", float("-inf")),
        reverse=True,
    )


def _extract_json(text: str) -> dict:
    stripped = text.strip()
    for candidate in (stripped,):
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
    for marker in ("```json", "```"):
        if marker not in stripped:
            continue
        start = stripped.index(marker) + len(marker)
        end = stripped.find("```", start)
        candidate = stripped[start : end if end >= 0 else None].strip()
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
    decoder = json.JSONDecoder()
    for index, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("LLM response did not contain a JSON object")


def _llm_history(state: dict) -> list[dict]:
    return [
        {
            "rec_id": rec["id"],
            "state": rec["state"],
            "parameters": rec["parameters"],
            "objectives": rec.get("objective_values", {}),
            "score": rec.get("objective_score"),
            "failure": rec.get("failure"),
            "observations": rec.get("observations", []),
            "feedback": rec.get("feedback", []),
        }
        for rec in state["recommendations"]
        if rec["state"] in TERMINAL_REC_STATES
    ]


def _llm_call_json(state: dict, purpose: str, instructions: str) -> tuple[dict, dict]:
    config = state["algorithm_config"]
    endpoint = str(config.get("llm_endpoint") or "").rstrip("/")
    model = str(config.get("llm_model") or "")
    if not endpoint or not model:
        raise RuntimeError(
            f"{state['algorithm']} requires --llm-endpoint and --llm-model"
        )
    url = endpoint if endpoint.endswith("/chat/completions") else f"{endpoint}/chat/completions"
    parameter_contract = [
        {
            "name": parameter["name"],
            "kind": parameter["kind"],
            "minimum": parameter.get("minimum"),
            "maximum": parameter.get("maximum"),
            "options": parameter.get("options"),
            "scale": parameter["scale"],
        }
        for parameter in state["search_parameters"]
    ]
    request_payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a deterministic AutoML strategist. Return one JSON object "
                    "only. Never return executable code or values outside the supplied "
                    "normalized [0,1] domain."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "purpose": purpose,
                        "instructions": instructions,
                        "network": state["network_arch"],
                        "action": state["action"],
                        "objectives": state["objectives"],
                        "parameters": parameter_contract,
                        "history": _llm_history(state),
                        "research_program": config.get("research_program"),
                        "evolvable_text_parameters": config.get(
                            "evolvable_text_parameters", []
                        ),
                    },
                    sort_keys=True,
                    allow_nan=False,
                ),
            },
        ],
        "temperature": config["llm_temperature"],
        "max_tokens": config["llm_max_tokens"],
        "response_format": {"type": "json_object"},
    }
    api_key = os.getenv("AUTOML_LLM_API_KEY") or os.getenv("NVIDIA_API_KEY")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        url,
        data=_canonical_bytes(request_payload),
        headers=headers,
        method="POST",
    )
    started = time.monotonic()
    error = "unknown error"
    for attempt in range(1, int(config["llm_max_retries"]) + 1):
        try:
            with urllib.request.urlopen(
                request,
                timeout=float(config["llm_timeout"]),
            ) as response:
                envelope = json.loads(response.read().decode("utf-8"))
            content = envelope["choices"][0]["message"]["content"]
            value = content if isinstance(content, dict) else _extract_json(str(content))
            usage = envelope.get("usage") or {}
            return value, {
                "provider": "openai_compatible",
                "model": model,
                "purpose": purpose,
                "attempts": attempt,
                "latency_ms": round((time.monotonic() - started) * 1000, 1),
                "usage": {
                    key: int(usage[key])
                    for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                    if isinstance(usage.get(key), (int, float))
                },
            }
        except (
            OSError,
            KeyError,
            IndexError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            error = str(exc)
            if api_key:
                error = error.replace(api_key, "[REDACTED]")
            if attempt < int(config["llm_max_retries"]):
                time.sleep(float(config["llm_retry_delay"]) * attempt)
    raise RuntimeError(error[:500])


def _validate_llm_vector(state: dict, response: dict) -> tuple[list[float], dict]:
    raw_vector = response.get("normalized_vector")
    if not isinstance(raw_vector, list) or len(raw_vector) != len(
        state["search_parameters"]
    ):
        raise ValueError(
            "LLM response normalized_vector has the wrong number of dimensions"
        )
    vector = []
    for index, value in enumerate(raw_vector):
        number = _finite_number(value, f"normalized_vector[{index}]")
        if number < 0 or number > 1:
            raise ValueError("LLM normalized_vector values must be within [0,1]")
        vector.append(number)
    value_overrides = {}
    allowed = set(state["algorithm_config"].get("evolvable_text_parameters", []))
    proposed_text = response.get("evolvable_text") or {}
    if not isinstance(proposed_text, dict):
        raise ValueError("LLM evolvable_text must be an object")
    for name, value in proposed_text.items():
        if name not in allowed:
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Evolvable text {name!r} must be a non-empty string")
        value_overrides[name] = value
    metadata = {
        "reason": str(response.get("reason") or "")[:2000],
        "value_overrides": value_overrides,
    }
    return vector, metadata


def _llm_vector(state: dict, purpose: str) -> tuple[list[float], dict]:
    try:
        response, call_metadata = _llm_call_json(
            state,
            purpose,
            (
                "Propose normalized_vector with exactly one number per parameter, "
                "optional evolvable_text, and a concise reason. Use experiment "
                "results and failure feedback to improve over prior candidates."
            ),
        )
        vector, proposal_metadata = _validate_llm_vector(state, response)
        return vector, {
            "proposer": purpose,
            "llm": {**call_metadata, **proposal_metadata, "fallback": False},
            "value_overrides": proposal_metadata["value_overrides"],
        }
    except (RuntimeError, ValueError) as exc:
        vector = _random_vector(state, salt=97)
        return vector, {
            "proposer": f"{purpose}_fallback",
            "llm": {
                "purpose": purpose,
                "fallback": True,
                "error": str(exc)[:500],
            },
        }


def _vector_for_algorithm(state: dict) -> tuple[list[float], dict]:
    algorithm = state["algorithm"]
    if algorithm == "bayesian":
        return _bayesian_vector(state), {"proposer": "gp_expected_improvement"}
    if algorithm == "bfbo":
        return _bfbo_vector(state), {"proposer": "gp_ucb_local_penalization"}
    if algorithm == "bohb":
        successful = sum(rec["state"] == "SUCCEEDED" for rec in state["recommendations"])
        if successful >= state["algorithm_config"]["min_points_in_model"]:
            return _bohb_vector(state), {"proposer": "bohb_tpe_density_ratio"}
    if algorithm == "dehb":
        return _dehb_vector(state), {"proposer": "differential_evolution"}
    if algorithm == "llm":
        return _llm_vector(state, "llm_proposal")
    if algorithm == "hybrid":
        limit = state["algorithm_config"]["max_experiments"]
        phase = (
            "hybrid_refinement"
            if len(state["recommendations"]) >= max(2, limit // 2)
            else "hybrid_exploration"
        )
        try:
            response, llm_metadata = _llm_call_json(
                state,
                phase,
                (
                    "Choose algorithm as bayesian or bfbo for this phase and give a "
                    "concise reason. Return {algorithm, reason}. Reserve at least 20% "
                    "of the experiment budget for refinement."
                ),
            )
            chosen = response.get("algorithm")
            if chosen not in {"bayesian", "bfbo"}:
                raise ValueError("hybrid LLM chose an unsupported sub-algorithm")
            vector = _bayesian_vector(state) if chosen == "bayesian" else _bfbo_vector(state)
            return vector, {
                "proposer": phase,
                "sub_algorithm": chosen,
                "llm": {
                    **llm_metadata,
                    "reason": str(response.get("reason") or "")[:2000],
                    "fallback": False,
                },
            }
        except (RuntimeError, ValueError) as exc:
            vector = _bayesian_vector(state) if phase.endswith("refinement") else _bfbo_vector(state)
            return vector, {
                "proposer": f"{phase}_fallback",
                "sub_algorithm": "bayesian" if phase.endswith("refinement") else "bfbo",
                "llm": {"fallback": True, "error": str(exc)[:500]},
            }
    if algorithm == "autoresearch":
        return _llm_vector(state, "autoresearch_reflection")
    return _random_vector(state), {"proposer": "random"}


def _find_recommendation(state: dict, rec_id: str) -> dict:
    rec = next((item for item in state["recommendations"] if item["id"] == rec_id), None)
    if rec is None:
        raise SystemExit(f"Unknown recommendation: {rec_id}")
    return rec


def _refresh_best(state: dict) -> None:
    successful = [rec for rec in state["recommendations"] if rec["state"] == "SUCCEEDED"]
    if not successful:
        state["best_rec_id"] = None
        return
    if state["algorithm"] in BUDGETED_ALGORITHMS:
        highest_budget = max(rec.get("budget") or 0 for rec in successful)
        successful = [
            rec for rec in successful if (rec.get("budget") or 0) == highest_budget
        ]
    state["best_rec_id"] = max(
        successful,
        key=lambda rec: rec.get("objective_score", float("-inf")),
    )["id"]


def _maybe_complete(state: dict) -> None:
    if state["algorithm"] in BUDGETED_ALGORITHMS:
        return
    limit = (
        state["algorithm_config"]["max_experiments"]
        if state["algorithm"] in {"hybrid", "autoresearch"}
        else state["max_recommendations"]
    )
    if (
        len(state["recommendations"]) >= limit
        and all(rec["state"] in TERMINAL_REC_STATES for rec in state["recommendations"])
    ):
        state["status"] = "COMPLETE"


def _pareto_front(state: dict) -> list[dict]:
    objectives = state.get("objectives") or []
    if len(objectives) < 2:
        return []
    candidates = [
        rec
        for rec in state["recommendations"]
        if rec["state"] == "SUCCEEDED"
        and all(
            objective["name"] in rec.get("objective_values", {})
            for objective in objectives
        )
    ]

    def dominates(left: dict, right: dict) -> bool:
        no_worse = True
        strictly_better = False
        for objective in objectives:
            name = objective["name"]
            left_value = left["objective_values"][name]
            right_value = right["objective_values"][name]
            if objective["direction"] == "maximize":
                no_worse = no_worse and left_value >= right_value
                strictly_better = strictly_better or left_value > right_value
            else:
                no_worse = no_worse and left_value <= right_value
                strictly_better = strictly_better or left_value < right_value
        return no_worse and strictly_better

    front = [
        rec
        for rec in candidates
        if not any(
            other is not rec and dominates(other, rec)
            for other in candidates
        )
    ]
    return [
        {
            "rec_id": rec["id"],
            "objective_values": rec["objective_values"],
            "objective_score": rec["objective_score"],
            "artifact_uri": rec.get("artifact_uri"),
            "budget": rec.get("budget"),
        }
        for rec in sorted(front, key=lambda item: item["index"])
    ]


def _summary(state: dict) -> dict:
    counts = {
        status.lower(): sum(rec["state"] == status for rec in state["recommendations"])
        for status in ("READY", "RUNNING", "SUCCEEDED", "FAILED", "CANCELED")
    }
    best = None
    if state["best_rec_id"] is not None:
        rec = _find_recommendation(state, state["best_rec_id"])
        best = {
            "rec_id": rec["id"],
            "metric": rec["metric"],
            "objective_values": rec.get("objective_values", {}),
            "objective_score": rec.get("objective_score"),
            "parameters": rec["parameters"],
            "checkpoint_uri": rec["checkpoint_uri"],
            "artifact_uri": rec.get("artifact_uri"),
            "budget": rec.get("budget"),
        }
    return {
        "experiment_id": state["id"],
        "status": state["status"],
        "algorithm": state["algorithm"],
        "metric_name": state["metric_name"],
        "direction": state["direction"],
        "objectives": state.get("objectives", []),
        "max_recommendations": state["max_recommendations"],
        "generated": len(state["recommendations"]),
        "counts": counts,
        "best": best,
        "pareto_front": _pareto_front(state),
    }


def cmd_init(args: argparse.Namespace) -> int:
    state_path = args.state.resolve()
    schema_path = args.schema.resolve()
    schema = _read_document(schema_path)
    if not isinstance(schema, dict):
        raise SystemExit(f"Search schema must be a JSON/YAML object: {schema_path}")
    base_spec = (
        _read_document(args.base_spec.resolve())
        if args.base_spec
        else copy.deepcopy(schema.get("default") or {})
    )
    if not isinstance(base_spec, dict):
        raise SystemExit("The base spec must be an object")
    search_overrides = (
        _read_document(args.search_space.resolve()) if args.search_space else {}
    )
    if not isinstance(search_overrides, dict):
        raise SystemExit("The search-space override must be an object keyed by parameter name")

    try:
        names = _parameter_names(schema, args.parameter)
        parameters = [
            _build_parameter(name, schema, base_spec, search_overrides) for name in names
        ]
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    experiment_id = args.experiment_id or (
        f"automl-{re.sub(r'[^A-Za-z0-9._-]+', '-', args.network_arch)}-"
        f"{uuid.uuid4().hex[:8]}"
    )
    if not ID_RE.fullmatch(experiment_id):
        raise SystemExit(f"Invalid experiment id: {experiment_id!r}")

    objectives = list(args.objective or [])
    if not objectives:
        objectives = [
            {
                "name": args.metric,
                "direction": args.direction,
                "weight": 1.0,
                "scale": 1.0,
            }
        ]
    elif args.metric and objectives[0]["name"] != args.metric:
        raise SystemExit("--metric must match the first --objective name")
    primary = objectives[0]
    algorithm_config = _algorithm_config(args)
    if args.algorithm in {"llm", "hybrid", "autoresearch"} and (
        not algorithm_config["llm_endpoint"] or not algorithm_config["llm_model"]
    ):
        raise SystemExit(
            f"{args.algorithm} requires --llm-endpoint and --llm-model "
            "(or AUTOML_LLM_ENDPOINT and AUTOML_LLM_MODEL)"
        )
    parameters_by_name = {parameter["name"]: parameter for parameter in parameters}
    for name in algorithm_config["evolvable_text_parameters"]:
        parameter = parameters_by_name.get(name)
        if parameter is None:
            raise SystemExit(f"Unknown evolvable text parameter: {name}")
        if parameter["kind"] != "categorical" or not all(
            isinstance(option, str) for option in parameter.get("options", [])
        ):
            raise SystemExit(
                f"Evolvable text parameter {name!r} must be string-valued categorical"
            )

    created_at = _now()
    state = {
        "schema_version": 1,
        "id": experiment_id,
        "status": "ACTIVE",
        "algorithm": args.algorithm,
        "algorithm_version": "skill-v2",
        "seed": args.seed,
        "network_arch": args.network_arch,
        "action": args.action,
        "metric_name": primary["name"],
        "direction": primary["direction"],
        "objectives": objectives,
        "max_recommendations": args.max_recommendations,
        "max_concurrent": args.max_concurrent,
        "algorithm_config": algorithm_config,
        "tracking": {
            "wandb_enabled": args.wandb,
            "wandb_project": args.wandb_project,
            "wandb_entity": args.wandb_entity,
            "wandb_mode": args.wandb_mode,
        },
        "candidate_count": args.candidate_count,
        "search_schema_source": str(schema_path),
        # Hash the packaged source bytes rather than the parsed object. TAO
        # schemas may contain Python-compatible Infinity bounds; those are
        # normalized away when deriving the finite search domain but cannot be
        # emitted by strict JSON canonicalization.
        "search_schema_sha256": hashlib.sha256(schema_path.read_bytes()).hexdigest(),
        "base_spec_sha256": _sha256(base_spec),
        "base_spec": base_spec,
        "search_parameters": parameters,
        "recommendations": [],
        "best_rec_id": None,
        "created_at": created_at,
        "updated_at": created_at,
    }

    with _state_lock(state_path):
        if state_path.exists():
            raise SystemExit(
                f"Refusing to overwrite existing AutoML state: {state_path}. "
                "Use status/recommend to resume it."
            )
        _write_state(state_path, state)
    _print({"state": str(state_path), **_summary(state), "search_parameters": parameters})
    return 0


def _new_recommendation(
    state: dict,
    *,
    vector: list[float] | None = None,
    parent: dict | None = None,
    budget: int | None = None,
    rung: int | None = None,
    generation: int | None = None,
    bracket: int | None = None,
    config_id: str | None = None,
    metadata: dict | None = None,
) -> dict:
    if vector is None:
        if parent is None:
            raise ValueError("A new recommendation needs a vector or parent")
        vector = list(parent["vector"])
    values = _parameter_values(
        state["search_parameters"],
        vector,
        base_spec=state["base_spec"],
        network_arch=state["network_arch"],
    )
    for name, value in (metadata or {}).get("value_overrides", {}).items():
        values[name] = value
    overrides: dict[str, Any] = {}
    spec = copy.deepcopy(state["base_spec"])
    for name, value in values.items():
        _set_nested(overrides, name, value)
        _set_nested(spec, name, value)
    if budget is not None:
        budget_key = _budget_key(state)
        if budget_key is None:
            raise ValueError(
                f"{state['algorithm']} requires an epoch/budget key in the base spec"
            )
        _set_nested(overrides, budget_key, budget)
        _set_nested(spec, budget_key, budget)
        for interval_name in (
            "train.checkpoint_interval",
            "train.validation_interval",
            "train.validation_freq",
        ):
            if _get_nested(spec, interval_name) is not None:
                interval = (
                    state["algorithm_config"]["eval_interval"]
                    if state["algorithm"] == "pbt"
                    else budget
                )
                _set_nested(overrides, interval_name, interval)
                _set_nested(spec, interval_name, interval)
    index = len(state["recommendations"])
    if config_id is None:
        config_id = parent["config_id"] if parent else f"config-{index:04d}"
    resume_job = None
    if parent and parent["attempts"]:
        resume_job = parent["attempts"][-1]["job_id"]
    recommendation = {
        "id": f"rec-{index:04d}",
        "index": index,
        "config_id": config_id,
        "state": "READY",
        "generated_at": _now(),
        "vector": vector,
        "parameters": values,
        "spec_overrides": overrides,
        "spec": spec,
        "attempts": [],
        "metric": None,
        "objective_values": {},
        "objective_score": None,
        "checkpoint_uri": None,
        "artifact_uri": None,
        "failure": None,
        "budget": budget,
        "rung": rung,
        "generation": generation,
        "bracket": bracket,
        "parent_rec_id": parent["id"] if parent else None,
        "resume_from_job_id": resume_job,
        "observations": [],
        "feedback": [],
        "decision": None,
        "metadata": dict(metadata or {}),
    }
    state["recommendations"].append(recommendation)
    return recommendation


def _schedule_successive_halving(state: dict, slots: int) -> list[dict]:
    reduction = int(state["algorithm_config"]["reduction_factor"])
    created = []
    brackets = _hyperband_brackets(state)
    for bracket in brackets:
        bracket_index = bracket["index"]
        budgets = bracket["budgets"]
        bracket_recs = [
            rec
            for rec in state["recommendations"]
            if rec.get("bracket") == bracket_index
        ]
        first_rung = [rec for rec in bracket_recs if rec.get("rung") == 0]
        while (
            len(first_rung) < bracket["initial_count"]
            and len(created) < slots
        ):
            vector, metadata = _vector_for_algorithm(state)
            rec = _new_recommendation(
                state,
                vector=vector,
                budget=budgets[0],
                rung=0,
                bracket=bracket_index,
                metadata={**metadata, "hyperband_bracket": bracket_index},
            )
            first_rung.append(rec)
            bracket_recs.append(rec)
            created.append(rec)
        if created:
            return created
        if len(first_rung) < bracket["initial_count"]:
            return []

        bracket_exhausted = False
        for rung, budget in enumerate(budgets[:-1]):
            rung_recs = [rec for rec in bracket_recs if rec.get("rung") == rung]
            if any(rec["state"] not in TERMINAL_REC_STATES for rec in rung_recs):
                return []
            pending = [
                rec
                for rec in rung_recs
                if rec.get("decision") == "PROMOTE_PENDING"
            ]
            if (
                not pending
                and rung_recs
                and all(rec.get("decision") is None for rec in rung_recs)
            ):
                winners = _ranked(state, rung_recs)[:
                    max(1, len(rung_recs) // reduction)
                ]
                winner_ids = {rec["id"] for rec in winners}
                for rec in rung_recs:
                    rec["decision"] = (
                        "PROMOTE_PENDING"
                        if rec["id"] in winner_ids
                        else "STOPPED"
                    )
                pending = winners
                if not winners:
                    bracket_exhausted = True
                    break
            for parent in pending[: max(0, slots - len(created))]:
                parent["decision"] = "PROMOTED"
                created.append(
                    _new_recommendation(
                        state,
                        parent=parent,
                        budget=budgets[rung + 1],
                        rung=rung + 1,
                        bracket=bracket_index,
                        metadata={
                            "promotion_from_budget": budget,
                            "hyperband_bracket": bracket_index,
                        },
                    )
                )
            if created or pending:
                return created

        final_recs = [
            rec
            for rec in bracket_recs
            if rec.get("rung") == len(budgets) - 1
        ]
        bracket_complete = bracket_exhausted or (
            bool(final_recs)
            and all(rec["state"] in TERMINAL_REC_STATES for rec in final_recs)
        )
        if not bracket_complete:
            return []
        for rec in final_recs:
            if rec.get("decision") is None:
                rec["decision"] = "BRACKET_COMPLETE"

    state["status"] = "COMPLETE"
    return []


def _schedule_asha(state: dict, slots: int) -> list[dict]:
    ladder = _budget_ladder(state)
    configured_limit = state["algorithm_config"].get("max_trials")
    trial_limit = int(configured_limit) if configured_limit is not None else None
    final_target = int(state["algorithm_config"]["min_top_configs"])
    reduction = int(state["algorithm_config"]["reduction_factor"])
    created = []

    for rung in range(len(ladder) - 1):
        rung_recs = [
            rec for rec in state["recommendations"] if rec.get("rung") == rung
        ]
        completed = [
            rec for rec in rung_recs if rec["state"] in TERMINAL_REC_STATES
        ]
        quota = len(completed) // reduction
        already_promoted = sum(rec.get("decision") == "PROMOTED" for rec in rung_recs)
        if quota > already_promoted:
            winners = _ranked(state, completed)[:quota]
            for winner in winners:
                if winner.get("decision") != "PROMOTED":
                    winner["decision"] = "PROMOTE_PENDING"
            winner_ids = {winner["id"] for winner in winners}
            for rec in completed:
                if rec["id"] not in winner_ids and rec.get("decision") is None:
                    rec["decision"] = "WAITING_FOR_QUOTA"

    pending = [
        rec
        for rec in state["recommendations"]
        if rec.get("decision") == "PROMOTE_PENDING"
    ]
    for parent in pending[:slots]:
        parent["decision"] = "PROMOTED"
        created.append(
            _new_recommendation(
                state,
                parent=parent,
                budget=ladder[int(parent["rung"]) + 1],
                rung=int(parent["rung"]) + 1,
                metadata={"asynchronous_promotion": True},
            )
        )
    if len(created) >= slots:
        return created

    final_successes = [
        rec
        for rec in state["recommendations"]
        if rec.get("rung") == len(ladder) - 1 and rec["state"] == "SUCCEEDED"
    ]
    target_reached = len(final_successes) >= final_target
    first_rung = [rec for rec in state["recommendations"] if rec.get("rung") == 0]
    while (
        not target_reached
        and (trial_limit is None or len(first_rung) < trial_limit)
        and len(created) < slots
    ):
        vector, metadata = _vector_for_algorithm(state)
        rec = _new_recommendation(
            state,
            vector=vector,
            budget=ladder[0],
            rung=0,
            metadata={**metadata, "asynchronous_trial": True},
        )
        first_rung.append(rec)
        created.append(rec)
    if created:
        return created

    active = _active_recommendations(state)
    pending = any(
        rec.get("decision") == "PROMOTE_PENDING"
        for rec in state["recommendations"]
    )
    exhausted = trial_limit is not None and len(first_rung) >= trial_limit
    if (target_reached or exhausted) and not active and not pending:
        state["status"] = "COMPLETE"
        for rec in state["recommendations"]:
            if rec.get("decision") == "WAITING_FOR_QUOTA":
                rec["decision"] = "STOPPED"
    return []


def _perturb_vector(state: dict, vector: list[float], salt: int) -> list[float]:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("PBT requires numpy: python -m pip install numpy") from exc
    rng = np.random.default_rng(state["seed"] + salt * 31337)
    factor = float(state["algorithm_config"]["perturbation_factor"])
    multipliers = rng.choice([1.0 / factor, factor], size=len(vector))
    return [float(value) for value in np.clip(np.asarray(vector) * multipliers, 0, 1)]


def _schedule_pbt(state: dict, slots: int) -> list[dict]:
    config = state["algorithm_config"]
    population_size = int(config["population_size"])
    max_generations = int(config["max_generations"])
    interval = int(config["eval_interval"])
    created = []
    generation_zero = [
        rec for rec in state["recommendations"] if rec.get("generation") == 0
    ]
    while len(generation_zero) < population_size and len(created) < slots:
        vector, metadata = _vector_for_algorithm(state)
        member = len(generation_zero)
        rec = _new_recommendation(
            state,
            vector=vector,
            budget=interval,
            generation=0,
            config_id=f"config-{member:04d}",
            metadata={**metadata, "population_member": member},
        )
        generation_zero.append(rec)
        created.append(rec)
    if created:
        return created
    if len(generation_zero) < population_size:
        return []

    for generation in range(max_generations - 1):
        current = [
            rec
            for rec in state["recommendations"]
            if rec.get("generation") == generation
        ]
        if len(current) < population_size or any(
            rec["state"] not in TERMINAL_REC_STATES for rec in current
        ):
            return []
        next_generation = [
            rec
            for rec in state["recommendations"]
            if rec.get("generation") == generation + 1
        ]
        if len(next_generation) >= population_size:
            continue
        ranked = _ranked(state, current)
        if not ranked:
            state["status"] = "COMPLETE"
            return []
        elite_count = max(1, population_size // 2)
        elites = ranked[:elite_count]
        by_member = {
            int(rec["metadata"]["population_member"]): rec for rec in current
        }
        while len(next_generation) < population_size and len(created) < slots:
            member = len(next_generation)
            incumbent = by_member.get(member, elites[member % len(elites)])
            parent = (
                incumbent
                if incumbent in elites
                else elites[member % len(elites)]
            )
            vector = (
                list(parent["vector"])
                if incumbent in elites
                else _perturb_vector(state, parent["vector"], len(state["recommendations"]))
            )
            rec = _new_recommendation(
                state,
                vector=vector,
                parent=parent,
                budget=(generation + 2) * interval,
                generation=generation + 1,
                config_id=f"config-{member:04d}",
                metadata={
                    "population_member": member,
                    "exploit": parent["id"] if incumbent not in elites else None,
                },
            )
            next_generation.append(rec)
            created.append(rec)
        if created or len(next_generation) < population_size:
            return created

    final_generation = [
        rec
        for rec in state["recommendations"]
        if rec.get("generation") == max_generations - 1
    ]
    if final_generation and all(
        rec["state"] in TERMINAL_REC_STATES for rec in final_generation
    ):
        state["status"] = "COMPLETE"
    return []


def _schedule_recommendations(state: dict, slots: int) -> list[dict]:
    if state["algorithm"] == "pbt":
        return _schedule_pbt(state, slots)
    if state["algorithm"] == "asha":
        return _schedule_asha(state, slots)
    if state["algorithm"] in BUDGETED_ALGORITHMS:
        return _schedule_successive_halving(state, slots)
    limit = (
        state["algorithm_config"]["max_experiments"]
        if state["algorithm"] in {"hybrid", "autoresearch"}
        else state["max_recommendations"]
    )
    fresh = len(state["recommendations"])
    if fresh >= limit:
        if all(rec["state"] in TERMINAL_REC_STATES for rec in state["recommendations"]):
            state["status"] = "COMPLETE"
        return []
    created = []
    for _ in range(min(slots, limit - fresh)):
        vector, metadata = _vector_for_algorithm(state)
        created.append(
            _new_recommendation(state, vector=vector, metadata=metadata)
        )
    return created


def cmd_recommend(args: argparse.Namespace) -> int:
    state_path = args.state.resolve()
    with _state_lock(state_path):
        state = _load_state(state_path)
        ready = [
            rec for rec in _active_recommendations(state) if rec["state"] == "READY"
        ]
        if ready:
            _print(
                {
                    "ready": True,
                    "created": False,
                    "reason": "existing_ready_recommendation",
                    "recommendation": ready[0],
                    "recommendations": ready,
                }
            )
            return 0
        _maybe_complete(state)
        if state["status"] != "ACTIVE":
            _write_state(state_path, state)
            _print(
                {
                    "ready": False,
                    "created": False,
                    "reason": "experiment_complete",
                    **_summary(state),
                }
            )
            return 0
        active = _active_recommendations(state)
        slots = max(0, state["max_concurrent"] - len(active))
        recommendations = _schedule_recommendations(state, slots)
        _maybe_complete(state)
        _write_state(state_path, state)
    if recommendations:
        _print(
            {
                "ready": True,
                "created": True,
                "recommendation": recommendations[0],
                "recommendations": recommendations,
            }
        )
    else:
        _print(
            {
                "ready": False,
                "created": False,
                "reason": (
                    "experiment_complete"
                    if state["status"] == "COMPLETE"
                    else "recommendation_in_flight"
                ),
                **_summary(state),
            }
        )
    return 0


def cmd_bind_job(args: argparse.Namespace) -> int:
    if not ID_RE.fullmatch(args.job_id):
        raise SystemExit(f"Invalid job id: {args.job_id!r}")
    state_path = args.state.resolve()
    with _state_lock(state_path):
        state = _load_state(state_path)
        rec = _find_recommendation(state, args.rec_id)
        if rec["state"] == "RUNNING":
            if rec["attempts"] and rec["attempts"][-1]["job_id"] == args.job_id:
                _print({"bound": True, "idempotent": True, "recommendation": rec})
                return 0
            raise SystemExit(f"{args.rec_id} is already bound to another running job")
        if rec["state"] != "READY":
            raise SystemExit(f"{args.rec_id} is {rec['state']}, not READY")
        rec["attempts"].append(
            {
                "job_id": args.job_id,
                "bound_at": _now(),
                "finished_at": None,
                "outcome": None,
                "message": "",
            }
        )
        rec["state"] = "RUNNING"
        _write_state(state_path, state)
    _print({"bound": True, "idempotent": False, "recommendation": rec})
    return 0


def _report_payload(args: argparse.Namespace, state: dict, rec: dict) -> dict:
    if not rec["attempts"]:
        raise SystemExit(f"{rec['id']} has no bound platform job")
    active_job_id = rec["attempts"][-1]["job_id"]

    if args.metric_record:
        payload = _read_document(args.metric_record.resolve())
        if not isinstance(payload, dict):
            raise SystemExit("Metric record must be an object")
        _jsonschema_validate(payload, METRIC_SCHEMA)
        for key, expected in (
            ("experiment_id", state["id"]),
            ("rec_id", rec["id"]),
            ("job_id", active_job_id),
            ("primary_metric", state["metric_name"]),
            ("direction", state["direction"]),
        ):
            if payload[key] != expected:
                raise SystemExit(
                    f"Metric record {key}={payload[key]!r} does not match {expected!r}"
                )
        if payload["status"] == "COMPLETE":
            try:
                objective_score = _objective_score(state, payload["metrics"])
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc
            checkpoint_uri = payload["artifacts"].get("checkpoint_uri")
            output_uris = payload["artifacts"].get("output_uris") or []
            artifact_uri = (
                checkpoint_uri
                or payload["artifacts"].get("primary_uri")
                or (output_uris[0] if output_uris else None)
            )
            if state["action"] == "train" and not checkpoint_uri:
                raise SystemExit(
                    "A successful SDK-free training metric record requires "
                    "artifacts.checkpoint_uri"
                )
            outcome = "SUCCESS"
            metric = payload["metrics"][state["metric_name"]]
            objective_values = {
                objective["name"]: payload["metrics"][objective["name"]]
                for objective in state["objectives"]
            }
            message = ""
        elif payload["status"] == "ERROR":
            outcome = payload["failure"]["class"]
            metric = None
            objective_values = {}
            objective_score = None
            checkpoint_uri = None
            artifact_uri = None
            message = payload["failure"]["message"]
        else:
            outcome = "CANCELED"
            metric = None
            objective_values = {}
            objective_score = None
            checkpoint_uri = None
            artifact_uri = None
            message = "platform job canceled"
        return {
            "job_id": active_job_id,
            "outcome": outcome,
            "metric": metric,
            "objective_values": objective_values,
            "objective_score": objective_score,
            "checkpoint_uri": checkpoint_uri,
            "artifact_uri": artifact_uri,
            "feedback": _sanitize_feedback(payload.get("feedback")),
            "message": message,
        }

    if not args.outcome:
        raise SystemExit("Pass either --metric-record or --outcome")
    if not args.job_id:
        raise SystemExit("Explicit --outcome reports require --job-id")
    job_id = args.job_id
    if job_id != active_job_id:
        raise SystemExit(
            f"Report job id {job_id!r} does not match active job {active_job_id!r}"
        )
    objective_values = dict(args.metric_value or [])
    if args.metric is not None:
        existing = objective_values.get(state["metric_name"])
        if existing is not None and existing != args.metric:
            raise SystemExit(
                f"Conflicting values for primary metric {state['metric_name']!r}"
            )
        objective_values[state["metric_name"]] = args.metric
    objective_score = None
    artifact_uri = args.artifact_uri or args.checkpoint_uri
    feedback = (
        _sanitize_feedback(_read_document(args.feedback.resolve()))
        if args.feedback
        else None
    )
    if args.outcome == "SUCCESS":
        try:
            objective_score = _objective_score(state, objective_values)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if state["action"] == "train" and not args.checkpoint_uri:
            raise SystemExit(
                "--outcome SUCCESS requires --metric and --checkpoint-uri "
                "for single-objective training (or all --metric-value entries "
                "and --checkpoint-uri for multi-objective training)"
            )
        if not artifact_uri:
            raise SystemExit(
                "--outcome SUCCESS requires --checkpoint-uri or --artifact-uri"
            )
    return {
        "job_id": job_id,
        "outcome": args.outcome,
        "metric": objective_values.get(state["metric_name"]),
        "objective_values": objective_values if args.outcome == "SUCCESS" else {},
        "objective_score": objective_score,
        "checkpoint_uri": args.checkpoint_uri,
        "artifact_uri": artifact_uri,
        "feedback": feedback,
        "message": args.message or "",
    }


def _validate_repeated_report(rec: dict, payload: dict) -> None:
    """Reject a conflicting replay while accepting an identical terminal fact."""
    attempt = rec["attempts"][-1]
    conflicts = []
    if payload["job_id"] != attempt["job_id"]:
        conflicts.append(
            f"job_id {payload['job_id']!r} != {attempt['job_id']!r}"
        )
    if payload["outcome"] != attempt["outcome"]:
        conflicts.append(
            f"outcome {payload['outcome']!r} != {attempt['outcome']!r}"
        )
    if payload["outcome"] == "SUCCESS":
        try:
            metric = _finite_number(payload["metric"], "reported metric")
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if metric != rec["metric"]:
            conflicts.append(f"metric {metric!r} != {rec['metric']!r}")
        if payload["checkpoint_uri"] != rec["checkpoint_uri"]:
            conflicts.append(
                "checkpoint_uri "
                f"{payload['checkpoint_uri']!r} != {rec['checkpoint_uri']!r}"
            )
        if payload["artifact_uri"] != rec.get("artifact_uri"):
            conflicts.append(
                "artifact_uri "
                f"{payload['artifact_uri']!r} != {rec.get('artifact_uri')!r}"
            )
        if payload["objective_values"] != rec.get("objective_values", {}):
            conflicts.append("objective_values differ")
        if payload["objective_score"] != rec.get("objective_score"):
            conflicts.append("objective_score differs")
        if payload["feedback"] != (rec.get("feedback") or [None])[-1]:
            conflicts.append("feedback differs")
    if conflicts:
        raise SystemExit(
            f"Conflicting repeated report for {rec['id']}: " + "; ".join(conflicts)
        )


def _track_wandb(state: dict, rec: dict, state_path: Path) -> None:
    tracking = state.get("tracking") or {}
    if not tracking.get("wandb_enabled") or tracking.get("wandb_mode") == "disabled":
        return
    result = {
        "mode": tracking["wandb_mode"],
        "project": tracking.get("wandb_project") or "TAO AutoML",
        "logged": False,
    }
    try:
        import wandb
    except ImportError:
        result["reason"] = "wandb helper is not installed"
        rec["metadata"]["wandb"] = result
        return
    if tracking["wandb_mode"] == "online" and not os.getenv("WANDB_API_KEY"):
        result["reason"] = "WANDB_API_KEY is not configured"
        rec["metadata"]["wandb"] = result
        return
    try:
        run = wandb.init(
            project=result["project"],
            entity=tracking.get("wandb_entity"),
            group=f"automl_{state['id']}",
            name=f"{state['id']}-{rec['id']}",
            job_type="automl-recommendation",
            dir=str(state_path.parent),
            mode=tracking["wandb_mode"],
            reinit=True,
            config={
                "algorithm": state["algorithm"],
                "network": state["network_arch"],
                "action": state["action"],
                "rec_id": rec["id"],
                "parameters": rec["parameters"],
                "budget": rec.get("budget"),
                "rung": rec.get("rung"),
                "generation": rec.get("generation"),
            },
        )
        log_values = {
            **{f"objective/{key}": value for key, value in rec["objective_values"].items()},
            "objective/scalarized": rec.get("objective_score"),
            "recommendation/index": rec["index"],
            "recommendation/succeeded": int(rec["state"] == "SUCCEEDED"),
        }
        run.log(log_values, step=rec["index"])
        run.summary["best_rec_id"] = state.get("best_rec_id")
        run.finish()
        result["logged"] = True
    except Exception as exc:  # pragma: no cover - provider/runtime failures vary
        result["reason"] = str(exc)[:500]
    rec["metadata"]["wandb"] = result


def cmd_report(args: argparse.Namespace) -> int:
    state_path = args.state.resolve()
    with _state_lock(state_path):
        state = _load_state(state_path)
        rec = _find_recommendation(state, args.rec_id)
        payload = _report_payload(args, state, rec)
        if rec["state"] in TERMINAL_REC_STATES:
            _validate_repeated_report(rec, payload)
            _print(
                {
                    "reported": True,
                    "idempotent": True,
                    "recommendation": rec,
                    **_summary(state),
                }
            )
            return 0
        if (
            rec["state"] == "READY"
            and rec["attempts"]
            and rec["attempts"][-1]["outcome"] == "ERR_INFRA"
        ):
            _validate_repeated_report(rec, payload)
            _print(
                {
                    "reported": True,
                    "idempotent": True,
                    "recommendation": rec,
                    **_summary(state),
                }
            )
            return 0
        if rec["state"] != "RUNNING":
            raise SystemExit(f"{args.rec_id} is {rec['state']}, not RUNNING")

        attempt = rec["attempts"][-1]
        attempt["finished_at"] = _now()
        attempt["outcome"] = payload["outcome"]
        attempt["message"] = payload["message"]

        if payload["outcome"] == "ERR_INFRA":
            # Preserve the recommendation and its vector exactly. The platform
            # skill submits a new job-record with retry_of=<old job> and binds
            # that new id here; no optimization budget is consumed.
            rec["state"] = "READY"
            rec["failure"] = None
        elif payload["outcome"] == "SUCCESS":
            try:
                rec["metric"] = _finite_number(payload["metric"], "reported metric")
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc
            rec["checkpoint_uri"] = payload["checkpoint_uri"]
            rec["artifact_uri"] = payload["artifact_uri"]
            rec["objective_values"] = payload["objective_values"]
            rec["objective_score"] = payload["objective_score"]
            if payload["feedback"] is not None:
                rec["feedback"].append(payload["feedback"])
            rec["failure"] = None
            rec["state"] = "SUCCEEDED"
        elif payload["outcome"] == "ERR_PROGRAM":
            rec["state"] = "FAILED"
            rec["failure"] = {"class": "ERR_PROGRAM", "message": payload["message"]}
        else:
            rec["state"] = "CANCELED"
            rec["failure"] = {"class": "CANCELED", "message": payload["message"]}

        _refresh_best(state)
        _maybe_complete(state)
        if payload["outcome"] != "ERR_INFRA":
            _track_wandb(state, rec, state_path)
        _write_state(state_path, state)
    _print({"reported": True, "idempotent": False, "recommendation": rec, **_summary(state)})
    return 0


def _learning_curve_stop(state: dict, rec: dict) -> tuple[bool, dict]:
    if state["algorithm"] != "hyperband_es":
        return False, {"reason": "algorithm_does_not_use_predictive_stopping"}
    minimum = int(state["algorithm_config"]["min_early_stop_epochs"])
    observations = rec["observations"]
    if len(observations) < minimum:
        return False, {"reason": "insufficient_observations"}
    try:
        import numpy as np
        from scipy.optimize import OptimizeWarning, curve_fit
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "HyperBand-ES observations require numpy and scipy"
        ) from exc
    metric_name = state["metric_name"]
    x_values = np.asarray([row["step"] for row in observations], dtype=float)
    y_values = np.asarray(
        [row["metrics"][metric_name] for row in observations], dtype=float
    )

    def power_law(x_value, a, b, c):
        return a * np.power(x_value, b) + c

    try:
        initial = [y_values[0] - y_values[-1], -0.5, y_values[-1]]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            fitted, _ = curve_fit(
                power_law,
                x_values,
                y_values,
                p0=initial,
                maxfev=2000,
            )
        predicted = float(
            power_law(state["algorithm_config"]["max_epochs"], *fitted)
        )
        fitted_values = power_law(x_values, *fitted)
        residual = float(np.sum((y_values - fitted_values) ** 2))
        total = float(np.sum((y_values - np.mean(y_values)) ** 2))
        confidence = max(0.0, min(1.0, 1.0 - residual / total)) if total else 0.0
    except (RuntimeError, ValueError, FloatingPointError, TypeError):
        return False, {"reason": "curve_fit_failed"}
    peers = []
    for other in state["recommendations"]:
        if other["id"] == rec["id"]:
            continue
        peer_observations = other.get("observations") or []
        if peer_observations and metric_name in peer_observations[-1]["metrics"]:
            peers.append(float(peer_observations[-1]["metrics"][metric_name]))
        elif other["state"] == "SUCCEEDED" and other["metric"] is not None:
            peers.append(float(other["metric"]))
    details = {
        "reason": "prediction_evaluated",
        "predicted_final": predicted,
        "confidence": confidence,
        "peer_count": len(peers),
    }
    if not peers or confidence < state["algorithm_config"]["early_stop_threshold"]:
        return False, details
    margin = 0.05
    if state["direction"] == "maximize":
        should_stop = predicted < max(peers) * (1.0 - margin)
    else:
        should_stop = predicted > min(peers) * (1.0 + margin)
    return bool(should_stop), details


def cmd_observe(args: argparse.Namespace) -> int:
    state_path = args.state.resolve()
    metrics = dict(args.metric_value)
    with _state_lock(state_path):
        state = _load_state(state_path)
        rec = _find_recommendation(state, args.rec_id)
        if rec["state"] != "RUNNING":
            raise SystemExit(f"{args.rec_id} is {rec['state']}, not RUNNING")
        active_job_id = rec["attempts"][-1]["job_id"]
        if args.job_id != active_job_id:
            raise SystemExit(
                f"Observation job id {args.job_id!r} does not match {active_job_id!r}"
            )
        if state["metric_name"] not in metrics:
            raise SystemExit(
                f"Observation requires primary metric {state['metric_name']!r}"
            )
        for name, value in metrics.items():
            try:
                metrics[name] = _finite_number(value, name)
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc
        feedback = (
            _sanitize_feedback(_read_document(args.feedback.resolve()))
            if args.feedback
            else None
        )
        repeated = next(
            (row for row in rec["observations"] if row["step"] == args.step),
            None,
        )
        if repeated:
            if repeated["metrics"] != metrics or repeated.get("feedback") != feedback:
                raise SystemExit(
                    f"Conflicting observation for {args.rec_id} at step {args.step}"
                )
            should_stop, details = _learning_curve_stop(state, rec)
            _print(
                {
                    "observed": True,
                    "idempotent": True,
                    "should_cancel": should_stop,
                    "prediction": details,
                    "recommendation": rec,
                }
            )
            return 0
        rec["observations"].append(
            {
                "step": args.step,
                "metrics": metrics,
                "observed_at": _now(),
                **({"feedback": feedback} if feedback is not None else {}),
            }
        )
        rec["observations"].sort(key=lambda row: row["step"])
        should_stop, details = _learning_curve_stop(state, rec)
        if should_stop:
            rec["decision"] = "EARLY_STOP"
        _write_state(state_path, state)
    _print(
        {
            "observed": True,
            "idempotent": False,
            "should_cancel": should_stop,
            "prediction": details,
            "recommendation": rec,
        }
    )
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    state_path = args.state.resolve()
    with _state_lock(state_path):
        state = _load_state(state_path)
    output = _summary(state)
    if args.full:
        output["search_parameters"] = state["search_parameters"]
        output["recommendations"] = state["recommendations"]
    _print(output)
    return 0


def _pop_nested(target: dict, dotted: str) -> Any:
    current = target
    parts = dotted.split(".")
    for part in parts[:-1]:
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    if isinstance(current, dict) and parts[-1] in current:
        return current.pop(parts[-1])
    return None


def _best_rec_payload(
    state: dict,
    checkpoint_override: str | None,
    artifact_override: str | None,
) -> dict:
    if state["best_rec_id"] is None:
        raise SystemExit("Cannot finalize: the experiment has no successful recommendation")
    best = _find_recommendation(state, state["best_rec_id"])
    checkpoint_uri = checkpoint_override or best["checkpoint_uri"]
    artifact_uri = artifact_override or checkpoint_uri or best.get("artifact_uri")
    if state["action"] == "train" and not checkpoint_uri:
        raise SystemExit(
            "Cannot finalize: the best recommendation has no checkpoint URI. "
            "Report one in the metric record or pass --checkpoint-uri."
        )
    if not artifact_uri:
        raise SystemExit(
            "Cannot finalize: the best recommendation has no artifact URI. "
            "Report one or pass --artifact-uri."
        )
    specs = copy.deepcopy(best["spec_overrides"])
    observed_budget = {}
    for key in BUDGET_KEYS:
        value = _pop_nested(specs, key)
        if value is not None:
            observed_budget[key.split(".")[-1]] = value
    return {
        "schema_version": 1,
        "experiment_id": state["id"],
        "metric_name": state["metric_name"],
        "direction": state["direction"],
        "best": {
            "rec_id": best["id"],
            "score": best["metric"],
            "objective_values": best["objective_values"],
            "objective_score": best["objective_score"],
            "specs": specs,
            "observed_budget": observed_budget,
            **({"checkpoint_uri": checkpoint_uri} if checkpoint_uri else {}),
            "artifact_uri": artifact_uri,
        },
        "all_recs": [
            {
                **{"rec_id": rec["id"], "score": rec["metric"]},
                **(
                    {"job_id": rec["attempts"][-1]["job_id"]}
                    if rec["attempts"]
                    else {}
                ),
            }
            for rec in state["recommendations"]
        ],
    }


def cmd_finalize(args: argparse.Namespace) -> int:
    state_path = args.state.resolve()
    with _state_lock(state_path):
        state = _load_state(state_path)
        if state["status"] != "COMPLETE":
            raise SystemExit(
                "Cannot finalize an active experiment; finish the recommendation "
                "budget first"
            )
        payload = _best_rec_payload(state, args.checkpoint_uri, args.artifact_uri)
        _jsonschema_validate(payload, BEST_REC_SCHEMA)
        state["status"] = "COMPLETE"
        if args.checkpoint_uri:
            best = _find_recommendation(state, state["best_rec_id"])
            best["checkpoint_uri"] = args.checkpoint_uri
            best["artifact_uri"] = args.artifact_uri or args.checkpoint_uri
        elif args.artifact_uri:
            best = _find_recommendation(state, state["best_rec_id"])
            best["artifact_uri"] = args.artifact_uri
        _write_state(state_path, state)
    if args.out:
        out_path = args.out.resolve()
        _atomic_write(out_path, payload)
        _print({"best_rec": str(out_path), "payload": payload})
    else:
        _print(payload)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="Create a new durable AutoML experiment")
    init.add_argument("--state", required=True, type=Path)
    init.add_argument("--schema", required=True, type=Path)
    init.add_argument("--base-spec", type=Path)
    init.add_argument("--search-space", type=Path)
    init.add_argument("--parameter", action="append")
    init.add_argument("--experiment-id")
    init.add_argument("--network-arch", required=True)
    init.add_argument("--action", default="train")
    init.add_argument("--algorithm", choices=ALGORITHMS, default="bayesian")
    init.add_argument("--metric", required=True)
    init.add_argument("--direction", required=True, choices=("maximize", "minimize"))
    init.add_argument("--objective", action="append", type=_parse_objective)
    init.add_argument("--max-recommendations", type=int, default=5)
    init.add_argument("--max-concurrent", type=int, default=1)
    init.add_argument("--candidate-count", type=int, default=1024)
    init.add_argument("--seed", type=int, default=0)
    init.add_argument("--max-epochs", type=int, default=27)
    init.add_argument("--reduction-factor", type=int, default=3)
    init.add_argument("--epoch-multiplier", type=int, default=1)
    init.add_argument("--population-size", type=int, default=10)
    init.add_argument("--max-generations", type=int, default=20)
    init.add_argument("--eval-interval", type=int, default=10)
    init.add_argument("--perturbation-factor", type=float, default=1.2)
    init.add_argument("--mutation-factor", type=float, default=0.5)
    init.add_argument("--crossover-probability", type=float, default=0.5)
    init.add_argument("--early-stop-threshold", type=float, default=0.1)
    init.add_argument("--min-early-stop-epochs", type=int, default=3)
    init.add_argument("--kde-samples", type=int, default=64)
    init.add_argument("--top-fraction", type=float, default=0.15)
    init.add_argument("--min-points-in-model", type=int, default=10)
    init.add_argument("--max-trials", type=int)
    init.add_argument("--min-top-configs", type=int, default=5)
    init.add_argument("--max-experiments", type=int, default=50)
    init.add_argument("--research-program", type=Path)
    init.add_argument("--evolvable-text-parameter", action="append")
    init.add_argument("--llm-endpoint")
    init.add_argument("--llm-model")
    init.add_argument("--llm-temperature", type=float, default=0.7)
    init.add_argument("--llm-max-tokens", type=int, default=4096)
    init.add_argument("--llm-timeout", type=float, default=120.0)
    init.add_argument("--llm-max-retries", type=int, default=3)
    init.add_argument("--llm-retry-delay", type=float, default=2.0)
    init.add_argument("--wandb", action="store_true")
    init.add_argument("--wandb-project")
    init.add_argument("--wandb-entity")
    init.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="disabled",
    )
    init.set_defaults(fn=cmd_init)

    recommend = sub.add_parser("recommend", help="Return or create the next recommendation")
    recommend.add_argument("--state", required=True, type=Path)
    recommend.set_defaults(fn=cmd_recommend)

    bind = sub.add_parser(
        "bind-job",
        help="Bind a platform job-record id to a READY recommendation",
    )
    bind.add_argument("--state", required=True, type=Path)
    bind.add_argument("--rec-id", required=True)
    bind.add_argument("--job-id", required=True)
    bind.set_defaults(fn=cmd_bind_job)

    report = sub.add_parser("report", help="Report a terminal platform result")
    report.add_argument("--state", required=True, type=Path)
    report.add_argument("--rec-id", required=True)
    source = report.add_mutually_exclusive_group(required=True)
    source.add_argument("--metric-record", type=Path)
    source.add_argument(
        "--outcome",
        choices=("SUCCESS", "ERR_INFRA", "ERR_PROGRAM", "CANCELED"),
    )
    report.add_argument("--job-id")
    report.add_argument("--metric", type=float)
    report.add_argument("--metric-value", action="append", type=_parse_named_metric)
    report.add_argument("--checkpoint-uri")
    report.add_argument("--artifact-uri")
    report.add_argument("--feedback", type=Path)
    report.add_argument("--message", default="")
    report.set_defaults(fn=cmd_report)

    observe = sub.add_parser(
        "observe",
        help="Record an intermediate metric and evaluate predictive early stopping",
    )
    observe.add_argument("--state", required=True, type=Path)
    observe.add_argument("--rec-id", required=True)
    observe.add_argument("--job-id", required=True)
    observe.add_argument("--step", required=True, type=int)
    observe.add_argument(
        "--metric-value",
        action="append",
        type=_parse_named_metric,
        required=True,
    )
    observe.add_argument("--feedback", type=Path)
    observe.set_defaults(fn=cmd_observe)

    status = sub.add_parser("status", help="Read durable experiment status")
    status.add_argument("--state", required=True, type=Path)
    status.add_argument("--full", action="store_true")
    status.set_defaults(fn=cmd_status)

    finalize = sub.add_parser("finalize", help="Emit validated best_rec.json")
    finalize.add_argument("--state", required=True, type=Path)
    finalize.add_argument("--checkpoint-uri")
    finalize.add_argument("--artifact-uri")
    finalize.add_argument("--out", type=Path)
    finalize.set_defaults(fn=cmd_finalize)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if getattr(args, "max_recommendations", 1) < 1:
        raise SystemExit("--max-recommendations must be at least 1")
    if getattr(args, "candidate_count", 32) < 32:
        raise SystemExit("--candidate-count must be at least 32")
    if getattr(args, "seed", 0) < 0:
        raise SystemExit("--seed must be non-negative")
    for name in (
        "max_concurrent",
        "max_epochs",
        "epoch_multiplier",
        "population_size",
        "max_generations",
        "eval_interval",
        "kde_samples",
        "min_points_in_model",
        "min_top_configs",
        "max_experiments",
        "llm_max_tokens",
        "llm_max_retries",
    ):
        if getattr(args, name, 1) < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be at least 1")
    if getattr(args, "reduction_factor", 2) < 2:
        raise SystemExit("--reduction-factor must be at least 2")
    if getattr(args, "max_trials", None) is not None and args.max_trials < 1:
        raise SystemExit("--max-trials must be at least 1 when provided")
    if getattr(args, "mutation_factor", 0.5) <= 0:
        raise SystemExit("--mutation-factor must be positive")
    if getattr(args, "perturbation_factor", 1.2) <= 0:
        raise SystemExit("--perturbation-factor must be positive")
    if not 0 <= getattr(args, "crossover_probability", 0.5) <= 1:
        raise SystemExit("--crossover-probability must be within [0,1]")
    if not 0 < getattr(args, "top_fraction", 0.15) <= 1:
        raise SystemExit("--top-fraction must be within (0,1]")
    if not 0 <= getattr(args, "llm_temperature", 0.7) <= 2:
        raise SystemExit("--llm-temperature must be within [0,2]")
    if getattr(args, "llm_timeout", 120.0) <= 0:
        raise SystemExit("--llm-timeout must be positive")
    if getattr(args, "llm_retry_delay", 2.0) < 0:
        raise SystemExit("--llm-retry-delay must be non-negative")
    if hasattr(args, "step") and args.step < 1:
        raise SystemExit("--step must be at least 1")
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
