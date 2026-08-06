#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Skill-owned GEPA adapters for batched TAO non-training actions.

The caller owns platform execution through ``run_batch(candidate, items)``.
This module owns candidate application, aligned scoring, leak-free reflection
records, optional official-set-metric reranking, and the GEPA optimization call.
It deliberately has no dependency on NVIDIA optimizer or execution packages.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

try:
    from gepa.core.adapter import EvaluationBatch
except Exception:  # pragma: no cover - optional helper import path
    @dataclass
    class EvaluationBatch:
        outputs: list[Any]
        scores: list[float]
        trajectories: list[Any] | None = None
        objective_scores: list[dict[str, float] | None] | None = None


MetricFn = Callable[[Any, Any], Any]
AggregateMetricFn = Callable[[Sequence[Any], Sequence[Any]], float | Mapping[str, Any]]
CandidateCostFn = Callable[[Mapping[str, Any]], float]
ReflectionEvidenceFn = Callable[
    [Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any] | None
]

_SENSITIVE_FEEDBACK_KEYS = {
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


def sanitize_reflective_feedback(value: Any) -> Any:
    """Remove benchmark answers and private media identity from LLM feedback."""
    if isinstance(value, Mapping):
        sanitized = {}
        for key, child in value.items():
            normalized = str(key).lower().replace("-", "_")
            if normalized in _SENSITIVE_FEEDBACK_KEYS or normalized.endswith("_path"):
                continue
            sanitized[str(key)] = sanitize_reflective_feedback(child)
        return sanitized
    if isinstance(value, (list, tuple)):
        return [sanitize_reflective_feedback(child) for child in value]
    return value


def _set_dotted_value(target: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    if not dotted_key or any(not part for part in parts):
        raise ValueError(f"Invalid dotted candidate key: {dotted_key!r}")
    node = target
    for part in parts[:-1]:
        current = node.get(part)
        if current is None:
            current = {}
            node[part] = current
        if not isinstance(current, dict):
            raise TypeError(
                f"Candidate key {dotted_key!r} crosses non-object field {part!r}"
            )
        node = current
    node[parts[-1]] = value


class TAOActionBatchRunner:
    """Apply a candidate to a base spec and execute one caller-owned TAO batch."""

    def __init__(
        self,
        base_specs: Mapping[str, Any],
        evaluate_action: Callable[[dict[str, Any], list[dict[str, Any]]], Sequence[Any]],
        *,
        value_coercers: Mapping[str, Callable[[Any], Any]] | None = None,
    ):
        self.base_specs = copy.deepcopy(dict(base_specs))
        self.evaluate_action = evaluate_action
        self.value_coercers = dict(value_coercers or {})

    def run_batch(
        self,
        candidate: Mapping[str, Any],
        items: Sequence[dict[str, Any]],
    ) -> list[Any]:
        specs = copy.deepcopy(self.base_specs)
        for key, raw_value in candidate.items():
            coerce = self.value_coercers.get(str(key))
            value = coerce(raw_value) if coerce else raw_value
            _set_dotted_value(specs, str(key), value)
        return list(self.evaluate_action(specs, list(items)))


class RoutedTAOActionBatchRunner:
    """Execute route-specific candidates while restoring original item order."""

    def __init__(self, runner, route_fn, route_candidates):
        self.runner = runner
        self.route_fn = route_fn
        self.route_candidates = {
            str(route): dict(overrides)
            for route, overrides in route_candidates.items()
        }
        self.last_route_counts: dict[str, int] = {}

    def run_batch(self, candidate, items):
        rows = list(items)
        routed: dict[str, list[tuple[int, dict[str, Any]]]] = {}
        for index, item in enumerate(rows):
            routed.setdefault(str(self.route_fn(item)), []).append((index, item))
        outputs: list[Any] = [None] * len(rows)
        self.last_route_counts = {route: len(batch) for route, batch in routed.items()}
        for route, batch in routed.items():
            routed_candidate = {**dict(candidate), **self.route_candidates.get(route, {})}
            result = list(
                self.runner.run_batch(routed_candidate, [item for _, item in batch])
            )
            if len(result) != len(batch):
                raise ValueError(
                    f"Route {route!r} returned {len(result)} outputs for "
                    f"{len(batch)} input items"
                )
            for (index, _), output in zip(batch, result):
                outputs[index] = output
        return outputs


class GEPAReflectionLM:
    """Adapt a skill-owned OpenAI-compatible client to GEPA's LM protocol."""

    def __init__(self, client, *, system_prompt: str | None = None):
        self.client = client
        self.system_prompt = system_prompt

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        messages = (
            [{"role": "user", "content": prompt}]
            if isinstance(prompt, str)
            else [dict(message) for message in prompt]
        )
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})
        response = self.client.chat(messages, json_mode=False)
        if not response.ok:
            raise RuntimeError(f"GEPA reflection LLM failed: {response.error}")
        return response.content


@dataclass
class ReflectionResponse:
    ok: bool
    content: str = ""
    error: str | None = None


class OpenAICompatibleClient:
    """Dependency-free chat client for GEPA reflection language models."""

    def __init__(
        self,
        endpoint: str,
        model: str,
        *,
        api_key: str | None = None,
        timeout: float = 120.0,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ):
        if not endpoint or not model:
            raise ValueError("Reflection client requires endpoint and model")
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.api_key = api_key or os.getenv("AUTOML_LLM_API_KEY") or os.getenv(
            "NVIDIA_API_KEY"
        )
        self.timeout = float(timeout)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)

    def chat(self, messages, json_mode=False) -> ReflectionResponse:
        url = (
            self.endpoint
            if self.endpoint.endswith("/chat/completions")
            else f"{self.endpoint}/chat/completions"
        )
        body = {
            "model": self.model,
            "messages": list(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            url,
            data=json.dumps(body, allow_nan=False, separators=(",", ":")).encode(),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode())
            return ReflectionResponse(
                ok=True,
                content=str(payload["choices"][0]["message"]["content"] or ""),
            )
        except Exception as exc:  # pragma: no cover - provider failures vary
            error = str(exc)
            if self.api_key:
                error = error.replace(self.api_key, "[REDACTED]")
            return ReflectionResponse(ok=False, error=error[:500])


class TAOGEPAAdapter:
    """Evaluate candidates with one aligned TAO action batch per candidate."""

    propose_new_texts = None

    def __init__(
        self,
        runner,
        metric_fn: MetricFn,
        *,
        fixed_candidate: Mapping[str, Any] | None = None,
        metric_context_fn: Callable[[dict[str, Any]], Mapping[str, Any]] | None = None,
        cache_outputs: bool = True,
        reflection_evidence_fn: ReflectionEvidenceFn | None = None,
        vision_components: Sequence[str] | None = None,
        config_choices: Mapping[str, Sequence[Any]] | None = None,
    ):
        self.runner = runner
        self.metric_fn = metric_fn
        self.fixed_candidate = dict(fixed_candidate or {})
        self.metric_context_fn = metric_context_fn
        self.cache_outputs = cache_outputs
        self.reflection_evidence_fn = reflection_evidence_fn
        self.vision_components = (
            {str(component) for component in vision_components}
            if vision_components is not None
            else None
        )
        self.config_choices = {
            str(component): tuple(str(choice) for choice in choices)
            for component, choices in (config_choices or {}).items()
        }
        for component, choices in self.config_choices.items():
            if not choices or len(set(choices)) != len(choices):
                raise ValueError(
                    f"Config component {component!r} needs unique non-empty choices"
                )
        self._output_cache: dict[str, list[Any]] = {}
        self._config_log: list[dict[str, Any]] = []
        self._component_reflection_lm = None
        if self.config_choices:
            self.propose_new_texts = self._propose_components

    def set_component_reflection_lm(self, reflection_lm) -> None:
        self._component_reflection_lm = reflection_lm

    def validate_seed_candidate(self, candidate: Mapping[str, Any]) -> None:
        for component, choices in self.config_choices.items():
            if component not in candidate:
                raise ValueError(f"Joint GEPA seed is missing config component {component!r}")
            if str(candidate[component]) not in choices:
                raise ValueError(
                    f"Config component {component!r} seed {candidate[component]!r} "
                    f"is not in {list(choices)!r}"
                )

    def full_candidate(self, candidate: Mapping[str, Any]) -> dict[str, Any]:
        return {**self.fixed_candidate, **dict(candidate)}

    @staticmethod
    def _cache_key(candidate, items) -> str:
        value = json.dumps(
            {"candidate": candidate, "items": list(items)},
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        )
        return hashlib.sha256(value.encode()).hexdigest()

    def run_outputs(self, candidate, items) -> list[Any]:
        batch = list(items)
        full = self.full_candidate(candidate)
        key = self._cache_key(full, batch)
        if self.cache_outputs and key in self._output_cache:
            return list(self._output_cache[key])
        if not callable(getattr(self.runner, "run_batch", None)):
            raise TypeError("TAO Auto-Prompter runner must implement run_batch")
        outputs = list(self.runner.run_batch(full, batch))
        if len(outputs) != len(batch):
            raise ValueError(
                f"run_batch returned {len(outputs)} outputs for {len(batch)} input items"
            )
        if self.cache_outputs:
            self._output_cache[key] = list(outputs)
        return outputs

    @staticmethod
    def _metric_result(value) -> tuple[float, Any, dict[str, float] | None]:
        if isinstance(value, tuple):
            if len(value) == 3:
                score, feedback, objectives = value
            elif len(value) == 2:
                score, feedback = value
                objectives = None
            else:
                raise TypeError("metric_fn tuple result must contain two or three values")
        else:
            score, feedback, objectives = value, "", None
        score = float(score)
        if not math.isfinite(score):
            raise ValueError("metric_fn returned a non-finite score")
        if objectives is not None and not isinstance(objectives, dict):
            raise TypeError("objective scores must be a dictionary or None")
        return score, sanitize_reflective_feedback(feedback), objectives

    def evaluate(self, batch, candidate, capture_traces=False) -> EvaluationBatch:
        items = list(batch)
        raw_outputs = self.run_outputs(candidate, items)
        outputs, scores, objectives = [], [], []
        trajectories = [] if capture_traces else None
        for item, output in zip(items, raw_outputs):
            try:
                context = dict(self.metric_context_fn(item)) if self.metric_context_fn else {}
                score, feedback, objective = self._metric_result(
                    self.metric_fn(output, item.get("gold"), **context)
                )
            except Exception as exc:
                score, feedback, objective = 0.0, f"evaluation failed: {exc}", None
            outputs.append(output)
            scores.append(score)
            objectives.append(objective)
            if trajectories is not None:
                trajectories.append(
                    {
                        "item": item,
                        "query": item.get("query") or "(video analysis task)",
                        "output": output,
                        "feedback": feedback,
                        "score": score,
                    }
                )
        if self.config_choices and scores:
            full = self.full_candidate(candidate)
            self._config_log.append(
                {
                    "config": {
                        component: str(full.get(component, ""))
                        for component in self.config_choices
                    },
                    "score": sum(scores) / len(scores),
                }
            )
        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
            objective_scores=objectives if any(row is not None for row in objectives) else None,
        )

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        full = self.full_candidate(candidate)
        records = {component: [] for component in components_to_update}
        needs_visual_evidence = (
            self.reflection_evidence_fn is not None
            and (
                self.vision_components is None
                or bool(set(components_to_update) & self.vision_components)
            )
        )
        for trajectory in eval_batch.trajectories or []:
            feedback = trajectory.get("feedback", "")
            if not isinstance(feedback, str):
                feedback = json.dumps(feedback, sort_keys=True, default=str)
            record = {
                "Inputs": {"query": trajectory.get("query")},
                "Generated Outputs": str(trajectory.get("output", ""))[:800],
                "Feedback": feedback,
            }
            evidence = None
            if needs_visual_evidence and float(trajectory.get("score", 0)) < 1:
                evidence = self.reflection_evidence_fn(trajectory.get("item", {}), full)
                if evidence is not None and not isinstance(evidence, Mapping):
                    raise TypeError("reflection_evidence_fn must return a mapping or None")
            for component in components_to_update:
                row = copy.deepcopy(record)
                if evidence and (
                    self.vision_components is None or component in self.vision_components
                ):
                    row["Visual Evidence"] = copy.deepcopy(dict(evidence))
                records[component].append(row)
        return records

    def _config_history(self, component):
        values: dict[str, list[float]] = {}
        for record in self._config_log:
            value = record["config"].get(component)
            if value:
                values.setdefault(value, []).append(float(record["score"]))
        means = {key: sum(rows) / len(rows) for key, rows in values.items()}
        counts = {key: len(rows) for key, rows in values.items()}
        return means, counts

    def _propose_config(self, component, current, beta=1.0):
        choices = self.config_choices[component]
        means, counts = self._config_history(component)
        untried = [choice for choice in choices if choice not in means]
        if untried:
            return untried[0]
        return max(
            choices,
            key=lambda choice: means[choice] + beta / math.sqrt(1 + counts[choice]),
            default=str(current),
        )

    def _propose_components(self, candidate, reflective_dataset, components_to_update):
        proposals = {}
        for component in components_to_update:
            if component in self.config_choices:
                proposals[component] = self._propose_config(component, candidate.get(component))
                continue
            records = reflective_dataset.get(component)
            if not records:
                continue
            if self._component_reflection_lm is None:
                raise RuntimeError("Joint GEPA text proposal requires a reflection LM")
            from gepa.strategies.instruction_proposal import InstructionProposalSignature

            proposals[component] = InstructionProposalSignature.run(
                lm=self._component_reflection_lm,
                input_dict={
                    "current_instruction_doc": candidate[component],
                    "dataset_with_feedback": records,
                    "prompt_template": None,
                },
            )["new_instruction"]
        return proposals


@dataclass
class AutoPrompterResult:
    best_candidate: dict[str, Any]
    best_full_candidate: dict[str, Any]
    selected_candidate_index: int
    gepa_candidate_index: int
    validation_score: float
    validation_metrics: dict[str, Any]
    candidate_validation_metrics: list[dict[str, Any]]
    test_score: float | None
    test_metrics: dict[str, Any] | None
    gepa_result: Any

    def to_dict(self):
        return {
            key: getattr(self, key)
            for key in (
                "best_candidate",
                "best_full_candidate",
                "selected_candidate_index",
                "gepa_candidate_index",
                "validation_score",
                "validation_metrics",
                "candidate_validation_metrics",
                "test_score",
                "test_metrics",
            )
        }


class GEPAutoPrompter:
    """Run GEPA, rerank on validation only, then score one frozen test winner."""

    def __init__(
        self,
        adapter: TAOGEPAAdapter,
        *,
        reflection_lm,
        aggregate_metric_fn: AggregateMetricFn | None = None,
        aggregate_metric_key: str = "macro_f1",
        candidate_cost_fn: CandidateCostFn | None = None,
        candidate_cost_weight: float = 0.0,
        **gepa_kwargs,
    ):
        self.adapter = adapter
        self.reflection_lm = reflection_lm
        self.aggregate_metric_fn = aggregate_metric_fn
        self.aggregate_metric_key = aggregate_metric_key
        self.candidate_cost_fn = candidate_cost_fn
        self.candidate_cost_weight = float(candidate_cost_weight)
        if not math.isfinite(self.candidate_cost_weight) or self.candidate_cost_weight < 0:
            raise ValueError("candidate_cost_weight must be finite and non-negative")
        if self.candidate_cost_weight and self.candidate_cost_fn is None:
            raise ValueError("candidate_cost_fn is required when cost weight is set")
        self.gepa_kwargs = dict(gepa_kwargs)
        if self.adapter.config_choices:
            self.adapter.set_component_reflection_lm(reflection_lm)

    def _aggregate(self, outputs, items):
        if self.aggregate_metric_fn is None:
            raise RuntimeError("aggregate_metric_fn is required for aggregate reranking")
        raw = self.aggregate_metric_fn(outputs, [item.get("gold") for item in items])
        metrics = dict(raw) if isinstance(raw, Mapping) else {self.aggregate_metric_key: raw}
        if self.aggregate_metric_key not in metrics:
            raise KeyError(f"Aggregate metric has no {self.aggregate_metric_key!r} field")
        score = float(metrics[self.aggregate_metric_key])
        if not math.isfinite(score):
            raise ValueError("Aggregate metric returned a non-finite score")
        return score, metrics

    def optimize(self, seed_candidate, trainset, valset, *, budget, testset=None):
        try:
            import gepa
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("GEPA Auto-Prompter requires the 'gepa' helper") from exc
        train_items, val_items = list(trainset), list(valset)
        if not train_items or not val_items or budget <= 0:
            raise ValueError("GEPA requires non-empty train/validation sets and budget > 0")
        self.adapter.validate_seed_candidate(seed_candidate)
        kwargs = dict(self.gepa_kwargs)
        reserved = {
            "seed_candidate",
            "trainset",
            "valset",
            "adapter",
            "reflection_lm",
            "max_metric_calls",
        }
        overlap = reserved & kwargs.keys()
        if overlap:
            raise TypeError(f"GEPA options are managed by the skill: {sorted(overlap)}")
        kwargs.setdefault("cache_evaluation", True)
        result = gepa.optimize(
            seed_candidate=dict(seed_candidate),
            trainset=train_items,
            valset=val_items,
            adapter=self.adapter,
            reflection_lm=self.reflection_lm,
            max_metric_calls=budget,
            **kwargs,
        )
        candidates = [dict(candidate) for candidate in result.candidates]
        if not candidates:
            raise RuntimeError("GEPA returned no candidates")
        gepa_index = int(result.best_idx)
        candidate_metrics = []
        if self.aggregate_metric_fn is None:
            selected_index = gepa_index
            validation_score = float(result.val_aggregate_scores[selected_index])
            validation_metrics = {"gepa_proxy": validation_score}
        else:
            for index, candidate in enumerate(candidates):
                outputs = self.adapter.run_outputs(candidate, val_items)
                score, metrics = self._aggregate(outputs, val_items)
                row = {
                    "candidate_index": index,
                    "score": score,
                    "metrics": metrics,
                    "gepa_proxy": float(result.val_aggregate_scores[index]),
                }
                if self.candidate_cost_fn:
                    cost = float(self.candidate_cost_fn(self.adapter.full_candidate(candidate)))
                    if not math.isfinite(cost) or cost < 0:
                        raise ValueError("Candidate cost must be finite and non-negative")
                    row["cost"] = cost
                    row["utility"] = score - self.candidate_cost_weight * cost
                candidate_metrics.append(row)
            selected = max(
                candidate_metrics,
                key=lambda row: (
                    row.get("utility", row["score"]),
                    row["score"],
                    -row.get("cost", 0),
                    row["gepa_proxy"],
                    -row["candidate_index"],
                ),
            )
            selected_index = int(selected["candidate_index"])
            validation_score = float(selected["score"])
            validation_metrics = dict(selected["metrics"])
        best = candidates[selected_index]
        test_score = test_metrics = None
        if testset is not None:
            test_items = list(testset)
            if not test_items:
                raise ValueError("testset must be non-empty when provided")
            test_score, test_metrics = self._aggregate(
                self.adapter.run_outputs(best, test_items), test_items
            )
        return AutoPrompterResult(
            best_candidate=best,
            best_full_candidate=self.adapter.full_candidate(best),
            selected_candidate_index=selected_index,
            gepa_candidate_index=gepa_index,
            validation_score=validation_score,
            validation_metrics=validation_metrics,
            candidate_validation_metrics=candidate_metrics,
            test_score=test_score,
            test_metrics=test_metrics,
            gepa_result=result,
        )
