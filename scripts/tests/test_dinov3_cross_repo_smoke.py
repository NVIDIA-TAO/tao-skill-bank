# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-repository dummy-data smoke tests for DINOv3 DEFT."""

import json
import os
from pathlib import Path
import sys
from urllib.parse import unquote, urlparse

import pytest
import yaml


try:
    import numpy as np
    import pandas as pd
    from PIL import Image

    from nvidia_tao_ds.mining.dinov3.materialize import materialize_manifest
    from nvidia_tao_ds.mining.dinov3.workflow.controller import (
        _native_implementation_files,
        _validate_score_embedding_binding,
    )
    from nvidia_tao_ds.mining.dinov3.workflow.execution import (
        StageRequest,
        build_runner,
        client_job_id,
    )
    from nvidia_tao_pytorch.ssl.dinov3.dataloader.dataset import DinoV3Dataset
    from nvidia_tao_pytorch.ssl.dinov3.utils.refinement_attestation import (
        canonical_digest,
        file_sha256,
        native_attestation,
    )
except ImportError as error:
    if os.environ.get("TAO_DEFT_REQUIRE_CROSS_REPO_SMOKE") == "1":
        raise
    pytest.skip(
        f"cross-repository smoke requires the TAO DS container: {error.name}",
        allow_module_level=True,
    )


def _transform(image):
    """Return a minimal DINOv3 training crop contract."""
    return {"global_crops": [image.copy()], "local_crops": []}


def test_balanced_ds_manifest_loads_real_images_in_pytorch(tmp_path: Path) -> None:
    """Exercise the real DS producer and PyTorch consumer with replay rows."""
    images = []
    for index, color in enumerate(("red", "green", "blue")):
        path = tmp_path / f"sample-{index}.png"
        Image.new("RGB", (8, 8), color).save(path)
        images.append(path)

    query_path = tmp_path / "queries.parquet"
    pd.DataFrame(
        {
            "sample_id": ["query-minority", "query-majority-1", "query-majority-2"],
            "task": ["minority", "majority", "majority"],
        }
    ).to_parquet(query_path, index=False)
    delta_path = tmp_path / "delta.parquet"
    pd.DataFrame(
        {
            "sample_id": ["sample-minority", "sample-majority-1", "sample-majority-2"],
            "query_id": ["query-minority", "query-majority-1", "query-majority-2"],
            "storage_type": ["file", "file", "file"],
            "path": [str(path.resolve()) for path in images],
        }
    ).to_parquet(delta_path, index=False)

    artifact = materialize_manifest(
        delta_path=delta_path,
        query_path=query_path,
        balance_column="task",
        output_dir=tmp_path / "materialized",
    )
    view = artifact["payload"]["training_view"]
    parsed = urlparse(view["uri"])
    manifest_path = Path(unquote(parsed.path))
    dataset = DinoV3Dataset(
        root=tmp_path,
        manifest_path=manifest_path,
        transform=_transform,
    )

    assert artifact["payload"]["row_count"] == 3
    assert artifact["payload"]["training_view_rows"] == 4
    assert len(dataset) == 4
    assert [
        (record["sample_id"], record["replay_repeat"])
        for record in dataset.all_images
    ] == [
        ("sample-majority-1", 0),
        ("sample-majority-2", 0),
        ("sample-minority", 0),
        ("sample-minority", 1),
    ]
    loaded = [dataset[index] for index in range(len(dataset))]
    assert all(len(item["global_crops"]) == 1 for item in loaded)
    assert all(item["global_crops"][0].size == (8, 8) for item in loaded)
    assert loaded[2]["global_crops"][0].tobytes() == (
        loaded[3]["global_crops"][0].tobytes()
    )


def test_ds_stage_request_executes_precomputed_grit_leaf(tmp_path: Path) -> None:
    """Cross the DS process boundary into PyTorch's precomputed GRIT leaf."""
    input_path = tmp_path / "consensus.parquet"
    pd.DataFrame({
        "sample_id": ["a", "b", "c", "d"],
        "task": ["domain-a", "domain-a", "domain-b", "domain-b"],
        "global_consensus": [0.1, 0.9, 0.2, 0.8],
        "dense_consensus": [0.8, 0.2, 0.9, 0.1],
        "embedding": [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8]],
        "role": ["query", "query", "query", "query"],
    }).to_parquet(input_path, index=False)
    output_dir = tmp_path / "score"
    config = {
        "input_parquet": str(input_path),
        "output_dir": str(output_dir),
        "precomputed_consensus": True,
        "request_sha256": "sha256:" + "a" * 64,
    }
    code = (
        "import json,sys; "
        "from nvidia_tao_pytorch.ssl.dinov3.data_refinement.cli import run; "
        "print(json.dumps(run(json.loads(sys.argv[1])), sort_keys=True))"
    )
    attestation = native_attestation("grit_score")
    request = StageRequest(
        client_job_id=client_job_id(
            run_scope=str(tmp_path),
            run_id="dummy-data",
            round_index=1,
            stage="score",
            command=[sys.executable, "-c", code, json.dumps(config)],
        ),
        run_id="dummy-data",
        round_index=1,
        stage="score",
        command=[sys.executable, "-c", code, json.dumps(config)],
        workdir=str(tmp_path),
        results_dir=str(output_dir),
        environment={
            "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
            "TAO_REFINEMENT_ENTRYPOINT_SHA256": attestation[
                "entrypoint_sha256"
            ],
            "TAO_REFINEMENT_IMPLEMENTATION_SHA256": attestation[
                "implementation_sha256"
            ],
        },
        resources={"nodes": 1},
        execution_contract={
            "membership": "static",
            "attempt_scope": "process",
            "retry_scope": "process",
            "attempt_id_scope": "backend_attempt",
            "launch_id_environment": None,
            "adapter_managed_resources": None,
            "required_capabilities": [],
        },
    )
    result = build_runner({"backend": "local"}, tmp_path).run(request)
    assert result.state == "COMPLETE"
    scored = pd.read_parquet(output_dir / "grit_scores.parquet")
    assert scored["sample_id"].tolist() == ["a", "b", "c", "d"]
    assert scored["grit_score"].between(0, 1).all()
    score_embeddings = np.asarray(scored["embedding"].tolist(), dtype=np.float32)
    _validate_score_embedding_binding(
        input_path, "grit_score", scored, score_embeddings
    )
    assert (output_dir / "score_commit.json").is_file()
    assert (output_dir / "_SUCCESS").is_file()


@pytest.mark.parametrize("name", ["workflow", "metrics", "stage-request"])
def test_skill_and_runtime_workflow_schemas_are_identical(name) -> None:
    skill_root = Path(__file__).resolve().parents[2]
    skill_schema = json.loads((
        skill_root / "skills/applications/tao-run-dinov3-ssl-deft/"
        f"references/{name}.schema.json"
    ).read_text(encoding="utf-8"))
    import nvidia_tao_ds.mining.dinov3.workflow as workflow_package
    skill_schema.pop("$comment", None)
    runtime_schema = yaml.safe_load((
        Path(workflow_package.__file__).resolve().parent /
        f"schemas/{name}.schema.yaml"
    ).read_text(encoding="utf-8"))
    assert runtime_schema == skill_schema


@pytest.mark.parametrize("subtask", ["grit_score", "train"])
def test_ds_native_lock_matches_pytorch_attestation(subtask: str) -> None:
    import nvidia_tao_pytorch.ssl.dinov3.scripts as scripts_package
    module = Path(scripts_package.__file__).resolve().parent / f"{subtask}.py"
    ds_files = _native_implementation_files(module, subtask)
    ds_attestation = {
        "closure_version": "1.0",
        "entrypoint_sha256": file_sha256(module),
        "implementation_sha256": canonical_digest({
            name: file_sha256(path) for name, path in ds_files.items()
        }),
    }
    assert ds_attestation == native_attestation(subtask)
