# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Virtualenv previous-data path binding and retry-isolation tests."""

from __future__ import annotations

import hashlib
import pathlib
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

IAA_SCRIPTS = (
    pathlib.Path(__file__).resolve().parents[2]
    / "skills"
    / "applications"
    / "tao-run-deft-iaa"
    / "scripts"
)
sys.path.insert(0, str(IAA_SCRIPTS))

import run_deft_action as action  # noqa: E402


def _context(tmp_path: pathlib.Path, values: list[str]):
    workspace = tmp_path / "workspace"
    results = workspace / "results" / "run"
    dataset = workspace / "data" / "iaa"
    stage = results / "iter_2" / "embeddings" / "previous"
    stage.mkdir(parents=True)
    dataset.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"filepath": values, "label": list(range(len(values))) }).to_parquet(
        stage / "prev_pool.parquet", index=False
    )
    return SimpleNamespace(
        platform="virtualenv",
        name="viz_previous_embed",
        stage_dir=stage,
        results_dir=results,
        dataset_root=dataset,
    )


def _digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_previous_pool_rebinds_container_aliases_and_preserves_host_paths(tmp_path):
    results_image = tmp_path / "workspace/results/run/iter_1/generated.jpg"
    data_image = tmp_path / "workspace/data/iaa/source.jpg"
    results_image.parent.mkdir(parents=True)
    data_image.parent.mkdir(parents=True)
    results_image.write_bytes(b"generated")
    data_image.write_bytes(b"source")
    context = _context(
        tmp_path,
        [
            "/results/iter_1/generated.jpg",
            "/data/iaa/source.jpg",
            str(results_image),
        ],
    )

    report = action._rebind_virtualenv_previous_pool(context)  # noqa: SLF001
    frame = pd.read_parquet(context.stage_dir / "prev_pool.parquet")

    assert report == {"rows": 3, "rewritten": 2}
    assert frame["filepath"].tolist() == [
        str(results_image),
        str(data_image),
        str(results_image),
    ]


def test_previous_pool_accepts_canonical_dataset_symlink_alias(tmp_path):
    target = tmp_path / "workspace/data/iaa/images_raw/train/source.jpg"
    alias = tmp_path / "workspace/data/iaa/images/train_alias.jpg"
    target.parent.mkdir(parents=True)
    alias.parent.mkdir(parents=True)
    target.write_bytes(b"source")
    alias.symlink_to(pathlib.Path("../images_raw/train/source.jpg"))
    context = _context(tmp_path, ["/data/iaa/images/train_alias.jpg"])

    report = action._rebind_virtualenv_previous_pool(context)  # noqa: SLF001
    frame = pd.read_parquet(context.stage_dir / "prev_pool.parquet")

    assert report == {"rows": 1, "rewritten": 1}
    assert frame["filepath"].tolist() == [str(alias)]


@pytest.mark.parametrize("kind", ["escape", "multi_hop_escape", "broken", "directory"])
def test_previous_pool_rejects_unsafe_symlink_or_non_regular_target(tmp_path, kind):
    dataset = tmp_path / "workspace/data/iaa"
    aliases = dataset / "images"
    aliases.mkdir(parents=True)
    alias = aliases / "bad.jpg"
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"outside")

    if kind == "escape":
        alias.symlink_to(outside)
    elif kind == "multi_hop_escape":
        intermediate = dataset / "images_raw/link.jpg"
        intermediate.parent.mkdir(parents=True)
        intermediate.symlink_to(outside)
        alias.symlink_to(pathlib.Path("../images_raw/link.jpg"))
    elif kind == "broken":
        alias.symlink_to(pathlib.Path("../images_raw/missing.jpg"))
    else:
        target_dir = dataset / "images_raw/directory"
        target_dir.mkdir(parents=True)
        alias.symlink_to(pathlib.Path("../images_raw/directory"))

    context = _context(tmp_path, [str(alias)])

    with pytest.raises(ValueError, match="missing|invalid symlink|regular file"):
        action._rebind_virtualenv_previous_pool(context)  # noqa: SLF001


@pytest.mark.parametrize(
    "bad_path",
    [
        "/results/../escape.jpg",
        "/data/../../escape.jpg",
        "/tmp/outside-approved-roots.jpg",
        "relative/image.jpg",
        "/data/iaa/images/../source.jpg",
    ],
)
def test_previous_pool_rejects_traversal_and_outside_roots(tmp_path, bad_path):
    outside = pathlib.Path("/tmp/outside-approved-roots.jpg")
    if bad_path == str(outside):
        outside.write_bytes(b"outside")
    context = _context(tmp_path, [bad_path])

    with pytest.raises(ValueError, match="traversal|absolute path|outside approved roots"):
        action._rebind_virtualenv_previous_pool(context)  # noqa: SLF001


def test_attempt2_rebind_is_idempotent_and_does_not_touch_completed_peers(tmp_path):
    image = tmp_path / "workspace/results/run/iter_1/generated.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"generated")
    context = _context(tmp_path, ["/results/iter_1/generated.jpg"])
    weak = context.results_dir / "iter_2/embeddings/viz_weak/embeddings.parquet"
    mined = context.results_dir / "iter_2/embeddings/augmented/mined_embeddings.parquet"
    weak.parent.mkdir(parents=True)
    mined.parent.mkdir(parents=True)
    weak.write_bytes(b"completed weak")
    mined.write_bytes(b"completed mined")
    peer_digests = (_digest(weak), _digest(mined))

    first = action._rebind_virtualenv_previous_pool(context)  # noqa: SLF001
    rebound_digest = _digest(context.stage_dir / "prev_pool.parquet")
    second = action._rebind_virtualenv_previous_pool(context)  # noqa: SLF001

    assert first == {"rows": 1, "rewritten": 1}
    assert second == {"rows": 1, "rewritten": 0}
    assert _digest(context.stage_dir / "prev_pool.parquet") == rebound_digest
    assert (_digest(weak), _digest(mined)) == peer_digests
