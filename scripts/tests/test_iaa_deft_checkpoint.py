# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for IAA checkpoint publication attempt lineage."""

import json
import os
import sys
import time
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
IAA_SCRIPTS = (
    REPO_ROOT / "skills/applications/tao-run-deft-iaa/scripts"
)
sys.path.insert(0, str(IAA_SCRIPTS))
from checkpoint_contract import (  # noqa: E402
    checkpoint_lineage_started_ns,
    validate_best_checkpoint,
)
from iaa_deft.utils import get_current_checkpoint  # noqa: E402


def _checkpoint(train: Path, mtime_ns: int) -> Path:
    path = train / "model_epoch_000_step_00312.pth"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"checkpoint")
    os.utime(path, ns=(mtime_ns, mtime_ns))
    return path


def test_retry_publishes_checkpoint_from_earlier_attempt_in_same_lineage(tmp_path):
    train = tmp_path / "train"
    first_attempt_started = time.time_ns() - 120_000_000_000
    checkpoint = _checkpoint(train, first_attempt_started + 30_000_000_000)
    retry_started = first_attempt_started + 90_000_000_000

    published = Path(
        get_current_checkpoint(
            str(train), earliest_mtime_ns=first_attempt_started
        )
    )
    metadata = json.loads(
        (train / "best/clip_best_val_t2i_mAP.json").read_text()
    )
    provenance = validate_best_checkpoint(
        published, train, started_ns=first_attempt_started
    )

    assert checkpoint.stat().st_mtime_ns < retry_started
    assert published.resolve() == checkpoint
    assert metadata["selection_strategy"] == "newest_fallback"
    assert metadata["metric"] is None
    assert provenance["checkpoint_selection_strategy"] == "newest_fallback"


def test_checkpoint_older_than_attempt_lineage_is_rejected_before_publication(tmp_path):
    train = tmp_path / "train"
    lineage_started = time.time_ns() - 60_000_000_000
    _checkpoint(train, lineage_started - 1)
    best = train / "best/clip_best_val_t2i_mAP.pth"

    with pytest.raises(ValueError, match="predates the train attempt lineage"):
        get_current_checkpoint(str(train), earliest_mtime_ns=lineage_started)

    assert not os.path.lexists(best)
    assert not (train / "best/clip_best_val_t2i_mAP.json").exists()


def test_legacy_retry_uses_run_start_instead_of_retry_start(tmp_path):
    run_started_ns = time.time_ns() - 120_000_000_000
    retry_started_ns = run_started_ns + 90_000_000_000
    run_started_at = time.strftime(
        "%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(run_started_ns / 1_000_000_000)
    )

    lineage_started_ns = checkpoint_lineage_started_ns(
        {"attempt": 2, "started_ns": retry_started_ns},
        run_started_at,
    )

    train = tmp_path / "train"
    checkpoint = _checkpoint(train, run_started_ns + 30_000_000_000)
    published = Path(
        get_current_checkpoint(
            str(train), earliest_mtime_ns=lineage_started_ns
        )
    )

    assert lineage_started_ns <= run_started_ns
    assert checkpoint.stat().st_mtime_ns < retry_started_ns
    assert lineage_started_ns < retry_started_ns
    assert published.resolve() == checkpoint


def test_explicit_lineage_cannot_predate_immutable_run():
    run_started_ns = (time.time_ns() // 1_000_000_000 - 60) * 1_000_000_000
    attempt_started_ns = run_started_ns + 30_000_000_000
    run_started_at = time.strftime(
        "%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(run_started_ns / 1_000_000_000)
    )

    with pytest.raises(ValueError, match="immutable run start"):
        checkpoint_lineage_started_ns(
            {
                "attempt": 2,
                "started_ns": attempt_started_ns,
                "lineage_started_ns": run_started_ns - 1,
            },
            run_started_at,
        )
