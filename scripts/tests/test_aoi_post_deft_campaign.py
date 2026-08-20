# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "aoi_post_deft_campaign.py"
SPEC = importlib.util.spec_from_file_location("aoi_post_deft_campaign", MODULE_PATH)
assert SPEC and SPEC.loader
campaign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(campaign)


def test_dry_run_materializes_all_guarded_branches(tmp_path):
    result = campaign.main(
        [
            "--campaign-dir",
            str(tmp_path),
            "--mix50-csv",
            "/remote/mix50.csv",
            "--mix100-csv",
            "/remote/mix100.csv",
            "--images-dir",
            "/remote/images",
            "--incumbent-checkpoint",
            "/remote/deft.pth",
            "--incumbent-far",
            "16.096636665087637",
            "--dry-run",
        ]
    )

    manifest = json.loads((tmp_path / "campaign_manifest.json").read_text())
    tasks = {task["tag"]: task for task in manifest["tasks"]}

    assert result == 0
    assert set(tasks) == {
        "postdeft_warm_full_mix050",
        "postdeft_warm_head_mix050",
        "postdeft_scratch_mix050",
        "postdeft_scratch_mix100",
    }
    assert manifest["gpus_per_trial"] == 8
    assert tasks["postdeft_warm_full_mix050"]["recommendations"] == 8
    assert tasks["postdeft_scratch_mix050"]["recommendations"] == 16
    assert tasks["postdeft_scratch_mix050"]["max_regression"] == 0.0
    assert tasks["postdeft_scratch_mix100"]["initial"] == campaign.SCRATCH100_INITIAL
