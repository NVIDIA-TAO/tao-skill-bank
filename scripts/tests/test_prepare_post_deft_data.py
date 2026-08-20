# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "prepare_post_deft_data.py"
SPEC = importlib.util.spec_from_file_location("prepare_post_deft_data", MODULE_PATH)
assert SPEC and SPEC.loader
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def test_build_mixtures_rejects_repeated_mined_rows(tmp_path):
    source = tmp_path / "combined.csv"
    rows = [
        {
            "input_path": "kpi/images/base",
            "golden_path": "kpi/images/golden",
            "label": "PASS",
            "object_name": "C1@1",
        },
        {
            "input_path": "augmentation/mining_pool/images",
            "golden_path": "augmentation/mining_pool/images",
            "label": "PASS",
            "object_name": "C2@1",
        },
        {
            "input_path": "augmentation/mining_pool/images/",
            "golden_path": "augmentation/mining_pool/images/",
            "label": "PASS",
            "object_name": "C2@1",
        },
        {
            "input_path": "results/synthetic_ng",
            "golden_path": "results/synthetic_ok",
            "label": "missing",
            "object_name": "C3@1",
        },
    ]
    with source.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=builder.FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    manifest = builder.build_mixtures(source, tmp_path / "out", (1.0,), 7)

    assert manifest["input_rows"] == 4
    assert manifest["unique_rows"] == 3
    assert manifest["duplicate_rows_rejected"] == 1
    assert manifest["mixtures"]["mix_100"]["rows"] == 3
