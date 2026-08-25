# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical application-local Airflow DAG for typed IAA DEFT actions."""

from __future__ import annotations

import datetime as dt
import os
import pathlib
import sys

from airflow.sdk import dag, get_current_context, task


RUNTIME_ROOT = pathlib.Path(os.environ["TAO_IAA_AIRFLOW_RUNTIME_ROOT"])
if not RUNTIME_ROOT.is_absolute():
    raise RuntimeError("TAO_IAA_AIRFLOW_RUNTIME_ROOT must be absolute")
sys.path.insert(0, str(RUNTIME_ROOT))

from airflow_dag_runtime import dispatch  # noqa: E402


@dag(
    dag_id="tao_deft_iaa_action_v1",
    schedule=None,
    catchup=False,
    tags=["tao-deft-iaa-action-v1", "tao-run-deft-iaa"],
    dagrun_timeout=dt.timedelta(hours=24),
    max_active_runs=1,
)
def tao_deft_iaa_action_v1():
    @task(
        task_id="execute_bound_action",
        retries=0,
        execution_timeout=dt.timedelta(hours=24),
        pool=os.environ.get("AIRFLOW_IAA_COORDINATOR_POOL", "iaa-coordinator"),
    )
    def execute_bound_action() -> dict:
        context = get_current_context()
        return dispatch(context["dag_run"].conf)

    execute_bound_action()


tao_deft_iaa_action_v1()
