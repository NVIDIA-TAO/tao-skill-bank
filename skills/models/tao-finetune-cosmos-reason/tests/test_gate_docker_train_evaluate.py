import importlib.util
import json
from pathlib import Path
import struct

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "gate_docker_train_evaluate.py"
SPEC = importlib.util.spec_from_file_location("gate_docker_train_evaluate", SCRIPT)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def write_adapter(root: Path, config: dict) -> Path:
    checkpoint = root / "checkpoints" / "stamp" / "safetensors" / "epoch_3"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    header = json.dumps({"layer.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    (checkpoint / "adapter_model.safetensors").write_bytes(
        struct.pack("<Q", len(header)) + header + b"\0\0\0\0"
    )
    return checkpoint


def test_validates_unique_adapter_checkpoint(tmp_path: Path) -> None:
    expected = {"r": 64, "use_rslora": False}
    expected_path = tmp_path / "expected.json"
    expected_path.write_text(json.dumps(expected), encoding="utf-8")
    checkpoint = write_adapter(tmp_path / "results", expected)

    assert gate.find_and_validate_checkpoint(tmp_path / "results", 3, expected_path) == checkpoint


def test_rejects_adapter_metadata_mismatch(tmp_path: Path) -> None:
    expected_path = tmp_path / "expected.json"
    expected_path.write_text(json.dumps({"r": 64}), encoding="utf-8")
    write_adapter(tmp_path / "results", {"r": 32})

    with pytest.raises(ValueError, match="adapter config mismatch"):
        gate.find_and_validate_checkpoint(tmp_path / "results", 3, expected_path)


def test_atomic_json_replaces_complete_document(tmp_path: Path) -> None:
    destination = tmp_path / "state.json"
    gate.atomic_json(destination, {"phase": "training"})
    gate.atomic_json(destination, {"phase": "evaluation", "job": "job-1"})

    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "phase": "evaluation",
        "job": "job-1",
    }
    assert list(tmp_path.glob(".*.tmp")) == []
