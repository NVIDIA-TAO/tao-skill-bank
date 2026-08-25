import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / "skills/applications/tao-run-deft-iaa/scripts/stage_action_cache.py"
SPEC = importlib.util.spec_from_file_location("stage_action_cache", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


def _request(tmp_path: Path) -> Path:
    cache = tmp_path / "cache"
    model = cache / "huggingface/hub/models--google--siglip2-so400m-patch16-256"
    wanted = model / "snapshots/revision/model.safetensors"
    blob = model / "blobs/digest"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"siglip")
    wanted.parent.mkdir(parents=True)
    wanted.symlink_to("../../blobs/digest")
    (model / "refs").mkdir()
    (model / "refs/main").write_text("revision")
    qwen = cache / "huggingface/hub/models--Qwen--Qwen-Image-Edit-2511/model"
    qwen.parent.mkdir(parents=True)
    qwen.write_bytes(b"qwen")
    entries = []
    for source in (model / "refs/main", wanted):
        entries.append({"path": source.relative_to(cache).as_posix(), "size": source.stat().st_size,
                        "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
    unsigned = {"root": str(cache), "entries": entries}
    request = tmp_path / "action.json"
    request.write_text(json.dumps({"cache_subset": {**unsigned, "sha256": MODULE.sha256_json(unsigned)}}))
    return request


def test_stages_only_bound_siglip_and_is_idempotent(tmp_path):
    request = _request(tmp_path)
    destination = tmp_path / "remote-cache"
    MODULE.stage(request, destination)
    MODULE.stage(request, destination)
    files = [p.relative_to(destination).as_posix() for p in destination.rglob("*") if p.is_file()]
    assert files == [
        "huggingface/hub/models--google--siglip2-so400m-patch16-256/refs/main",
        "huggingface/hub/models--google--siglip2-so400m-patch16-256/snapshots/revision/model.safetensors",
    ]
    assert not any("Qwen" in path or "/xet/" in path for path in files)
    assert sum(p.stat().st_size for p in destination.rglob("*") if p.is_file()) == len(b"revision") + len(b"siglip")


def test_rejects_digest_mismatch_and_changed_source(tmp_path):
    request = _request(tmp_path)
    payload = json.loads(request.read_text())
    payload["cache_subset"]["sha256"] = "0" * 64
    request.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="manifest digest mismatch"):
        MODULE.stage(request, tmp_path / "dest")
    request = _request(tmp_path / "second")
    payload = json.loads(request.read_text())
    source = Path(payload["cache_subset"]["root"]) / payload["cache_subset"]["entries"][0]["path"]
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        MODULE.stage(request, tmp_path / "dest2")


def test_rejects_traversal(tmp_path):
    request = _request(tmp_path)
    payload = json.loads(request.read_text())
    payload["cache_subset"]["entries"][0]["path"] = "../escape"
    unsigned = {"root": payload["cache_subset"]["root"], "entries": payload["cache_subset"]["entries"]}
    payload["cache_subset"]["sha256"] = MODULE.sha256_json(unsigned)
    request.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="unsafe"):
        MODULE.stage(request, tmp_path / "dest")


def test_adapter_request_without_cache_subset_is_explicitly_skipped(tmp_path):
    request = tmp_path / "adapter.action.json"
    request.write_text(json.dumps({"name": "gap_analysis"}))
    with pytest.raises(ValueError, match="must skip cache staging"):
        MODULE.stage(request, tmp_path / "dest")


def test_rejects_a_truncated_staged_copy_before_promotion(monkeypatch, tmp_path):
    request = _request(tmp_path)
    destination = tmp_path / "remote-cache"
    destination.mkdir()
    (destination / "preserved").write_bytes(b"old cache")
    original_copy = MODULE.shutil.copyfile

    def truncated_copy(source, target):
        result = original_copy(source, target)
        target = Path(target)
        if target.name == "model.safetensors":
            target.write_bytes(target.read_bytes()[:2])
        return result

    monkeypatch.setattr(MODULE.shutil, "copyfile", truncated_copy)
    with pytest.raises(ValueError, match="staged cache size mismatch"):
        MODULE.stage(request, destination)
    assert (destination / "preserved").read_bytes() == b"old cache"
