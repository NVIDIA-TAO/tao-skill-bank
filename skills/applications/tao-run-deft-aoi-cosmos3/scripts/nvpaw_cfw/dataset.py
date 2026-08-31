"""Random-access NVPAW JSONL dataset that preserves multi-image semantics."""

from __future__ import annotations

import array
import fcntl
import hashlib
import json
import os
from pathlib import Path
from typing import Any, BinaryIO

try:
    from torch.utils.data import Dataset as TorchDataset
except ImportError:  # Host-side contract tests do not require the runtime.
    class TorchDataset:  # type: ignore[no-redef]
        pass

from .artifacts import (
    INDEX_FORMAT,
    PIXEL_POLICY,
    index_metadata_path,
    index_status,
    load_index_metadata,
    source_fingerprint,
)


def build_jsonl_index(
    jsonl_path: str | Path,
    index_path: str | Path,
    expected_rows: int,
    expected_sha256: str,
    expected_image_items: int,
) -> dict[str, Any]:
    source = Path(jsonl_path).expanduser()
    target = Path(index_path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"NVPAW JSONL does not exist: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.with_suffix(target.suffix + ".lock")
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        valid, _, cached = index_status(
            source, target, expected_rows, expected_sha256, expected_image_items
        )
        if valid and cached is not None:
            return cached

        offsets = array.array("Q")
        digest = hashlib.sha256()
        byte_offset = 0
        image_items = 0
        min_values: list[int] = []
        max_values: list[int] = []
        with source.open("rb") as stream:
            for row_number, raw_line in enumerate(stream, start=1):
                offsets.append(byte_offset)
                byte_offset += len(raw_line)
                digest.update(raw_line)
                try:
                    row = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{source}:{row_number}: invalid JSON") from exc
                if not isinstance(row, dict) or not isinstance(row.get("messages"), list):
                    raise ValueError(f"{source}:{row_number}: messages must be a list")
                for message in row["messages"]:
                    content = message.get("content") if isinstance(message, dict) else None
                    if not isinstance(content, list):
                        continue
                    for item in content:
                        if not isinstance(item, dict) or item.get("type") != "image":
                            continue
                        image_items += 1
                        low = item.get("min_pixels")
                        high = item.get("max_pixels")
                        if type(low) is not int or type(high) is not int:
                            raise TypeError(
                                f"{source}:{row_number}: image min_pixels/max_pixels must be integers"
                            )
                        if low <= 0 or high <= 0 or low > high:
                            raise ValueError(
                                f"{source}:{row_number}: invalid pixel bounds {low}/{high}"
                            )
                        min_values.append(low)
                        max_values.append(high)
        actual_sha256 = digest.hexdigest()
        if len(offsets) != expected_rows:
            raise ValueError(f"expected {expected_rows} rows, found {len(offsets)}")
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"JSONL SHA256 changed: expected {expected_sha256}, found {actual_sha256}"
            )
        if image_items != expected_image_items:
            raise ValueError(f"expected {expected_image_items} image items, found {image_items}")
        if not image_items:
            raise ValueError("NVPAW JSONL must contain image items")

        temporary_index = target.with_suffix(target.suffix + f".tmp.{os.getpid()}")
        metadata_path = index_metadata_path(target)
        temporary_metadata = metadata_path.with_suffix(
            metadata_path.suffix + f".tmp.{os.getpid()}"
        )
        metadata = {
            **source_fingerprint(source),
            "row_count": len(offsets),
            "sha256": actual_sha256,
            "image_items": image_items,
            "min_pixels_range": [min(min_values), max(min_values)],
            "max_pixels_range": [min(max_values), max(max_values)],
            "index_format": INDEX_FORMAT,
            "pixel_policy": PIXEL_POLICY,
        }
        try:
            with temporary_index.open("wb") as output:
                offsets.tofile(output)
                output.flush()
                os.fsync(output.fileno())
            with temporary_metadata.open("w", encoding="utf-8") as output:
                json.dump(metadata, output, indent=2, sort_keys=True)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary_index, target)
            os.replace(temporary_metadata, metadata_path)
        finally:
            for temporary in (temporary_index, temporary_metadata):
                if temporary.exists():
                    temporary.unlink()
        return metadata


class NVPAWJsonlDataset(TorchDataset):
    """Map-style protocol consumed by the Cosmos Framework data distributor."""

    def __init__(
        self,
        jsonl_path: str | Path,
        media_root: str | Path,
        index_path: str | Path,
        expected_rows: int,
        expected_sha256: str,
        expected_image_items: int,
    ) -> None:
        self._source_file: BinaryIO | None = None
        self._source_pid: int | None = None
        self.jsonl_path = str(Path(jsonl_path).expanduser().resolve())
        resolved_media_root = Path(media_root).expanduser().resolve(strict=True)
        if not resolved_media_root.is_dir():
            raise NotADirectoryError(f"NVPAW media root is not a directory: {resolved_media_root}")
        self.media_root = str(resolved_media_root)
        self.index_path = str(Path(index_path).expanduser().resolve())
        self.expected_rows = int(expected_rows)
        self.expected_sha256 = expected_sha256
        self.expected_image_items = int(expected_image_items)
        build_jsonl_index(
            self.jsonl_path,
            self.index_path,
            self.expected_rows,
            self.expected_sha256,
            self.expected_image_items,
        )
        offsets = array.array("Q")
        with open(self.index_path, "rb") as stream:
            offsets.fromfile(stream, self.expected_rows)
        if len(offsets) != self.expected_rows:
            raise ValueError("offset index row count changed after validation")
        self._offsets = offsets

    def __len__(self) -> int:
        return self.expected_rows

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_source_file"] = None
        state["_source_pid"] = None
        return state

    def close(self) -> None:
        """Close this worker's lazily opened source file."""

        if getattr(self, "_source_file", None) is not None:
            self._source_file.close()
            self._source_file = None
            self._source_pid = None

    def __del__(self) -> None:
        self.close()

    def _file(self) -> BinaryIO:
        pid = os.getpid()
        if self._source_file is None or self._source_pid != pid:
            if self._source_file is not None:
                self._source_file.close()
            self._source_file = open(self.jsonl_path, "rb")
            self._source_pid = pid
        return self._source_file

    def __getitem__(self, index: int) -> dict[str, Any]:
        if not 0 <= int(index) < len(self):
            raise IndexError(index)
        stream = self._file()
        stream.seek(self._offsets[int(index)])
        raw_line = stream.readline()
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at source index {index}") from exc
        if not isinstance(row, dict) or not isinstance(row.get("messages"), list):
            raise TypeError(f"source row {index} must contain messages")
        media: dict[str, bytes] = {}
        messages: list[dict[str, Any]] = []
        image_number = 0
        for message in row["messages"]:
            if not isinstance(message, dict):
                raise TypeError(f"source row {index} contains a non-object message")
            rewritten = dict(message)
            content = message.get("content")
            if isinstance(content, list):
                rewritten_content: list[Any] = []
                for item in content:
                    if not isinstance(item, dict) or item.get("type") != "image":
                        rewritten_content.append(item)
                        continue
                    image_path = item.get("image")
                    if not isinstance(image_path, str) or not image_path:
                        raise TypeError(f"source row {index} has an invalid image path")
                    path = Path(image_path).expanduser()
                    if not path.is_absolute():
                        path = Path(self.media_root) / path
                    path = path.resolve()
                    if not path.is_file():
                        raise FileNotFoundError(f"source row {index} image is missing: {path}")
                    key = f"image_{image_number:02d}"
                    image_number += 1
                    media[key] = path.read_bytes()
                    rewritten_item = dict(item)
                    rewritten_item["image"] = key
                    rewritten_content.append(rewritten_item)
                rewritten["content"] = rewritten_content
            messages.append(rewritten)
        if not media:
            raise ValueError(f"source row {index} has no input images")
        return {
            "texts": messages,
            "media": media,
            "_nvpaw_source_index": int(index),
            "_nvpaw_row_id": str(row.get("id", "")),
        }
