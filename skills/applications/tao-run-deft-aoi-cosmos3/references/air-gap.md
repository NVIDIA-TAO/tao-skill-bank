# Air-gap contract

Before entering an air gap, stage the immutable Cosmos Framework and
data-services images, the complete Qwen3-VL snapshot, all three annotation
JSONL files and referenced images, the exact evaluator, Python dependencies,
and the application/model/platform skill files. Verify hashes outside and
inside the boundary.

In air-gap mode, state freezes `allow_package_install`, `allow_remote_fetch`,
`allow_container_pull`, and `allow_registry_login` to false. Use only local
image digests and local absolute model/data paths. A missing asset or native
CLI is a hard stop; do not probe a remote endpoint or silently change modes.
