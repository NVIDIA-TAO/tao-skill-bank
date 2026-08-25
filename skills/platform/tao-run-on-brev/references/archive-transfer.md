# Large Brev Tree Transfer

Read this reference when a committed Brev result or dataset tree contains too
many files for a file-by-file transfer, or when model-cache content evidence is
required.

## Approval and canonical archive

Do not hand-assemble a `tar` pipeline. Use the packaged
`scripts/brev_archive_transfer.py` helper from the exact staged skill snapshot.
The launch review must state the source, destination, explicit member/byte
limits, transfer archive location, and whether an existing destination will be
recoverably replaced.

Run `pack` on the Brev instance, transfer only the resulting archive and JSON
receipt, then run `extract` locally:

```bash
TRANSFER="$BANK/skills/platform/tao-run-on-brev/scripts/brev_archive_transfer.py"
python3 "$TRANSFER" pack \
  --source "$REMOTE_SOURCE" --archive "$REMOTE_ARCHIVE" \
  --receipt "$REMOTE_RECEIPT" \
  --max-members "$APPROVED_MAX_MEMBERS" --max-bytes "$APPROVED_MAX_BYTES"

python3 "$TRANSFER" extract \
  --archive "$LOCAL_ARCHIVE" --pack-receipt "$LOCAL_PACK_RECEIPT" \
  --destination "$LOCAL_DESTINATION" --output-receipt "$LOCAL_EXTRACT_RECEIPT" \
  --max-members "$APPROVED_MAX_MEMBERS" --max-bytes "$APPROVED_MAX_BYTES"
```

### Immutable request snapshots

Always use this archive path for a signed request, controller, patches, or
other immutable input snapshot. Those snapshot directories may intentionally
be mode `0500`. A recursive `brev copy`, `scp -r`, or directory rsync can apply
that mode to the destination directory before creating its children, leaving a
partial tree that cannot be completed. Do not chmod, copy file-by-file, or
rebuild the snapshot manually. `pack` reads the immutable source without
mutating it; `extract` validates the receipt and complete member graph in a
writable staging directory before atomically publishing the destination.

Add `--replace-existing` only when the launch review approved replacement. The
helper preserves the old destination as a digest-named backup; it never deletes
that backup. Packing rejects non-regular objects and every link except an exact
relative file symlink whose fully resolved target is an existing regular file
inside the source root through ordinary directory parents. It normalizes
archive metadata without rewriting accepted relative link text, SHA-256 binds
the completed archive, and atomically publishes the archive and receipt.
Extraction verifies that receipt and digest before writing, validates the
complete member graph first, rejects absolute/traversing, duplicate, cyclic,
escaping, hardlink, device, unsafe-parent, and unsupported members, extracts
regular files before creating accepted links, enforces both limits, and
atomically promotes only exact member and byte totals.

A failed pack leaves no published archive. A failed extraction removes only its
new staging tree and restores an existing destination if promotion had begun.
An existing staging or backup path fails closed: inspect it and preserve or move
it explicitly before retrying. Completion requires both receipts, matching
archive SHA-256, exact totals, and the canonical destination.

## Bounded parallel chunk transfer

When one archive stream is a measured transfer bottleneck, use deterministic
fixed-size chunks instead of inventing a second archive format. Keep the chunk
manifest outside `REMOTE_CHUNKS`; `split` binds the pack receipt, whole archive,
ordered offsets, and every chunk size and SHA-256. Approve `CHUNK_BYTES` and
`MAX_CHUNKS` explicitly. A retry verifies and reuses each completed chunk and
atomically replaces only a missing chunk; changed or unexpected files fail
closed.

```bash
python3 "$TRANSFER" split \
  --archive "$REMOTE_ARCHIVE" --pack-receipt "$REMOTE_RECEIPT" \
  --output-dir "$REMOTE_CHUNKS" --manifest "$REMOTE_CHUNK_MANIFEST" \
  --chunk-bytes "$CHUNK_BYTES" --max-chunks "$MAX_CHUNKS"
```

Transfer the manifest first, then use the packaged transfer verb. It parses only
the exact ordered `chunks[].name` values and uses **at most four** concurrent
rsync streams. Each stream owns a different manifest-declared destination and
uses `--partial --append-verify` for bounded resume. The helper explicitly sets
`ControlMaster=no`, `ControlPath=none`, and `ControlPersist=no`; otherwise an
SSH client configuration can silently multiplex all workers through one TCP
connection and defeat parallel transfer.

```bash
python3 "$TRANSFER" transfer-chunks \
  --manifest "$LOCAL_CHUNK_MANIFEST" \
  --remote-host "$BREV_SSH_ALIAS" \
  --remote-chunks-dir "$REMOTE_CHUNKS" \
  --local-chunks-dir "$LOCAL_CHUNKS" \
  --output-receipt "$LOCAL_TRANSFER_RECEIPT" \
  --max-chunks "$MAX_CHUNKS" --max-chunk-bytes "$CHUNK_BYTES" \
  --streams 4 --timeout-seconds 7200
```

Do not derive names from a directory listing, hand-build an `xargs` pipeline,
or start a whole-archive writer at the same time. A retry reuses only valid
completed chunks and manifest-owned partial destinations. Do not join until the
transfer verb is terminal and its receipt exists.
The timeout is per chunk attempt. On timeout or nonzero rsync exit, the helper
publishes no receipt and retains only the manifest-owned partials; rerun the
same command once connectivity is healthy.

Reassemble with the packaged helper. It rejects a missing, extra, duplicate,
out-of-order, oversized, or tampered chunk before writing, joins into a
digest-scoped temporary file, verifies the whole archive SHA-256 and size, and
atomically publishes the archive:

```bash
python3 "$TRANSFER" join \
  --chunks-dir "$LOCAL_CHUNKS" --manifest "$LOCAL_CHUNK_MANIFEST" \
  --archive "$LOCAL_ARCHIVE" --output-receipt "$LOCAL_JOIN_RECEIPT" \
  --max-chunks "$MAX_CHUNKS" --max-chunk-bytes "$CHUNK_BYTES"
```

Keep chunks by default for interruption recovery. Add `--cleanup-chunks` only
after successful reassembly when the approved policy permits cleanup; it
removes only the exact digest-bound chunk files after the archive and join
receipt are durable. Preserve the manifest. Remove remote chunks only after the
local join and extraction receipts pass, and only by their exact manifest names.

## Model-cache evidence

For downloaded model evidence, run `digest-tree` against the exact immutable
revision directory and its approved cache root. File links are accepted only
when they resolve to regular blobs inside that root; directory links, escapes,
empty trees, and limit overruns fail closed. Preserve the resulting
content-based tree receipt with the endpoint preflight evidence.
