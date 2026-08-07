# CARD 50 — real-image k-NN mining + leakage check — iteration $ITER

HOST FACTS (measured, non-negotiable):
- `embedding image_embeddings` CRASHES on this GPU (sm_75 cuDNN). Run it on CPU: NO `--gpus`, add `-e CUDA_VISIBLE_DEVICES=`. `tmm nearest_neighbors` MAY use `--gpus`.
- Both Hydra CLIs abort without an `experiment_specs/` dir at the container package path `nvidia_tao_ds/mining/{embedding,tmm}/experiment_specs` — the mounts below provide it.
- Pool CSV paths are directory-form; the REAL pool files are flat `{object_name}_SolderLight.jpg` under `$WS/augmentation/mining_pool/images/`.
- Cosine retention cutoff = 0.9 (`state.config.mining_filter.min_similarity`).

STATE GATE:
```bash
bash -c '$DPY -c "import json,pandas as pd;st=json.load(open(\"$RD/deft_state.json\"));p=st[\"iterations\"][\"$ITER\"][\"routing_mining_parquet\"];print(\"ROWS=\"+str(len(pd.read_parquet(p))))"'
```
| gate output | do |
|---|---|
| `ROWS=0` | legal skip: `$DPY $SKILL_ROOT/scripts/commit_stage.py --duration-sec $(( $(date +%s) - STAGE_T0 + 1 )) --results-dir $RD --iter-label $ITER --stage data_mining --skip --summary "routing produced zero mining rows"` (fallback if --skip rejected: `--status ok`, same summary) → `STAGE_DONE 50` |
| `ROWS=N` (N>0) | steps 1→4 |

1) Embed targets + pool (CPU), then k-NN (GPU) — ONE command, three container runs, evidence logs kept. For iter2+ the pool embeddings AND their PASS log are copied from iter1 (the audit requires a real TAO PASS marker in every mining log — never hand-write a log):
```bash
bash -c 'set -e; MF=$RD/$ITER/mining_filter; mkdir -p $MF/pkg_specs/{embedding,tmm}; TP=$($DPY -c "import json;print(json.load(open(\"$RD/deft_state.json\"))[\"iterations\"][\"$ITER\"][\"routing_mining_parquet\"])"); printf "model: SigLIP\nmodel_path: google/siglip-base-patch16-224\nmodel_config_path: \"\"\nbatch_size: 64\ninput_parquet: \"\"\noutput_parquet: \"\"\n" > $MF/pkg_specs/embedding/image_embeddings.yaml; printf "topn: 5\nknn_metric: cosine\nfilter_by_label: false\nsource_embed_column_name: embedding\ntarget_embed_column_name: embedding\nsource_parquet: \"\"\ntarget_parquet: \"\"\noutput_parquet: \"\"\n" > $MF/pkg_specs/tmm/nearest_neighbors.yaml; PKG=/usr/local/lib/python3.12/dist-packages/nvidia_tao_ds/mining; $DPY -c "
import os, pandas as pd
pool=pd.read_csv(\"$WS/augmentation/mining_pool/mining_pool.csv\")
df=pd.DataFrame({\"filepath\":[\"$WS/augmentation/mining_pool/images/\"+str(o)+\"_SolderLight.jpg\" for o in pool[\"object_name\"]],\"label\":pool[\"label\"]})
df=df[df[\"filepath\"].map(os.path.exists)].reset_index(drop=True)
assert len(df)>0, \"pool_files: 0 existing files resolved from mining_pool.csv\"
df.to_parquet(\"$RD/$ITER/mining_filter/pool_files.parquet\")
print(\"pool_files rows:\", len(df))"; docker run --rm --ipc=host -e CUDA_VISIBLE_DEVICES= -v $WS:$WS -v $RD:$RD -v $MF/pkg_specs/embedding:$PKG/embedding/experiment_specs -w $MF $DS_IMG embedding image_embeddings input_parquet=$TP output_parquet=$MF/target_embeddings.parquet > $MF/target_embeddings.log 2>&1; if [ -f $RD/iter1/mining_filter/source_embeddings.parquet ] && [ "$ITER" != iter1 ]; then cp $RD/iter1/mining_filter/source_embeddings.parquet $MF/source_embeddings.parquet; cp $RD/iter1/mining_filter/source_embeddings.log $MF/source_embeddings.log; else docker run --rm --ipc=host -e CUDA_VISIBLE_DEVICES= -v $WS:$WS -v $RD:$RD -v $MF/pkg_specs/embedding:$PKG/embedding/experiment_specs -w $MF $DS_IMG embedding image_embeddings input_parquet=$MF/pool_files.parquet output_parquet=$MF/source_embeddings.parquet > $MF/source_embeddings.log 2>&1; fi; docker run --rm --gpus all --ipc=host -v $WS:$WS -v $RD:$RD -v $MF/pkg_specs/tmm:$PKG/tmm/experiment_specs -w $MF $DS_IMG tmm nearest_neighbors source_parquet=$MF/source_embeddings.parquet target_parquet=$MF/target_embeddings.parquet output_parquet=$MF/mined.parquet > $MF/nearest_neighbors.log 2>&1; tail -n 1 $MF/target_embeddings.log $MF/source_embeddings.log $MF/nearest_neighbors.log; ls $MF/mined.parquet'
```
Every log must end `Execution status: PASS`. A FAIL in an embedding log while `--gpus` was used = the sm_75 trap: rerun that exact step CPU-only as written.

2) Cosine filter (≥0.9) → write BOTH `mined_filtered.parquet` (this is what gets committed; kept rows only) and ChangeNet rows with the path-form rule (file→DIRECTORY collapse on BOTH input_path and golden_path; PASS rows from the OK-only pool stage into mined-input AND mined-golden dirs; object_name must have the `_SolderLight` suffix STRIPPED — the dataloader re-appends it) + `knn_summary.csv` — ONE command:
```bash
bash -c 'set -e; $DPY -c "
import pandas as pd, shutil, pathlib
m=pd.read_parquet(\"$RD/$ITER/mining_filter/mined.parquet\")
simc=[c for c in m.columns if \"sim\" in c.lower() or \"dist\" in c.lower() or \"score\" in c.lower()][0]
cand=len(m.drop_duplicates(subset=[m.columns[0]]))
keep=m[m[simc]>=0.9].drop_duplicates(subset=[m.columns[0]])
keep.to_parquet(\"$RD/$ITER/mining_filter/mined_filtered.parquet\")
run=pathlib.Path(\"$RD\").name; base=pathlib.Path(\"$WS\")
mi=base/f\"results/{run}/$ITER/dataset/images/mined_input\"; mg=base/f\"results/{run}/$ITER/dataset/images/mined_golden\"
mi.mkdir(parents=True,exist_ok=True); mg.mkdir(parents=True,exist_ok=True)
rows=[]
for _,r in keep.iterrows():
    src=pathlib.Path(str(r.iloc[0]))
    if not src.exists(): continue
    obj=src.stem[:-len(\"_SolderLight\")] if src.stem.endswith(\"_SolderLight\") else src.stem
    dst=obj+\"_SolderLight\"+src.suffix
    shutil.copy2(src, mi/dst); shutil.copy2(src, mg/dst)
    rows.append({\"input_path\":f\"results/{run}/$ITER/dataset/images/mined_input\",\"golden_path\":f\"results/{run}/$ITER/dataset/images/mined_golden\",\"label\":\"PASS\",\"object_name\":obj})
pd.DataFrame(rows).to_csv(\"$RD/$ITER/mining_filter/mining_pool.csv\",index=False)
pd.DataFrame([{\"candidate_count\":cand,\"kept_count\":len(rows),\"rejected_count\":cand-len(keep),\"similarity_threshold\":0.9}]).to_csv(\"$RD/$ITER/mining_filter/knn_summary.csv\",index=False)
print(open(\"$RD/$ITER/mining_filter/knn_summary.csv\").read())"'
```

3) Mid-iteration leakage check (hard stop on any hit):
```bash
$DPY $SKILL_ROOT/scripts/validate_training_csv.py --csv $RD/$ITER/mining_filter/mining_pool.csv --workspace-root $WS --validation-csv $WS/train/base/validation_set.csv
```
On leakage: commit data_mining `--status error --summary "train/val leakage in mined rows"` → `STAGE_DONE 50` → stop.

4) Commit data_mining with full evidence — the committed parquet is the FILTERED one (its row count must equal kept_count) — ONE command:
```bash
bash -c 'set -e; MF=$RD/$ITER/mining_filter; K=$($DPY -c "import csv;print(next(csv.DictReader(open(\"$RD/$ITER/mining_filter/knn_summary.csv\")))[\"kept_count\"])"); $DPY $SKILL_ROOT/scripts/commit_stage.py --duration-sec $(( $(date +%s) - STAGE_T0 + 1 )) --results-dir $RD --iter-label $ITER --stage data_mining --mining-parquet $MF/mined_filtered.parquet --mining-count "$K" --mining-summary $MF/knn_summary.csv --mining-source-embeddings $MF/source_embeddings.parquet --mining-target-embeddings $MF/target_embeddings.parquet --mining-source-log $MF/source_embeddings.log --mining-target-log $MF/target_embeddings.log --mining-knn-log $MF/nearest_neighbors.log --summary "mined kept=$K (cosine>=0.9)"'
```
If commit_stage rejects: stderr names the missing evidence flag/file — produce exactly that, rerun. Never bypass.

Final message exactly: `STAGE_DONE 50`
