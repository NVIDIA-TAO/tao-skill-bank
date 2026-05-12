# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Load CIFAR-10, subsample, rename img→image, save Arrow to data/train + data/eval."""
import argparse, os
from pathlib import Path

import yaml
from datasets import load_dataset, load_from_disk


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    cfg = yaml.safe_load(open(ap.parse_args().config))

    out_train, out_eval = Path("data/train"), Path("data/eval")
    if out_train.exists() and out_eval.exists():
        print("[prepare] Arrow already present"); return

    token = os.environ.get("HF_TOKEN")
    ds = load_dataset(cfg["dataset_id"], token=token)

    src = cfg.get("image_column_src", "img")
    def rename(d):
        return d.rename_column(src, "image") if src != "image" and src in d.column_names else d

    train = rename(ds["train"]).shuffle(seed=42).select(range(min(cfg["n_train"], len(ds["train"]))))
    eval_ = rename(ds["test"]).shuffle(seed=42).select(range(min(cfg["n_eval"], len(ds["test"]))))

    train.save_to_disk(str(out_train))
    eval_.save_to_disk(str(out_eval))
    print(f"[prepare] saved {len(train)} train / {len(eval_)} eval")
    print(f"[prepare] columns: {train.column_names}")
    print(f"[prepare] labels: {train.features[cfg.get('label_column','label')].names}")


if __name__ == "__main__":
    main()
