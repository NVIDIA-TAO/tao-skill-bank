# LAM SegFormer fusion and checkpoint-soup results

Selection used only the 1,262-image labeled `val` split. The image-only test
split was not used for candidate selection or metric computation.

Every three-model weight vector is ordered as:

1. original-data AutoML BFBO;
2. standalone DEFT mix25;
3. AutoML BFBO trained on the fixed DEFT mix25 snapshot.

Because the retained-best validation epoch was not always a durable checkpoint
epoch, each source checkpoint is the valid saved checkpoint nearest to its
retained-best validation epoch. Every single-model baseline below was rescored
from that exact checkpoint in the same distributed scorer as its combinations.

| Backbone | Method | Best weights | Validation mIoU | Best exact single baseline | Gain |
|---|---|---:|---:|---:|---:|
| FAN-base | Probability fusion | `[0.34, 0.33, 0.33]` | **94.92203%** | 94.80116% | +0.12088 pp |
| FAN-base | Class-rank fusion | `[0.50, 0.25, 0.25]` | 94.91480% | 94.80116% | +0.11365 pp |
| FAN-base | Checkpoint soup | `[0.00, 0.00, 1.00]` | 94.80116% | 94.80116% | +0.00000 pp |
| FAN-large | Probability fusion | `[0.50, 0.25, 0.25]` | **94.87958%** | 94.74873% | +0.13085 pp |
| FAN-large | Class-rank fusion | `[0.50, 0.25, 0.25]` | 94.87562% | 94.74873% | +0.12689 pp |
| FAN-large | Checkpoint soup | `[0.00, 0.00, 1.00]` | 94.74873% | 94.74873% | +0.00000 pp |
| MiT-B5 | Probability fusion | `[0.70, 0.30, 0.00]` | **94.79783%** | 94.64462% | +0.15322 pp |
| MiT-B5 | Class-rank fusion | `[0.50, 0.25, 0.25]` | 94.70997% | 94.64462% | +0.06535 pp |
| MiT-B5 | Checkpoint soup | `[1.00, 0.00, 0.00]` | 94.64462% | 94.64462% | +0.00000 pp |

The campaign leader is FAN-base near-uniform probability fusion at 94.92203%
validation mIoU. Prediction fusion improved every backbone; direct parameter
averaging improved none of them.

The first six launch attempts failed before inference because direct
`SegFormerPlModel` construction received training-only specs without the Hydra
default `evaluate` and `inference` sections. The corrected scorer supplies the
minimal runtime defaults. The launcher also validates result artifacts before
accepting backend `COMPLETE`, preventing a wrapper-level zero exit status from
masking an application failure. Full provenance is recorded in
`downstream_fusion_soup_runtime_correction.json`.
