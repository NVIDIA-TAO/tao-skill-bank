# SynWTS to DEFT ITS adapter

`scripts/synwts_to_llava.py` turns the public SynWTS dataset (`mlcglab/synwts` on Hugging Face,
CC-BY-SA-4.0, the Isaac Sim digital twin of the Woven Traffic Safety dataset) into this skill's
binary video-QA records: 33-frame overhead-view windows centred on the five annotated phases of
each scenario, four yes/no questions per window derived from the per-frame boxes, the phase
captions and the VQA conversations, LLaVA-style annotations, and the media cuts. Windows whose
label a rule cannot decide are skipped rather than guessed; the manifest records which clause
fired for every label.

Stages (one entry point, driven by `scripts/sim2real_spec.json`):

```bash
python scripts/synwts_to_llava.py build  --spec scripts/sim2real_spec.json   # windows + labels + annotations + manifest, no media
python scripts/synwts_to_llava.py media  --spec scripts/sim2real_spec.json --sets train,kpi_sim   # cut the clips (ffmpeg on PATH)
python scripts/synwts_to_llava.py report --spec scripts/sim2real_spec.json   # label and skip table from the manifest
```

Edit the spec before the first run: `dataset.root`, `dataset.annotations` and `dataset.videos`
point at your SynWTS download; `workspace` is the DEFT workspace that receives
`data/<set>/annotations.json` and `data/<set>/media`; `windows.holdout_scenarios` scenarios
are held out as the simulated KPI set. Keep media out of any git tree.

Validated on an RTX Spark N1X (native Windows on Arm) and used for the SynWTS sim-to-real
stages of the ITS work in October and November 2026; standard library only, plus ffmpeg for
the media stage.
