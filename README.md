# Contact or Contour?

Code and per-image results for the ISBI 2027 submission

> **Contact or Contour? Supervising a Boundary Matters More Than Which Boundary for Separating Touching Nuclei**
> Varun Kumar Dasoju, Qingsu Cheng, Zeyun Yu (University of Wisconsin–Milwaukee)

The paper source and compiled PDF are in [`paper/`](paper/).

## Summary

The paper compares, under matched conditions (same network, loss, selection and decoder; ten training seeds;
image-paired bootstrap CIs), a **contact** target (only the borders where two nuclei touch) against a
**full-contour** target, a semantic network with a tuned distance watershed, and scratch-trained Cellpose. It
uses BBBC039, expanded NucleusSegData (Huh7, HepG2) and 53 external DSB2018 images.

| Effect | BBBC039 | External DSB2018 |
|---|---|---|
| Adding a boundary head (AJI) | +0.027 to +0.041 | +0.035 to +0.036 |
| Adding a boundary head (object F1, pre-specified) | +0.023 to +0.028 | +0.028 to +0.040 |
| Contact vs. contour (AJI) | +0.014 [.011, .018] | +0.001 [−.008, .008] |
| Contact vs. contour (object F1) | +0.005 [.002, .008] | +0.011 [.002, .021] |

Supervising a boundary helped more than the choice of boundary. Scratch-trained Cellpose had the best panoptic
quality on every test set.

## Layout

```
paper/     LaTeX source (main.tex, references.bib, spconf.sty, IEEEbib.bst), figures, compiled main.pdf
src/       breastseg package: network and losses (core.py), data loaders (data.py), metrics (instance_metrics.py)
scripts/   training, post-hoc decoding, statistics and figure scripts
configs/   locked experiment configurations
results/   per-seed and per-image metrics (CSV/JSON) behind every number in the paper
```

The package name `breastseg` is historical (the code base started as a breast-nucleus project); nothing in this
repository uses breast data.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # versions used for the paper; any torch>=2.6 with CUDA should work
export PYTHONPATH=src:scripts
```

Experiments ran on one NVIDIA RTX 5090 (Python 3.12.3, CUDA 12.8). Cellpose baselines use cellpose 3.1.1.2.

## Data

All data are public. Archives are verified against pinned SHA-256 hashes.

```bash
python scripts/download_public_data.py                  # BBBC039 -> data/bbbc039, DSB2018 stage 1 -> data/bbbc038
python scripts/download_nucleusseg.py --release extended # expanded NucleusSegData -> data/nucleusseg_extended
```

BBBC039 uses the official 100/50/50 split. NucleusSegData has no validation set, so post-processing is chosen
on its training images. No test image is used for any selection.

## Reproducing the paper

### 1. Statistics and figures from the saved results (no GPU, no data)

These scripts read `results/` only. Re-running them reproduces the committed JSON files byte for byte.

```bash
python scripts/analyze_boundary_vs_target.py      # boundary head vs. target, AJI and object F1 (Sec. 4.1, 4.4)
python scripts/analyze_boundary_head_effect.py    # boundary head vs. watershed baseline
python scripts/analyze_merge_robustness.py        # merged-touching rates, image and two-way bootstrap
python scripts/analyze_decoder_fix_effect.py      # effect of the decoder correction per target
python scripts/analyze_panel_v6_stats.py          # merge interval, two-way object F1, Huh7/HepG2 t-intervals
python scripts/analyze_tta_cp.py                  # Table 1 final-setting rows
python scripts/analyze_twoway_current.py          # seed x image bootstrap
python scripts/analyze_nsd_star_pair.py           # NucleusSegData widened-grid pair (Table 2, *)
python scripts/analyze_dsb_external_paired.py     # external paired contrasts (Table 3)
python scripts/analyze_astra_final_effects.py     # searched recipe
python scripts/make_isbi_forest.py                # Fig. 3 -> paper/fig_effects.pdf
python scripts/make_isbi_error_profile.py         # Fig. 2 -> paper/fig_error_profile.pdf
```

`scripts/analyze_design_effects.py` (Fig. 3 data) also needs `data/bbbc039`.

### 2. Training (GPU)

Model checkpoints are not included because of their size; the commands below regenerate them.

```bash
# BBBC039, two-head models and semantic baseline (10 seeds each, base protocol, 512^2)
BREASTSEG_BBBC_CONTACT_STUDY=bbbc_contact_10seed     python scripts/train_bbbc_contact.py --data data/bbbc039 --config configs/bbbc_contact_10seed.json
BREASTSEG_BBBC_CONTACT_STUDY=bbbc_contour_10seed     python scripts/train_bbbc_contact.py --data data/bbbc039 --config configs/bbbc_contour_10seed.json
BREASTSEG_BBBC_CONTACT_STUDY=bbbc_carved_10seed      python scripts/train_bbbc_contact.py --data data/bbbc039 --config configs/bbbc_carved_10seed.json
BREASTSEG_BBBC_CONTACT_STUDY=bbbc_semantic_ws_10seed python scripts/train_bbbc_contact.py --data data/bbbc039 --config configs/bbbc_semantic_ws_10seed.json

# Native resolution (epoch budget chosen by the validation-only pilots bbbc_contact_native_pilot_e20/e40)
BREASTSEG_BBBC_CONTACT_STUDY=bbbc_contact_native_10seed python scripts/train_bbbc_contact.py --data data/bbbc039 --config configs/bbbc_contact_native_10seed.json

# Validation-only recipe search (six arms, two seeds), then the locked ten-seed run of the chosen arm
bash scripts/run_astra_search.sh
bash scripts/run_astra_final.sh intensity200

# Cellpose 3 from scratch, auxiliary contact head, contact-weighted loss screen
python scripts/train_cellpose_bbbc.py --seeds 0 1 2 3 4 5 6 7 8 9
python scripts/train_contact_cellpose_bbbc.py --seeds 0 1 2 3 4
python scripts/train_iw_cellpose_bbbc.py --mode search --weight 3 --seeds 0 1    # and --weight 10; rule: configs/iw_cellpose_decision_rule.json

# NucleusSegData (expanded)
python scripts/train_nucleusseg_contact.py --data data/nucleusseg_extended --dataset-version extended --study nucleusseg_extended_contact_10seed --config configs/nucleusseg_extended_contact_10seed.json
python scripts/train_nucleusseg_contact.py --data data/nucleusseg_extended --dataset-version extended --study nucleusseg_extended_contour_10seed --config configs/nucleusseg_extended_contour_10seed.json
python scripts/train_cellpose_nucleusseg.py --seeds 0 1 2 3 4
```

Training writes checkpoints to `checkpoints/` and per-run metrics to `results/<study>/`.

### 3. Decoding, external test and error analysis (needs checkpoints)

```bash
python scripts/posthoc_bbbc_decoding.py --kind semantic --variant vanilla_b7 --locked components \
  --runs $(for s in 0 1 2 3 4 5 6 7 8 9; do echo bbbc_semantic_ws_10seed/vanilla_semantic_ws_b7_s$s; done) \
  --output results/posthoc_bbbc_decoding                 # validation-selected distance watershed
python scripts/analyze_component_preserving.py          # Table 1 base two-head rows (component-preserving decoder)
python scripts/posthoc_tta_component_preserving.py      # Table 1 final-setting rows (8-view TTA)
python scripts/redecode_nucleusseg_cp.py                # Table 2 base rows
python scripts/posthoc_nucleusseg_decoding.py --decoder component_preserving --study nucleusseg_extended_contour_10seed --models standard_contour_b7 \
  --output results/posthoc_nucleusseg_decoding_cp_contour   # Table 2, Full contour*
python scripts/posthoc_nucleusseg_decoding.py --decoder component_preserving --study nucleusseg_extended_contact_10seed --models standard_contact_b7 \
  --output results/posthoc_nucleusseg_decoding_cp           # Table 2, Contact*
python scripts/evaluate_dsb_external.py                 # Table 3 (zero-shot DSB2018)
python scripts/evaluate_dsb_external_decoder.py         # external watershed vs. connected components
python scripts/analyze_bbbc_merge_split.py              # merge/split rates (Fig. 2)
python scripts/analyze_stratified_comparison.py         # touching / small strata (Fig. 2)
python scripts/save_semantic_ws_predictions.py && python scripts/make_isbi_fig1.py   # Fig. 1
python scripts/cellpose_diameter_sensitivity.py         # Cellpose inference-diameter check
```

Each script's docstring states its inputs, selection rule and outputs.

## Where each number comes from

| Paper item | Source |
|---|---|
| Table 1 base rows | `results/bbbc_component_preserving/`, `results/bbbc_carved_component_preserving.json`, `results/posthoc_bbbc_decoding/bbbc_semantic_ws_10seed__*` |
| Table 1 final setting (TTA) | `results/posthoc_tta_cp/`, `results/tta_component_preserving_summary.json` |
| Table 1 searched recipe | `results/bbbc_astra_final_locked_10seed/` (recipe locked in `configs/astra_final_lock.json` before test decoding) |
| Table 1 Cellpose / contact head | `results/cellpose_bbbc_scratch/`, `results/contact_cellpose_bbbc_scratch/` |
| Contact-weighted Cellpose screen | `results/iw_cellpose_search/decision.json` |
| Two-way seed × image bootstrap | `results/twoway_bootstrap_current.json`, `results/panel_v6_stats.json` |
| Boundary head vs. target (AJI, object F1) | `results/boundary_vs_target.json`, `results/boundary_head_effect.json` |
| Fig. 2, Sec. 4.2 | `results/bbbc_stratified_comparison.json`, `results/bbbc_merge_split/`, `results/merge_robustness.json`, `results/small_native.json` |
| Fig. 3 | `results/design_effects.json` |
| Decoder correction | `results/decoder_fix_effect.json`, `results/panel_v6_stats.json` |
| Positive weight (Sec. 2) | `results/positive_weight.json` |
| Table 2 NucleusSegData | `results/nucleusseg_cp/`, `results/posthoc_nucleusseg_decoding_cp/`, `results/posthoc_nucleusseg_decoding_cp_contour/`, `results/nucleusseg_star_pair.json`, `results/cellpose_nucleusseg_scratch/` |
| Table 3 external | `results/dsb2018_external/` (`summary.json`, `paired.json`, `decoder.json`, `per_image.csv`), overlap audit `results/dsb2018_fluorescence_overlap_audit.json` |
| Cellpose diameter sensitivity | `results/cellpose_diameter_sensitivity/summary.json` |

## Protocol notes

- Checkpoints, thresholds and decoders are selected on validation images (NucleusSegData: training images) before
  test images are scored. Contact vs. contour on the base protocol is the confirmatory contrast. All other
  intervals are exploratory and unadjusted.
- An earlier global nearest-marker decoder was found faulty after test scoring: it handed marker-less components
  to neighbouring nuclei. No reported result uses it. Its effect is quantified in `results/decoder_fix_effect.json`.
- All post-hoc analyses (TTA, widened NucleusSegData grid, native resolution, searched recipe, corrected decoder)
  are reported, including the negative ones.

## License

Code is released under the [MIT License](LICENSE). The datasets are not redistributed here; they remain under
their original terms (BBBC039 and DSB2018: Broad Bioimage Benchmark Collection; NucleusSegData: research use
only, from its authors).
