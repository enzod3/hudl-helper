# multiview — multi-view football player reconstruction

Hudl Wide (sideline) + End Zone film of one play in, 22 player slots in field yards out: per-frame positions,
identities, team, visibility per camera, uncertainty, a bird's-eye replay. Developed and measured on render05
(Quadro RTX 6000, Turing) against Enzo's hand-tapped ground points on 55 paired plays.

## Layout

```
configs/            default.yaml (tested configuration + model artefacts), field_ncaa.yaml (field geometry)
src/multiview/
  ingestion/        ingest.py         ffprobe-equivalent via PyAV: per-frame pts, fps, audio, sha256
  synchronization/  sync.py           per-view snap from field-masked player motion (camera-motion compensated),
                                      side-by-side preview; stagec.py field-plane offset search (Stage C)
  calibration/      fieldmodel.py     canonical field + paint model + grid drawing
                    lines.py          numbered line candidates for labelling a stadium's first keyframe
                    calibrate.py      H(t): keyframe (labelled lines, or SIFT registration to a calibrated keyframe of
                                      the same camera), robust chamfer fit to the paint mask, propagation by background
                                      camera motion, periodic re-anchoring (rejects jumps > 3 yd), drift metrics
  detection/        detect.py         RF-DETR (COCO or fine-tuned player/official), 2x2 tiles, fp16
                    fieldmask.py      playing-field mask (turf hull + sideline cut + foot-on-turf), apply_fieldmask.py
                    build_*_ds.py, train_ft.py, eval_det.py   fine-tune set assembly, training, held-out evaluation
  segmentation/     segment.py        SAM 3.1 Object Multiplex box-prompted tracking (crop re-prompting for small
                                      players, fp16 + SDPA fallback for pre-Ampere GPUs); second_pass.py re-prompting
  ground_points/    foot_crops.py, train_foot.py, foot_real.py   ground-contact head (foot point + sigma from the box crop)
  tracking/         re-exports: tracklets, linking, constant-velocity Kalman + RTS in yards, IoU tracker
  cross_view/       re-exports: 0.5 s window association, time-disjoint merging, single-view pairing, teams
  reconstruction/   recon.py          the per-play pass: observations -> bias -> teams -> tracklets -> association ->
                                      slot hygiene -> 22 slots -> fusion -> recon.jsonl + replay.mp4
  visualization/    birdseye.py, switch_sheets.py
scripts/            run_pair.py (one play end to end), batch.py, eval_dots.py, fit_fusion.py, summary.py, syn_eval.py
tests/              fieldmodel, homography (line DLT), snap detection, fusion / Kalman / IoU tracker
```

The stage scripts use flat sibling imports; each puts every `src/multiview/*` directory on `sys.path`, so run them
as files (`python src/multiview/calibration/calibrate.py ...`), not as `-m` modules.

## Running one play

Two environments (separate on purpose): a SAM 3.1 env (`sam3` from facebookresearch/sam3 installed editable,
torch 2.7.1+cu126, pycocotools, PyAV, imageio-ffmpeg, safetensors) and an RF-DETR env (`rfdetr[train]==1.11.2`).
Data lives in a work directory (default `~/sam3_work`, the layout render05 uses):

```bash
export MULTIVIEW_WORK=~/sam3_work MULTIVIEW_SAM_PY=~/sam3env/bin/python MULTIVIEW_RFDETR_PY=~/rfdetr_env/bin/python
cd $MULTIVIEW_WORK
$MULTIVIEW_SAM_PY /path/to/multiview/scripts/run_pair.py CLIP WIDE_URL EZ_URL \
    --pre 2.5 --det-weights ft/run_real2/checkpoint_best_total.pth --wide-snap-fallback HUDL_SNAP_T \
    --recon-args "--assoc-gate 2.5 --foot-head foot/run1/head.pt --noise-model calib/headnm/fit_fusion.json"
# single view: pass '-' as EZ_URL
```

Steps (each skipped when its output exists; GPU steps share `logs/gpu.lock`): download -> probe -> frames ->
detect -> field mask -> sync -> calibrate -> Stage C -> SAM track -> recon [-> second SAM pass with --second-pass].
Outputs in `pairs/CLIP/`: `recon/recon.jsonl` (one row per slot per frame: player_id, team, x_yards, y_yards,
covariance, position_confidence, speed, visibility per camera, source fused/sideline/endzone/inferred, estimated,
raw per-view observations), `recon/replay.mp4`, `recon/recon_summary.json`, `<view>/grid_overlay.mp4`, `sync.json`.

A new stadium needs one labelled keyframe per camera (see `lines.py` + a labels JSON as in calibrate.py's
docstring); every other play from that camera seeds itself by SIFT registration.

## Measured

All dot numbers: held-out half of 55 paired NYIT / Susquehanna plays (odd clip ids, 25 plays, 150 hand-tapped
skill-player feet; one play unscorable), Hungarian matching with a 3 yd gate. The dots are mapped to yards through
the Wide calibration, so Wide-only is flattered; End Zone raw is the independent check.

| positions (COCO RF-DETR L, foot head, fitted noise model) | dots matched | median | p95 |
|---|---|---|---|
| fused (recon output) | 121 / 150 | 0.42 yd | 1.97 yd |
| Wide only | 95 | 0.27 yd | 1.89 yd |
| End Zone only / End Zone raw | 96 | 1.32 / 1.42 yd | 4.00 / 4.04 yd |

| detector, full pipeline (foot head, no noise model) | dots matched | median | p95 |
|---|---|---|---|
| COCO RF-DETR L | 118 / 150 | 0.68 yd | 2.51 yd |
| fine-tuned RF-DETR L (real-only v2) | 133 / 150 | 0.64 yd | 2.32 yd |
| fine-tuned + noise model (fitted on COCO-detector observations) | 129 / 150 | 0.45 yd | 2.08 yd |

| observation mode (foot head, no noise model) | dots matched | median | p95 | SAM GPU s / play |
|---|---|---|---|---|
| full SAM 3.1 masks | 118 / 150 | 0.68 yd | 2.51 yd | 272 |
| RF-DETR tracks only | 126 | 0.62 yd | 2.39 yd | 0 |
| adaptive (SAM where boxes overlap) | 124 | 0.73 yd | 2.45 yd | (~40% of detections) |
| single view: Wide only / End Zone only | 115 / 92 | 0.34 / 1.32 yd | 2.34 / 2.68 yd | |

Detector on the held-out NYIT game (149 frames, 6,605 on-field players, 2x2 tiles): fine-tuned real-only v2 vs COCO
RF-DETR L: recall 0.81 vs 0.77, clearly visible 0.96 vs 0.90, crowded line 0.72 vs 0.65, players < 40 px 0.50 vs
0.26, precision 0.68 vs 0.69, false positives 16.9 vs 15.2 per frame. Adding 1,457 synthetic frames gave no gain.

Ground-contact head (144 held-out dots): foot error 0.34 yd median / 1.40 p95, vs 0.75 / 1.90 for the mask-bottom
rule and 0.91 / 1.92 for the box bottom; its sigma is calibrated (93% of errors inside the 95% ellipse).

Tracking on 10 held-out synthetic plays (Wide only, ground-truth calibration, 1.5 yd gate, officials excluded):

| mode | HOTA | DetA | AssA | IDF1 | ID switches | fragmentations | identity complete | yd error median / p95 |
|---|---|---|---|---|---|---|---|---|
| full SAM 3.1 | 0.52 | 0.44 | 0.63 | 0.51 | 29.4 | 47.0 | 24% | 0.66 / 1.28 |
| RF-DETR tracks | 0.49 | 0.45 | 0.55 | 0.48 | 47.9 | 50.9 | 14% | 0.60 / 1.18 |
| adaptive | 0.48 | 0.45 | 0.51 | 0.49 | 55.0 | 82.0 | 9% | 0.61 / 1.21 |

SAM masks matter for identity (AssA, ID switches, completeness), not for position accuracy.

Known limits: End Zone calibration on some pans still drifts (one yard-line period ambiguity); away stadiums need a
labelled keyframe; association recovers ~15/22 players in both views per play; officials in white can survive slot
hygiene.

See ENV.md for model versions and checkpoint hashes. Weights and videos are not in this tree.
