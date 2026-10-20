# Environment and model artefacts (as run on render05, Quadro RTX 6000 24 GB, Turing sm_75, driver 550.107.02)

## SAM 3.1 env
- Python 3.12.13, torch 2.7.1+cu126, torchvision 0.22.1+cu126
- facebookresearch/sam3 @ `7f473cfd983d5eb43040ff608f66bff0d7bdcb53` (installed editable)
- Weights: HF mirror `AEmotionStudio/sam3.1` @ `694239a1479aab8fd1317c87c433c58acd7c6eab`
  - `sam3.1_multiplex.safetensors` sha256 `684ab5985ac69d8bb7a23b29e5acbdb09270c23908b721a54969cb3cb33f0cce`
  - `sam3.1_multiplex.pt` sha256 `0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6` (identical tensors)
  - License: Meta SAM License (2025-11-19); mirror LICENSE identical to the repo's.
- Turing: fp16 autocast only (bf16 OOMs without flash SDPA, fp32 hits hard bf16 casts); segment.py patches the
  decoder's forced flash-attention SDPA backend at runtime.
- Also: pycocotools, PyAV 19, imageio-ffmpeg 0.6, safetensors, opencv-python-headless 4.10, scipy 1.14.

## RF-DETR env
- Python 3.12.15, `rfdetr[train]==1.11.2`, pytorch-lightning 2.6.6, torch 2.7.1+cu126, supervision 0.31
- COCO weights `rf-detr-large-2026.pth` sha256 `0f4e20e19a99c0f8a62b5685f57f6c8b5c371c59081feda6752a0561a79ccf38`
  (Apache-2.0)
- Fine-tuned (player / official), real-only v2: `ft/run_real2/checkpoint_best_total.pth`
  sha256 `69f8d604e28bbe2a59bce0841ceadd86089ee9e060455ee24578c0f5e51d8a53`
  - data: 1,903 pseudo-labelled frames (Susquehanna plays; SAM track class + RF-DETR box geometry) + 250 SAM 3
    labelled frames (non-NYIT games); valid: 155 Susquehanna SAM 3 frames; test: 149 NYIT frames (held-out game)
- Synthetic + real v2: `ft/run_synreal2/checkpoint_best_total.pth`
  sha256 `436f3015f22d163be939b591ac4cb4849f7b654f0dd5747f370b83e7c9308d35` (+1,457 Unreal frames; no gain)

## Ground-contact head
- `foot/run1/head.pt` sha256 `ef55e39d968861f877ba5bd449e5ce54016df28826a936dddbe5e6951cb7324a`
  - small CNN on the 64x128 box crop -> foot point + sigma; 106k synthetic crops, fine-tuned on 170 dots (even clip
    ids), evaluated on 144 (odd)

## Noise model / fusion
- `calib/headnm/fit_fusion.json`: per-view sigma along / across each camera's depth axis vs distance from the
  camera, plus per-view bias, fitted on even clip ids.

## Data conventions
- Field: x 0..120 yd (goal lines at 10 / 110), y 0..53.33 yd, y = 0 on the Wide camera's side; NCAA hashes.
- Sync: `wide_time = ez_time + offset`; recon time `t` is seconds from each view's detected snap.
