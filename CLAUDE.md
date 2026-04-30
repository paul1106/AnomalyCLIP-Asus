# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AnomalyCLIP (ICLR 2024) is a zero-shot anomaly detection framework that adapts CLIP for detecting and segmenting anomalies across industrial and medical imaging domains. The key idea is learning object-agnostic text prompts that capture generic normality/abnormality regardless of foreground object class.

## Commands

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Prepare a dataset (generates `meta.json` for data loading):**
```bash
cd generate_dataset_json
python mvtec.py   # or visa.py, SDD.py, btad.py, etc.
```

**Train (prompt learner only; CLIP backbone is frozen):**
```bash
bash train.sh
# Or directly:
CUDA_VISIBLE_DEVICES=0 python train.py --dataset visa --train_data_path /path/to/Visa \
  --save_path ./checkpoints/9_12_4_multiscale/ \
  --features_list 24 --image_size 518 --batch_size 8 \
  --epoch 15 --depth 9 --n_ctx 12 --t_n_ctx 4
```

**Test on a full dataset:**
```bash
bash test.sh
# Or directly:
CUDA_VISIBLE_DEVICES=0 python test.py --dataset mvtec \
  --data_path /path/to/mvdataset \
  --checkpoint_path ./checkpoints/9_12_4_multiscale/epoch_15.pth \
  --save_path ./results/ \
  --features_list 6 12 18 24 --image_size 518 --depth 9 --n_ctx 12 --t_n_ctx 4
```

**Test a single image:**
```bash
python test_one_example.py \
  --image_path /path/to/image.png \
  --checkpoint_path ./checkpoints/9_12_4_multiscale/epoch_15.pth \
  --features_list 6 12 18 24 --image_size 518 --depth 9 --n_ctx 12 --t_n_ctx 4
```

## Architecture

### Core components

**`AnomalyCLIP_lib/`** — Modified CLIP (ViT-L/14@336px only):
- `AnomalyCLIP.py`: Core model. `VisionTransformer.DAPM_replace(DPAM_layer=20)` swaps the last 20 ViT attention blocks from standard self-attention to v-v self-attention (`Attention` class), enabling DPAM (Dual Path Attention Module). `ResidualAttentionBlock_learnable_token` handles injection of compound text prompts at intermediate text encoder layers.
- `model_load.py`: `load()` downloads/loads ViT-L/14@336px. **Note:** cache path is hardcoded to `/remote-home/iot_zhouqihang/root/.cache/clip` — update this for local environments. Also exports `compute_similarity` and `get_similarity_map` used by inference code.

**`prompt_ensemble.py`** — `AnomalyCLIP_PromptLearner`: The only trainable module. Holds:
- `ctx_pos` / `ctx_neg`: learnable context tokens for normal/anomaly text prompts (shape `[1, 1, n_ctx, dim]`)
- `compound_prompts_text`: list of `(depth-1)` learnable embeddings injected at each text transformer layer beyond the first
- `compound_prompt_projections`: projection layers (unused during standard training but present)

Forward pass returns `(prompts, tokenized_prompts, compound_prompts_text)` for `model.encode_text_learn()`.

**`dataset.py`** — `Dataset` loads from `{root}/meta.json`. The JSON must have structure:
```json
{"test": {"class_name": [{"img_path": "...", "mask_path": "...", "cls_name": "...", "specie_name": "...", "anomaly": 0|1}]}}
```
`generate_class_info(dataset_name)` maps dataset string names to their class lists — add new datasets here.

**`generate_dataset_json/`** — One script per supported dataset to generate `meta.json`. Use these as templates for custom datasets.

### Training flow (`train.py`)

1. Load frozen CLIP + `AnomalyCLIP_PromptLearner`
2. Call `model.visual.DAPM_replace(DPAM_layer=20)` to activate DPAM
3. Per batch: extract `image_features` (global) and `patch_features` (multi-layer) from frozen visual encoder
4. Compute text features from trainable prompts via `model.encode_text_learn()`
5. Loss = `lam * (FocalLoss + 2×BinaryDiceLoss on patch similarity maps)` + `CrossEntropy on image-level prediction`
6. Only `prompt_learner` parameters are optimized
7. Checkpoint saved as `{"prompt_learner": state_dict}`

### Inference flow (`test.py`)

- Features extracted from layers specified by `--features_list` (default `[6, 12, 18, 24]`)
- Anomaly map = sum of per-layer `(similarity_to_abnormal + 1 - similarity_to_normal) / 2`
- Post-processed with Gaussian filter (`--sigma 4`)
- Metrics reported: `image-auroc`, `image-ap`, `pixel-auroc`, `pixel-aupro`

### Key hyperparameters

| Arg | Default | Meaning |
|-----|---------|---------|
| `--depth` | 9 | Number of text transformer layers receiving compound prompts |
| `--n_ctx` | 12 | Length of learnable context tokens in text prompts |
| `--t_n_ctx` | 4 | Length of compound prompts injected per layer |
| `--features_list` | `[6,12,18,24]` | ViT layers to extract patch features from |
| `--feature_map_layer` | `[0,1,2,3]` | Indices into features_list to include in anomaly map |
| `--image_size` | 518 | Input resolution (ViT-L/14@336px trained at 336; 518 used at test time with interpolated positional embeddings) |
| `--sigma` | 4 | Gaussian smoothing sigma for anomaly maps |

### Checkpoint naming convention

`checkpoints/{depth}_{n_ctx}_{t_n_ctx}_multiscale/epoch_{N}.pth` (trained on VisA)  
`checkpoints/{depth}_{n_ctx}_{t_n_ctx}_multiscale_visa/epoch_{N}.pth` (trained on MVTec)

The paper's main result uses weights trained once on VisA and tested zero-shot on all other datasets (and vice versa for MVTec).
