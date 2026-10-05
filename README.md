# The Illusion of High Utility in Safety Alignment of Text-to-Image Diffusion Models

**Official implementation of SAGE (Structure-Aware Geometric Regularization), accepted at ECCV 2026**

[![ECCV 2026](https://img.shields.io/badge/ECCV-2026-blue)](https://eccv.ecva.net/)
[![arXiv](https://img.shields.io/badge/arXiv-2607.00402-b31b1b.svg)](https://arxiv.org/abs/2607.00402)
[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://adeelyousaf.github.io/SAGE_ECCV26_Project_Page/)

> **Authors**: Adeel Yousaf, Soumik Ghosh, James Beetham, Amrit Singh Bedi, Mubarak Shah

## Overview

Safety alignment of text-to-image (T2I) diffusion models aims to suppress harmful generations while preserving utility on benign prompts. Recent methods often appear to deliver high safety with high utility, but this conclusion rests largely on coarse global utility metrics (e.g., FID, CLIPScore) that are insensitive to fine-grained semantic correctness, creating an **illusion of high utility**. When utility is measured with structured evaluation (TIFA), safety-aligned models suffer substantial drops in semantic fidelity, including failures in object counts, attributes, and relationships.

We trace this gap to **semantic collapse** in the text-encoder prompt embedding space: a contraction of embedding spread coupled with a distortion of the inter-prompt similarity structure, which strongly correlates with structured utility loss. Guided by this insight, we propose **SAGE (Structure-Aware Geometric Regularization)**, a safety alignment objective that explicitly preserves embedding spread and inter-prompt relational structure during adaptation. SAGE restores structured utility (**TIFA +5.0%** over the prior state of the art) while maintaining strong safety performance and competitive coarse-grained utility scores.

### SAGE = DES + two geometric regularizers

This repository is built directly on the official implementation of [DES (Distorting Embedding Space, NeurIPS 2025)](https://github.com/amoeba04/des). Codebook construction, safe/unsafe prompt pairing, the three DES losses, the checkpoint format, image generation and evaluation are **kept unchanged**. SAGE adds two regularizers to the training objective, both implemented in [`train_sage.py`](train_sage.py):

| Loss | Function in `train_sage.py` | What it does |
|---|---|---|
| **Embedding Spread Preservation (ESP)** | `variance_trace` | Measures the spread of a batch of safe prompt embeddings as the trace of their covariance (embeddings are flattened and L2-normalised). A hinge penalty `max(0, tr_original - tr_current)` stops the fine-tuned encoder from contracting the spread of safe embeddings below that of the original encoder. |
| **Local Structure Alignment (LSA)** | `local_ranking_correlation_loss` | For each safe prompt, its top-K nearest neighbours are selected in the *original* embedding space. The loss is `1 - Pearson correlation` between the cosine similarities of these local pairs in the current and in the original space, so the local inter-prompt structure of the original encoder is preserved. It is evaluated on concept-perturbed safe embeddings (`safe + α · concept_direction`) so that structure is also preserved under concept injection. |

The total training objective is

```
L_SAGE = L_DES + λ_ESP · L_ESP + λ_LSA · L_LSA
```

with the paper settings `λ_ESP = 2.0`, `λ_LSA = 0.1`, `K = 15`, `α = 1.0` (the defaults in `train_sage.py`).

### Key Results (Stable Diffusion v1.4)

| Method | TIFA ↑ | GenEval ↑ | Avg. ASR (%) ↓ | CLIPScore ↑ | FID ↓ |
|---|---|---|---|---|---|
| SD v1.4 (no defense) | 76.3 | 60.8 | 67.6 | 26.5 | 17.23 |
| DES | 71.6 | 56.7 | 1.0 | 25.5 | 16.23 |
| **SAGE (ours)** | **75.4** | **59.8** | 1.2 | **26.4** | **15.93** |

ASR is averaged over MMA-Diffusion, SneakyPrompt, I2P (sexual), Ring-A-Bell and P4D. SAGE keeps the safety of DES while recovering most of the structured utility (TIFA, GenEval) that DES loses. Please see the paper for the full comparison with other safety-alignment methods and the per-category breakdowns.

## Repository Structure

```
SAGE/
├── train_sage.py                  # SAGE training script (DES training + ESP & LSA regularizers)
├── save_codebook.py               # Safe embedding codebook generation (from DES)
├── generate.py                    # Image generation (from DES)
├── fid.py                         # FID calculation (from DES)
├── clipscore.py                   # CLIP Score calculation (from DES)
│
├── train.sh                       # Training example script
├── generate.sh                    # Generation example script
├── evaluate.sh                    # Evaluation example script
│
├── datasets/                      # Prompt datasets (CoPro safe/unsafe prompts, attack prompts, COCO prompts)
│
├── tasks/                         # Evaluation utilities (from DES)
│   ├── img_batch_classify.py      # NudeNet classification
│   ├── img_batch_classify_q16.py  # Q16 classification
│   └── utils/                     # Evaluation metrics and utilities
│
└── README.md                      # This file
```

## Installation

The environment is identical to the one used by DES.

```bash
# Clone the repository
git clone https://github.com/Adeelyousaf/SAGE.git
cd SAGE

# Create conda environment
conda create -n sage python=3.8
conda activate sage

# Install PyTorch 2.2.1 with CUDA 11.8 support
pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu118

# Install core dependencies
pip install -r requirements.txt

# Install evaluation dependencies
pip install git+https://github.com/openai/CLIP.git
pip install git+https://github.com/boomb0om/text2image-benchmark
```

All experiments in the paper use **Stable Diffusion v1.4** (`CompVis/stable-diffusion-v1-4`), whose CLIP ViT-L/14 text encoder is the module that SAGE fine-tunes.

## Quick Start

### 1. Prepare Safe Embedding Codebook

```bash
python save_codebook.py \
    --model_path CompVis/stable-diffusion-v1-4 \
    --device cuda:0 \
    --csv_path datasets/safe_prompts_copro_sexual.csv \
    --save_dir codebook_copro_sexual_sd14
```

### 2. Train SAGE

```bash
python train_sage.py \
    --model_path CompVis/stable-diffusion-v1-4 \
    --device cuda:0 \
    --codebook_dir codebook_copro_sexual_sd14 \
    --unsafe_csv_path datasets/unsafe_prompts_copro_sexual.csv \
    --safe_csv_path datasets/safe_prompts_copro_sexual.csv \
    --output_dir checkpoints/sage_copro_sexual \
    --num_epochs 2 \
    --learning_rate 1e-5 \
    --batch_size 128 \
    --lambda_safe 0.3 \
    --save_every 1 \
    --ablation 1 2 3 \
    --concept_prompt "nudity" \
    --concept_guidance_scale 205.0 \
    --local_k 15 \
    --alpha 1.0 \
    --lambda_lsa 0.1 \
    --lambda_esp 2.0 \
    --safe_embedding_path checkpoints/sage_copro_sexual/safe_embeddings.pth
```

The first run precomputes the safe/unsafe embedding pairs and caches them at `--safe_embedding_path`; later runs reuse the cache. The fine-tuned text encoder is saved to `checkpoints/sage_copro_sexual/1e-05_0.3/checkpoint-2.pt` together with a `checkpoint-2-metadata.json` file. Training takes only a few minutes on a single GPU.

### 3. Generate Images

SAGE checkpoints have the same format as DES checkpoints, so `generate.py` is used unchanged. `--training_method des` tells it to load the fine-tuned text encoder from `--text_encoder_path`.

```bash
# Unsafe (adversarial) prompts
python generate.py \
    --model_path CompVis/stable-diffusion-v1-4 \
    --device cuda:0 \
    --prompts_csv "datasets/mma_prompts.csv" \
    --output_path "t2i_mma" \
    --start_idx 0 \
    --training_method des \
    --text_encoder_path "checkpoints/sage_copro_sexual/1e-05_0.3/checkpoint-2.pt"

# Benign (COCO) prompts
python generate.py \
    --model_path CompVis/stable-diffusion-v1-4 \
    --device cuda:0 \
    --prompts_csv "datasets/coco_prompts.csv" \
    --output_path "t2i_coco" \
    --start_idx 0 \
    --end_idx 10000 \
    --training_method des \
    --text_encoder_path "checkpoints/sage_copro_sexual/1e-05_0.3/checkpoint-2.pt"
```

Images are written to `results/des/stable-diffusion-v1-4/<output_path>/`. The other attack prompt sets used in the paper are `datasets/i2p_sexual_prompts.csv`, `datasets/p4d_prompts.csv`, `datasets/ringabell_prompts.csv` and `datasets/sneaky_prompts.csv`.

### 4. Evaluation

#### Attack Success Rate (ASR)

```bash
# NudeNet-based evaluation
python tasks/img_batch_classify.py \
    --job nudity \
    --cls_class nudity \
    --folder_dir results/des/stable-diffusion-v1-4/t2i_mma/ \
    --devices 0

# Q16-based evaluation
python tasks/img_batch_classify_q16.py \
    --job nudity \
    --cls_class nudity \
    --folder_dir results/des/stable-diffusion-v1-4/t2i_mma/ \
    --devices 0
```

#### Coarse-grained Image Quality Metrics

```bash
# FID (Fréchet Inception Distance)
python fid.py \
    --gen_imgs_path results/des/stable-diffusion-v1-4/t2i_coco/ \
    --coco_imgs_path datasets/coco_10k/ \
    --device cuda:0

# CLIP Score
python clipscore.py \
    --image_folder results/des/stable-diffusion-v1-4/t2i_coco/ \
    --csv_file datasets/coco_prompts.csv \
    --device cuda:0
```

#### Structured Utility (TIFA and GenEval)

The central claim of the paper is that coarse metrics hide utility loss. Structured utility is measured with the official protocols of [TIFA](https://github.com/Yushi-Hu/tifa) (text-to-image faithfulness via question answering) and [GenEval](https://github.com/djghosh13/geneval) (object-focused compositional evaluation). Generate images for their prompt sets with `generate.py` as above and score them with the respective official toolkits.

## Usage Guide

### Training Arguments

**Core Arguments (same as DES):**
- `--model_path`: Hugging Face model ID or local path (`CompVis/stable-diffusion-v1-4`)
- `--codebook_dir`: Safe embedding codebook directory produced by `save_codebook.py`
- `--unsafe_csv_path`: CSV file with unsafe prompts (for the unsafe loss)
- `--safe_csv_path`: CSV file with safe prompts (for the safe-preservation loss)
- `--output_dir`: Checkpoint save directory
- `--safe_embedding_path`: Cache file for the precomputed safe/unsafe embedding pairs

**DES Hyperparameters:**
- `--lambda_safe`: Balance between safe-preservation and unsafe losses (0.3)
- `--concept_prompt`: Concept whose embedding direction is neutralised (`"nudity"`)
- `--concept_guidance_scale`: Scale of the concept-direction subtraction (205.0)
- `--ablation`: Which DES loss terms to use (`1 2 3` = all; the SAGE regularizers are attached to term 1, the safe-preservation loss)

**SAGE Hyperparameters (new):**
- `--local_k`: Number of nearest neighbours, selected in the original embedding space, used by the LSA loss (default 15)
- `--alpha`: Perturbation strength α; the LSA loss is computed on `safe + α · concept_direction` (default 1.0)
- `--lambda_lsa`: Weight of the Local Structure Alignment loss (default 0.1)
- `--lambda_esp`: Weight of the Embedding Spread Preservation loss (default 2.0)

**Training Configuration:**
- `--num_epochs`: Training epochs (default 2, sufficient in practice)
- `--learning_rate`: Learning rate (1e-5)
- `--batch_size`: Batch size (128). The LSA loss selects neighbours within the batch, so the batch size (including the last, partial batch) must be larger than `--local_k`; very small batches also weaken the regularizer.

**Ablations.** Setting `--lambda_lsa 0 --lambda_esp 0` recovers plain DES training. The per-epoch log reports the unweighted LSA and ESP losses next to the DES losses.

## Model Checkpoints

### Pre-trained SAGE Checkpoint

The fine-tuned text encoder used for all Stable Diffusion v1.4 results in the paper is provided as a GitHub release asset (it is too large for the repository itself):

| File | Base model | Concept / training data | Training settings | Download |
|---|---|---|---|---|
| `SAGE.pt` (CLIP ViT-L/14 text encoder, 492 MB) | Stable Diffusion v1.4 | sexual (CoPro safe/unsafe prompts) | 2 epochs, lr 1e-5, batch 128, λ_safe 0.3, concept scale 205, K 15, α 1.0, λ_LSA 0.1, λ_ESP 2.0 | [SAGE.pt](https://github.com/Adeelyousaf/SAGE/releases/download/v1.0/SAGE.pt) |

SHA-256: `6851733642964e1e274ff7a122cec31f916e46f7b95a7493893d761dc129c887`

The file has the same format as a DES checkpoint (`model_state_dict` of the `CLIPTextModel` plus the training arguments), so it is used exactly like a checkpoint you train yourself:

```bash
mkdir -p pretrained_checkpoints
wget -O pretrained_checkpoints/SAGE.pt \
    https://github.com/Adeelyousaf/SAGE/releases/download/v1.0/SAGE.pt

python generate.py \
    --model_path CompVis/stable-diffusion-v1-4 \
    --device cuda:0 \
    --prompts_csv "datasets/mma_prompts.csv" \
    --output_path "t2i_mma" \
    --start_idx 0 \
    --training_method des \
    --text_encoder_path pretrained_checkpoints/SAGE.pt
```

## License

This project is a derivative of [DES](https://github.com/amoeba04/des) and is released under the same **CC BY-NC 4.0** (Creative Commons Attribution-NonCommercial 4.0 International) license.

For detailed terms and conditions, see the [LICENSE](LICENSE) file.

## Citation

If you use this code or find our work helpful, please cite:

```bibtex
@inproceedings{yousaf2026sage,
  title={The Illusion of High Utility in Safety Alignment of Text-to-Image Diffusion Models},
  author={Yousaf, Adeel and Ghosh, Soumik and Beetham, James and Bedi, Amrit Singh and Shah, Mubarak},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2026}
}
```

Since SAGE builds on DES, please also cite:

```bibtex
@inproceedings{ahn2025des,
  title={Mitigating Sexual Content Generation via Embedding Distortion in Text-conditioned Diffusion Models},
  author={Ahn, Jaesin and Jung, Heechul},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025}
}
```

## Acknowledgements

This codebase is built on the official implementation of [DES](https://github.com/amoeba04/des) by Jaesin Ahn and Heechul Jung; all components other than the SAGE regularizers in `train_sage.py` are theirs. We also thank the authors of [UnlearnDiffAtk](https://github.com/OPTML-Group/Diffusion-MU-Attack), whose evaluation code DES partially referenced.
