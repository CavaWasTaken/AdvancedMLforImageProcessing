# 🖼️ Advanced Machine Learning for Image Processing
### Laboratory Reports — A.Y. 2025/2026 | Politecnico di Torino

Course laboratory work for the **Advanced Machine Learning for Image Processing** course (ICT Engineering for Smart Societies). Four labs covering image restoration, 3D point cloud classification, multimodal image search, and generative diffusion models.

**Students:** Lorenzo Cavallaro (s346742) · Emanuele Cumino (s361790)

---

## 📋 Table of Contents

- [Lab 1 — Image Denoising & Super-Resolution](#lab-1----image-denoising--super-resolution)
- [Lab 2 — Point Cloud Classification with GNNs](#lab-2----point-cloud-classification-with-gnns)
- [Lab 3 — CLIP Image Search Engine](#lab-3----clip-image-search-engine)
- [Lab 4 — Stable Diffusion](#lab-4----stable-diffusion)
- [Getting Started](#getting-started)

---

## Lab 1 — Image Denoising & Super-Resolution

### Exercise 1: Grayscale Image Denoising (DnCNN)

A shortened DnCNN model is trained to remove additive white Gaussian noise (AWGN) from grayscale images using a patch-based strategy (32×32 random crops).

**Architecture:** Input conv → 8× Conv-BN-ReLU blocks → Output conv (C=64 channels)  
**Residual formulation:** `x̂ = x − f(x)` — predicts and subtracts the noise residual  
**Training:** Adam optimizer, MSE loss, 200 epochs, batch size 16, `ReduceLROnPlateau` scheduler on validation PSNR

**Key ablations:**
- Batch size 16 outperformed larger batches for generalization
- BatchNorm > GroupNorm (GroupNorm overfit on this dataset size)
- C=64 channels: best complexity/generalization trade-off

**Results (σ = 25):**

| Metric | Residual Model | Non-Residual Model |
|---|---|---|
| Test loss (MSE) | 0.0013 | 0.0016 |
| Avg denoised PSNR (dB) | **28.88** | 28.12 |
| Best validation PSNR (dB) | **30.10** | 29.33 |

**Out-of-distribution generalization (fixed σ=25 model):**

| Noise std | Noisy PSNR (dB) | Denoised PSNR (dB) |
|---|---|---|
| σ = 10 | 28.16 | 30.06 |
| σ = 50 | 14.75 | 18.22 |

**Blind denoising (σ ∼ U(10, 50)):**

| Noise std | Noisy PSNR (dB) | Denoised PSNR (dB) |
|---|---|---|
| σ = 10 | 28.16 | 31.63 |
| σ = 25 | 20.34 | 27.79 |
| σ = 50 | 14.77 | **23.88** |

> Blind training sacrifices ~1 dB at the nominal noise level but strongly improves robustness at high noise — preferable when noise level is unknown at inference time.

---

### Exercise 2: Satellite Image Super-Resolution (RCAN-inspired)

A CNN inspired by RCAN (Residual Channel Attention Network) upscales RGB satellite images by a factor of 4.

**Architecture:** Residual Channel Attention Blocks (RCAB) with global skip connection via bicubic interpolation and pixel shuffle upsampling  
**RCAB components:** Conv → ReLU → Conv → Global Average Pool → Sigmoid (channel attention) → residual add  
**Training:** L1 loss, batch size 16, patch-based cropping

**Results:**

| Training epochs | Avg test loss | Avg PSNR (dB) |
|---|---|---|
| 50 | 0.019856 | 29.86 |
| 1000 | 0.019739 | **30.23** |

---

## Lab 2 — Point Cloud Classification with GNNs

Binary classification (chair vs. table) on the **ModelNet10** dataset subset using a Graph Convolutional Network implemented from scratch — no external graph libraries.

**Dataset:** 1,281 training samples (889 chairs / 392 tables, imbalanced) · 200 balanced test samples · 2,048 points per cloud

**Graph construction:** k-NN graph with k=20, symmetric normalized adjacency: `D⁻¹/²AD⁻¹/²`

**Architecture:**
- GCN layer 1: 3 → 64 channels + ReLU
- GCN layer 2: 64 → 64 channels + ReLU
- Global average pooling (permutation-invariant)
- FC classifier: 64 → 2 logits

**Training:** Adam (lr = 2×10⁻⁴), Cross-Entropy loss, 20 epochs, batch size 32

**Results:**

| Metric | Value |
|---|---|
| Test accuracy | **99.00%** |
| Correct predictions | 198 / 200 |
| Mean confidence (correct) | 0.8347 |
| Mean confidence (incorrect) | 0.5309 |

The ~0.30 confidence gap between correct and incorrect predictions enables uncertainty-aware thresholding. Both misclassified samples were geometrically ambiguous (chair/table proportional overlap or misleading viewpoint angles).

---

## Lab 3 — CLIP Image Search Engine

A zero-shot image retrieval system built on **CLIP ViT-B/32** over the **Flickr8k** dataset (8,091 images).

**Pipeline:**
1. Load Flickr8k and preprocess all images with CLIP's transform
2. Extract 512-dimensional feature vectors for each image
3. Store all vectors in a `(N, 512)` database matrix (float16)
4. At query time, encode text or image query and rank by cosine similarity

**Memory footprint:** `8,091 × 512 × 2 bytes ≈ 8.3 MB`

**Example results:**
- Text query `"A boat"` → top-5 images all contain watercraft (CosSim: 0.2719–0.2876)
- Image-to-image query → top-5 results share scene, subject, and lighting (CosSim: 0.8466–1.0000)

CLIP's shared embedding space enables semantic retrieval without any task-specific fine-tuning.

---

## Lab 4 — Stable Diffusion

Exploration of **Stable Diffusion** for text-to-image generation, covering key hyperparameters, architecture internals, and the image-to-image pipeline.

### Experiments

**Text-to-Image generation** — prompt: `"a horse that is running in the far west"`  
Effect of `num_inference_steps` (10 / 25 / 50 / 75): more steps progressively refine details at the cost of compute.

**Model components:**
- `pipe.unet` — the denoising backbone; receives noisy latents + text conditioning and predicts the noise to subtract at each step
- `pipe.text_encoder` — CLIP-based encoder that maps the text prompt to embeddings guiding the UNet

**Diffusion process visualization** — a callback captures intermediate latents every 5 steps, decodes them to RGB, and shows the gradual transition from random noise to a coherent image.

**Classifier-Free Guidance (CFG):**

| `guidance_scale` | Effect |
|---|---|
| 1.0 | Weak prompt adherence, smoother/more ambiguous output |
| 7.5 | Default trade-off — good fidelity and quality |
| 20.0 | Strong prompt adherence, risk of artifacts/oversaturation |

**Image-to-Image Translation** — horse painting → zebra via `strength` parameter (GPU memory constraints prevented full execution in this environment).

---

## 🚀 Getting Started

### Requirements

```bash
pip install torch torchvision numpy matplotlib scikit-learn
pip install open-clip-torch diffusers transformers accelerate
```

### Repository structure (suggested)

```
├── lab1_denoising/
│   ├── dncnn.py              # DnCNN model
│   ├── train_fixed.py        # Fixed-sigma training
│   └── train_blind.py        # Blind training
├── lab1_superresolution/
│   └── rcan.py               # RCAN-inspired super-resolution model
├── lab2_pointcloud/
│   ├── dataset.py            # ModelNet10 dataloader
│   ├── gcn.py                # GCN layers from scratch
│   └── train.py
├── lab3_clip/
│   └── search_engine.py      # CLIP feature extraction + retrieval
├── lab4_diffusion/
│   └── stable_diffusion.py   # Inference, CFG, img2img experiments
└── README.md
```

---

## 📄 Citation

```
L. Cavallaro, E. Cumino, "Advanced Machine Learning for Image Processing — Laboratory Reports",
Politecnico di Torino, A.Y. 2025/2026.
```
