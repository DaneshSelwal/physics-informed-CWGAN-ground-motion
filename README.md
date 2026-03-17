# Physics-Informed CWGAN-GP for Earthquake Ground Motion Prediction

A **Conditional Wasserstein GAN with Gradient Penalty (CWGAN-GP)** that learns to predict earthquake **Spectral Acceleration (SA)** from seismological parameters, augmented with a **physics-informed monotonic distance-attenuation penalty**.

This is **Phase 1** of a multi-phase research project developed in collaboration with **IIT Mandi**.

## Overview

Ground Motion Models (GMMs) are essential in seismic hazard analysis for predicting the intensity of earthquake shaking at a site. This project replaces traditional empirical GMMs with a generative model that captures the full conditional distribution of Spectral Acceleration.

**Generator input:** earthquake metadata `(Mw, Rrup, Ztor, Vs30)` + spectral period `T` + latent noise `z`
**Generator output:** predicted `log(SA)` (natural log of Spectral Acceleration)

### Key Design Choices

- **WGAN-GP training** with gradient penalty (lambda_GP = 10) and 5:1 critic-to-generator update ratio for stable training
- **Physics-informed monotonic penalty** (lambda_mono = 10) enforces the physical constraint that SA must decrease with increasing rupture distance (Rrup)
- **Period Embedding MLP** — a shared 2-layer network maps the 1D log(Period) to a 16-dimensional representation used by both Generator and Critic
- **Residual blocks with LayerNorm** — both Generator and Critic use pre-activation residual blocks (2 x ResBlock(128))

## Architecture

![Architecture](Phase1_Architecture.png)

| Component  | Parameters |
|------------|-----------|
| Generator  | 74,545    |
| Critic     | 70,577    |

## Dataset

**NGA-Subduction (NGA-Sub)** — a widely used strong-motion database in earthquake engineering.

- **Source:** `nga_subduction.xlsx` (10,239 records x 48 columns)
- **Conditioning variables:** Mw (magnitude), Rrup (rupture distance), Ztor (depth to top of rupture), Vs30 (site shear-wave velocity)
- **Target:** Spectral Acceleration at 25 periods from PGA (T=0) to T=10s
- **Working dataset:** 255,975 samples after melting to long format (one row per record-period pair)
- **Train/Test split:** 80/20 (204,780 / 51,195 samples)

## Results

Training for 100 epochs with batch size 2048, Adam optimizer (lr=1e-4, betas=(0.5, 0.9)).

| Loss Curves | Real vs Predicted |
|:-----------:|:-----------------:|
| ![Loss Curves](loss_curves.png) | ![Real vs Predicted](real_vs_pred.png) |

| Residuals vs Period | Response Spectra (Single Event) |
|:-------------------:|:-------------------------------:|
| ![Residuals](residuals_vs_period.png) | ![Response Spectra](response_spectra_event.png) |

The monotonicity penalty converges to near zero (~0.0001), confirming the physics constraint is well-satisfied.

## Project Structure

```
.
├── Phase1_CWGAN_GP.ipynb        # Main notebook (current)
├── Phase1_CWGAN_GP_old.ipynb    # Earlier version (no train/test split)
├── Phase1_Architecture.png      # Model architecture diagram
├── nga_subduction.xlsx           # NGA-Sub earthquake dataset
├── global_G.pth                  # Trained Generator weights
├── global_D.pth                  # Trained Critic weights
├── condition_scaler.pkl          # Fitted StandardScaler for conditioning features
├── loss_curves.png               # Training loss plots
├── real_vs_pred.png              # Test-set scatter plot
├── residuals_vs_period.png       # Residuals vs spectral period
├── response_spectra_event.png    # Per-event response spectra comparison
└── old/                          # Artifacts from a previous run
```

## Getting Started

### Prerequisites

```
torch
numpy
pandas
matplotlib
scikit-learn
joblib
openpyxl
```

### Running on Google Colab (recommended)

1. Upload the project folder to Google Drive under `MyDrive/WGAN_IITM/`
2. Open `Phase1_CWGAN_GP.ipynb` in Colab
3. The notebook auto-detects Colab and mounts Google Drive
4. Run all cells sequentially

### Running Locally

```bash
pip install torch numpy pandas matplotlib scikit-learn joblib openpyxl
jupyter notebook Phase1_CWGAN_GP.ipynb
```

A CUDA GPU is recommended but not required — the models are small (~75K and ~71K parameters).

## Saved Artifacts

After training completes, the following artifacts are saved for downstream use (Phase 2):

| File | Description |
|------|-------------|
| `global_G.pth` | Generator state dict |
| `global_D.pth` | Critic state dict |
| `condition_scaler.pkl` | Fitted StandardScaler for the 5 conditioning features |

## License

This project is part of ongoing research. Please contact the authors before using the code or data for publications.
