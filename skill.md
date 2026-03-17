# Skills & Technologies

## Machine Learning / Deep Learning

- **Generative Adversarial Networks (GANs)** — Conditional WGAN-GP architecture with gradient penalty for stable training
- **Physics-Informed Machine Learning** — Custom monotonic distance-attenuation penalty encoding seismological prior knowledge into the loss function
- **PyTorch** — Model definition (nn.Module), custom training loop, autograd for gradient penalty computation, GPU acceleration
- **Residual Networks** — Pre-activation residual blocks with LayerNorm for both Generator and Critic
- **Learned Embeddings** — Shared period embedding MLP mapping continuous spectral period to a higher-dimensional representation

## Data Engineering

- **Pandas** — Loading, cleaning, and reshaping (~10K records x 48 columns) from wide to long format (255K samples)
- **Feature Engineering** — Log transforms (Rrup, Vs30, Period, SA), PGA replacement for log-domain compatibility
- **scikit-learn** — StandardScaler for feature normalization, train/test splitting, regression metrics (RMSE, MAE)
- **Serialization** — Model checkpointing with `torch.save`, scaler persistence with `joblib`

## Domain Knowledge

- **Earthquake Engineering** — Ground Motion Models, Spectral Acceleration, NGA-Subduction database
- **Seismological Parameters** — Moment magnitude (Mw), rupture distance (Rrup), depth to top of rupture (Ztor), site shear-wave velocity (Vs30)
- **Physical Constraints** — Distance-attenuation relationship (SA decreases with increasing Rrup)
- **Response Spectra Analysis** — Per-event evaluation across 25 spectral periods (PGA to T=10s)

## Evaluation & Visualization

- **Matplotlib** — Loss curves, real-vs-predicted scatter plots, residual analysis, per-event response spectra
- **Diagnostic Plots** — Residuals vs spectral period for period-dependent bias detection
- **Regression Metrics** — RMSE and MAE on held-out test set in log(SA) space

## Development Environment

- **Google Colab** — Primary development environment with CUDA GPU
- **Jupyter Notebooks** — Iterative experimentation with inline visualization
- **Git / GitHub** — Version control and project hosting
