
# Gait Forecasting Research Framework

A research-oriented Python framework for locomotion / gait forecasting from EMG-derived representations.

## Preserved functionality

The project keeps the original MVP features:

- data loading
- preprocessing
- NMF synergy extraction
- H(t)
- dH/dt
- Random Forest baseline
- MLP baseline
- forecasting horizons

## Upgraded architecture

EMG → EMG Conditioning → Window Construction → NMF → Muscle Synergies → Synergy Dynamics → Latent Motor State → State Space Model → Forecasting Layer → Future Phase Prediction

### Added capabilities

- automatic dataset discovery
- support for:
  - `cycle_manifest.csv`
  - `sample_level_dataset.csv`
  - `phase2_labels.csv`
  - `phase4_labels.csv`
  - `phase7_labels.csv`
  - `cycle_nmf_input/*.csv`
- automatic detection of:
  - EMG columns
  - subject IDs
  - cycle IDs
  - gait percent
  - labels
- overlapping windows
- horizons:
  - 50 ms
  - 100 ms
  - 150 ms
  - 200 ms
  - 300 ms
- temporal models:
  - GRU
  - BiLSTM
  - TCN
- latent motor state:
  - `z(t) = [H, dH, d²H]`
  - PCA latent state
  - autoencoder latent state
- linear state-space model:
  - `x(k+1) = A x(k) + B u(k)`
- cross-subject evaluation:
  - Leave-One-Subject-Out
  - Group K-Fold
- deployment analysis:
  - parameters
  - memory
  - latency
  - FLOPs
- publication-ready outputs:
  - figures
  - tables
  - CSV summaries
  - JSON logs

## Quick start

```bash
pip install -r requirements.txt
python -m gait_forecasting --data_dir ./data --output_dir ./outputs --demo --cv loso
```

## Output artifacts

The framework writes:

- `results.json`
- `summary.json`
- `final_summary.json`
- `metrics_summary.csv`
- `deployment_summary.csv`
- `metrics.md`
- `deployment.md`
- comparison figures under fold/window/horizon folders

## Notes

- UMAP plots are generated when `umap-learn` is installed. If it is missing, the code falls back to a 2D projection so the pipeline still runs.
- The code is modular and can be extended without breaking the existing MVP API.


## Research validation extensions

The pipeline now also produces:

- synergy cosine similarity matrices
- bootstrap NMF stability analysis
- phase-conditioned H(t)
- transition-importance summaries
- SHAP-based feature attribution
- subject-adaptation curves
- hardware deployment simulations
- statistical comparison tables
- paper-ready CSV / Markdown / LaTeX exports
