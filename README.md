# BOTDA_DL

Deep-learning and classical pipelines for Brillouin Optical Time-Domain Analysis (BOTDA), covering BGS/BPS regression, legacy baselines, and probabilistic heads. The codebase keeps modern experiments clean while preserving paper-era implementations for side-by-side comparison.

## Highlights

- BGS regression with dual outputs (peak and FWHM)
- BPS regression with peak estimation and classification variants
- Legacy baselines retained for reproducibility
- PDNN experiments and curated notebooks
- Synthetic data generation workflows (modern + legacy)

## Repository layout

```
BOTDA_DL/
├── src/botda_dl/              # Core package
│   ├── core/                  # Base analyzer (modern framework)
│   ├── models/                # BGS/BPS regression models
│   ├── utils/                 # Paths + constants
│   └── legacy/                # Paper baselines (kept for comparison)
├── notebooks/                 # Curated notebooks (analysis + PDNN + BGS/BPS)
├── scripts/                   # Standalone scripts (PDNN paper)
├── data/                      # Input data (BGS.txt, BPS.txt)
│   └── raw/                   # Raw data references
├── results/                   # Generated results (created at runtime)
├── logs/                      # Training logs (created at runtime)
└── models/                    # Trained models/scalers (created at runtime)
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
PYTHONPATH=src python main.py
```

By default, this runs modern and legacy pipelines and writes artifacts to `results/`.

### Skip legacy (faster runs)

```bash
PYTHONPATH=src python main.py --skip-legacy
```

## PDNN

The PDNN paper implementation lives in `scripts/pdnn_paper.py` and requires extra dependencies:

```bash
pip install -r requirements-pdnn.txt
```

## Notebooks

See `notebooks/README.md` for the curated list of analysis and experiment notebooks.

## Outputs

- Models, results, logs, and scalers are generated at runtime and not committed.
- Analysis summaries are stored as `*_analysis.json` and `*_analysis.pkl` per run.

## Notes

- The modern framework lives under `src/botda_dl`.
- Legacy pipelines preserve original paper behavior for comparison.
