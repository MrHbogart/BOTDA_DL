# BOTDA_DL

BOTDA deep-learning pipeline for BGS/BPS regression, classical fitting, and probabilistic heads (PDNN). The repo keeps the modern framework clean while preserving legacy baselines for comparison.

## Highlights

- BGS regression with dual outputs (peak + FWHM)
- BPS regression with peak estimation
- Legacy paper baselines for side-by-side comparison
- PDNN experiments and curated notebooks
- Synthetic data generation workflows (modern + legacy)

## Project Layout

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

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
PYTHONPATH=src python main.py
```

By default, this runs modern + legacy pipelines and writes artifacts to `results/`.

### Skip legacy (faster runs)

```bash
PYTHONPATH=src python main.py --skip-legacy
```

## PDNN

The PDNN paper implementation lives in `scripts/pdnn_paper.py` and needs extra dependencies:

```bash
pip install -r requirements-pdnn.txt
```

## Notebooks

- `notebooks/analysis/` - consolidated analysis
- `notebooks/pdnn/` - probabilistic PDNN workflows
- `notebooks/bgs/` - BGS regression notebooks
- `notebooks/bps/` - BPS regression + classification notebooks

## Data

- `data/BGS.txt`, `data/BPS.txt` are the primary inputs.
- `data/raw/` contains raw reference files used in notebooks.

## Outputs

- Models, results, logs, and scalers are generated at runtime and not committed.
- Analysis summaries are stored as `*_analysis.json` and `*_analysis.pkl` per run.

## Notes

- The modern framework lives under `src/botda_dl`.
- Legacy pipelines preserve original paper behavior for comparison.
