"""
Main pipeline orchestrator for BOTDA Deep Learning analysis.
Runs modern and legacy approaches and saves analysis artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle

import numpy as np

from botda_dl.utils import get_paths
from botda_dl.models import BGSRegression, BPSRegression
from botda_dl.legacy.config import get_paths as get_legacy_paths
from botda_dl.legacy.bgs.bgs_rgrs import BGS_RGRS
from botda_dl.legacy.bgs.bgs_rgrs_paper import BGS_RGRS_PAPER
from botda_dl.legacy.bps.bps_rgrs import BPS_RGRS
from botda_dl.legacy.bps.bps_rgrs_paper import BPS_RGRS_PAPER


def _serialize_for_json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.int32, np.int64)):
        return int(value)
    if isinstance(value, dict):
        return {k: _serialize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_serialize_for_json(v) for v in value]
    return value


def _save_analysis_results(results, results_dir: Path, name: str):
    if results is None:
        return
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / f"{name}_analysis.pkl", "wb") as f:
        pickle.dump(results, f)
    with open(results_dir / f"{name}_analysis.json", "w", encoding="utf-8") as f:
        json.dump(_serialize_for_json(results), f, indent=2)


def run_bgs_analysis(n_synthetic=300000, epochs=128, batch_size=2048, data_path=None):
    """
    Run BGS (Brillouin Gain Spectrum) Regression analysis.
    
    Args:
        n_synthetic: Number of synthetic samples
        epochs: Training epochs
        batch_size: Batch size
        data_path: Path to BGS data file
    """
    print("\n" + "="*70)
    print("BGS REGRESSION ANALYSIS")
    print("="*70)
    
    paths = get_paths("bgs_regression")
    
    if data_path is None:
        data_path = paths["data_dir"] / "BGS.txt"
    
    bgs = BGSRegression(
        data_path=data_path,
        model_path=paths["model_path"],
        log_dir=paths["log_dir"],
        results_dir=paths["results_dir"],
        scalers_dir=paths["scalers_dir"]
    )
    
    bgs.full_pipeline(n_synthetic=n_synthetic, epochs=epochs, batch_size=batch_size)
    _save_analysis_results(bgs.analyze_results, Path(paths["results_dir"]), "bgs_regression")
    
    return bgs


def run_bps_analysis(n_synthetic=300000, epochs=128, batch_size=2048, data_path=None):
    """
    Run BPS (Brillouin Phase Spectrum) Regression analysis.
    
    Args:
        n_synthetic: Number of synthetic samples
        epochs: Training epochs
        batch_size: Batch size
        data_path: Path to BPS data file
    """
    print("\n" + "="*70)
    print("BPS REGRESSION ANALYSIS")
    print("="*70)
    
    paths = get_paths("bps_regression")
    
    if data_path is None:
        data_path = paths["data_dir"] / "BPS.txt"
    
    bps = BPSRegression(
        data_path=data_path,
        model_path=paths["model_path"],
        log_dir=paths["log_dir"],
        results_dir=paths["results_dir"],
        scalers_dir=paths["scalers_dir"]
    )
    
    bps.full_pipeline(n_synthetic=n_synthetic, epochs=epochs, batch_size=batch_size)
    _save_analysis_results(bps.analyze_results, Path(paths["results_dir"]), "bps_regression")
    
    return bps


def run_legacy_analyses(n_synthetic=300000, epochs=32, batch_size=512):
    """Run legacy regression and paper baselines."""
    print("\n" + "="*70)
    print("LEGACY PIPELINE (REGRESSION + PAPER BASELINES)")
    print("="*70)

    paths_dict = get_legacy_paths()
    projects = [
        ("bgs_rgrs", BGS_RGRS(**paths_dict["bgs_rgrs"])),
        ("bgs_rgrs_paper", BGS_RGRS_PAPER(**paths_dict["bgs_rgrs_paper"])),
        ("bps_rgrs", BPS_RGRS(**paths_dict["bps_rgrs"])),
        ("bps_rgrs_paper", BPS_RGRS_PAPER(**paths_dict["bps_rgrs_paper"])),
    ]

    results = {}
    for name, project in projects:
        print(f"\n>>> Running {project.__class__.__name__}")
        project.full_pipeline(n_synthetic=n_synthetic, epochs=epochs, batch_size=batch_size)
        results[name] = getattr(project, "analyze_results", {})
        results_dir = Path(project.results_dir)
        _save_analysis_results(results[name], results_dir, name)

    return results


def run_all_analyses(n_synthetic=300000, epochs=128, batch_size=2048, run_legacy=True):
    """Run all analysis pipelines."""
    print("\n" + "="*70)
    print("BOTDA_DL - COMPLETE PIPELINE")
    print("="*70)
    
    analyses = {}
    
    try:
        analyses['bgs'] = run_bgs_analysis(n_synthetic, epochs, batch_size)
    except FileNotFoundError as e:
        print(f"Warning: BGS analysis skipped - {e}")
    
    try:
        analyses['bps'] = run_bps_analysis(n_synthetic, epochs, batch_size)
    except FileNotFoundError as e:
        print(f"Warning: BPS analysis skipped - {e}")

    if run_legacy:
        analyses["legacy"] = run_legacy_analyses(
            n_synthetic=min(n_synthetic, 100000),
            epochs=min(epochs, 64),
            batch_size=min(batch_size, 1024),
        )
    
    print("\n" + "="*70)
    print("ALL ANALYSES COMPLETED")
    print("="*70)
    
    return analyses


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BOTDA_DL pipelines.")
    parser.add_argument("--n-synthetic", type=int, default=300000)
    parser.add_argument("--epochs", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--skip-legacy", action="store_true")
    args = parser.parse_args()

    analyses = run_all_analyses(
        n_synthetic=args.n_synthetic,
        epochs=args.epochs,
        batch_size=args.batch_size,
        run_legacy=not args.skip_legacy,
    )
