from .config import get_paths
from .bgs.bgs_rgrs import BGS_RGRS
from .bgs.bgs_rgrs_paper import BGS_RGRS_PAPER, BGS_PAPER_BASE
from .bps.bps_rgrs import BPS_RGRS
from .bps.bps_rgrs_paper import BPS_RGRS_PAPER, BPS_PAPER_BASE
from .comparison import compare_and_visualize, print_statistical_summary
import os
import numpy as np

if __name__ == "__main__":
    paths_dict = get_paths()

    projects = [
        BGS_RGRS(**paths_dict["bgs_rgrs"]),
        BGS_RGRS_PAPER(**paths_dict["bgs_rgrs_paper"]),
        BPS_RGRS(**paths_dict["bps_rgrs"]),
        BPS_RGRS_PAPER(**paths_dict["bps_rgrs_paper"]),
    ]
    results = {}
    for project in projects:
        print(f"\n>>> Running {project.__class__.__name__}")
        project.full_pipeline()
        results[project.__class__.__name__] = getattr(project, 'analyze_results', {})




    # # Run base paper approaches for BGS and BPS
    # print("\n>>> Running BGS_PAPER_BASE (publication approach)")
    # bgs_paper = BGS_PAPER_BASE(wvec)
    # # Example: generate synthetic parameters and data for demo
    # bgs_parmat = np.random.uniform([30, 20, 10], [50, 35, 15], (100, 3))
    # X_bgs, y_bgs = bgs_paper.data_batchgen(bgs_parmat)
    # bgs_paper.create_model()
    # bgs_paper.train_model(X=X_bgs, y=y_bgs)
    # # Simulate real data for prediction (replace with your real data)
    # real_bgs_data = X_bgs[0, :, 1].reshape(-1, 1)  # Example: use one synthetic sample
    # par_mean_bgs, par_std_bgs = bgs_paper.batch_predict(real_bgs_data)
    # results['BGS_PAPER_BASE'] = {'bfs': par_mean_bgs[:, 0], 'fwhm': par_mean_bgs[:, 1]}

    # print("\n>>> Running BPS_PAPER_BASE (publication approach)")
    # bps_paper = BPS_PAPER_BASE(wvec)
    # bps_parmat = np.random.uniform([30, 20, 10], [50, 35, 15], (100, 3))
    # X_bps, y_bps = bps_paper.data_batchgen(bps_parmat)
    # bps_paper.create_model()
    # bps_paper.train_model(X=X_bps, y=y_bps)
    # real_bps_data = X_bps[0, :, 1]
    # par_mean_bps, par_std_bps = bps_paper.batch_predict(real_bps_data)
    # results['BPS_PAPER_BASE'] = {'bfs': par_mean_bps[:, 0]}

    # # Create results directory if not exists
    # save_dir = os.path.join(os.getcwd(), 'comparison_results')
    # os.makedirs(save_dir, exist_ok=True)

    # # Unified visualization and statistical summary
    # compare_and_visualize(results, save_dir=save_dir, show=True)
    # all_keys = set()
    # for res in results.values():
    #     all_keys.update(res.keys())
    # print_statistical_summary(results, list(all_keys))
